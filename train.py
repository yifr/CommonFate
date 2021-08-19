import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms as T
from models import cnn, vaes
from data_loader import SceneLoader
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_shapenet(args, model, loss_fn, scene_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    optimizer.zero_grad()
    for batch_idx, scene_idx in enumerate(tqdm(scene_loader.train_idxs)):
        data = scene_loader.get_scene(scene_idx)
        frames = data["frame"]
        if args.conv_dims > 2:
            frames = frames.reshape(1, args.n_frames, 256, 256).unsqueeze(0)

        predicted_shape = model(frames)
        gt_shape = data["shape_params"].mean(axis=0)

        losses = loss_fn(predicted_shape, gt_shape) 
        running_loss += losses['loss'].item()

        loss = losses['loss'] / args.minibatch_size # Normalize loss for gradient accumulation
        loss.backward()

        if (batch_idx + 1) % args.minibatch_size == 0 or batch_idx == (len(scene_loader.train_idxs) - 1):
            optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % args.log_interval == 0:
            frames_completed = batch_idx * args.n_frames
            total_frames = len(scene_loader.train_idxs) * args.n_frames
            percent_complete = 100 * batch_idx / len(scene_loader.train_idxs)

            print(
                f"Train Epoch: {epoch} [{frames_completed}/{total_frames} frames ({percent_complete:.0f}%)] \
                \tLoss: {running_loss / (batch_idx + 1):.4f}", flush=True
            )

    wandb.log({"Train Loss": running_loss / batch_idx, "epoch": epoch})
    running_loss = 0

def test_shapenet(args, model, loss_fn, scene_loader, epoch, best_test_score=10000):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for _, scene_idx in enumerate(scene_loader.test_idxs):
            data = scene_loader.get_scene(scene_idx)
            frames = data["frame"].to(args.device)
            if args.conv_dims > 2:
                frames = frames.reshape(1, args.n_frames, 256, 256).unsqueeze(0)

            predicted_shape = model(frames)
            gt_shape = data["shape_params"].mean(axis=0)

            loss = loss_fn(predicted_shape, gt_shape)
            test_loss += loss

    test_loss /= len(scene_loader.test_idxs)

    print(f"\nTest set: Average loss: {test_loss:.4f}\n")

    wandb.log({"Test Loss": test_loss, "epoch": epoch})

    if test_loss < best_test_score:
        torch.save(model.state_dict(), os.path.join(args.model_save_path, args.model_save_name))
        best_test_score = test_loss

    return best_test_score

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="CommonFate State Inference")
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training",
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="scenes/data",
        nargs="+",
        metavar="DIR",
        help="directory in which to look for scenes",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="shapenet",
        metavar="NAME",
        help="name to log run with in wandb",
    )
    parser.add_argument(
        "--n_scenes",
        type=int,
        default=1000,
        metavar="N",
        help="Total number of scenes to train on",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=20,
        metavar="N",
        help="Total number of frames per scene",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="saved_models/",
        metavar="P",
        help="path to save model",
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default="shapenet.pt",
        help="Name of model")

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        metavar="M",
        help="L2 norm (default: 0.01)",
    )
    parser.add_argument(
        "--device", 
        type=str,
        default="cuda", 
        help="cuda or cpu"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--pred_shape",
        type=int,
        default=2,
        metavar="P",
        help="Output size of neural network",
    )
    parser.add_argument(
        "--conv_dims", type=int, default=3, metavar="D", help="2D or 3D conv net"
    )
    parser.add_argument(
        "--load_existing",
        action="store_true",
        help="Begin training from an existing model",
    )

    args = parser.parse_args()
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # model = cnn.ShapeNet(out_size=args.pred_shape, conv_dims=args.conv_dims)
		model = vaes.VAE(128, 64)
    transforms = model.get_transforms()
    loss_fn = model.loss

    model = nn.DataParallel(model)
    model.to(args.device)

    os.makedirs(args.model_save_path, exist_ok=True)

    if args.load_existing:
        model.load_state_dict(torch.load(args.model_save_path))

    scene_loader = SceneLoader(
        root_dirs=args.scene_dir,
        n_scenes=args.n_scenes,
        n_frames=args.n_frames,
        img_size=128,
        device=args.device,
        as_rgb=False,
        transforms=transforms,
        seed=args.seed,
    )
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Start logging process
    resume_wandb = True if args.load_existing else False
    wandb.init(settings=wandb.Settings(start_method="fork"), resume=resume_wandb)
    if args.run_name:
        wandb.run.name = args.run_name
    wandb.config.update(args)
    wandb.watch(model)

    print("Initialized model and data loader, beginning training...")
    best_test = 100000
    for epoch in range(1, args.epochs + 1):
        train_shapenet(args, model, loss_fn, scene_loader, optimizer, epoch)
        best_test = test_shapenet(args, model, loss_fn, scene_loader, epoch, best_test)


if __name__ == "__main__":
    main()
