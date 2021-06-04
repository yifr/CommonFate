import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms as T
from models import cnn
from data_loader import SceneLoader
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_shapenet(args, model, device, scene_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    for batch_idx, scene_idx in enumerate(tqdm(scene_loader.train_idxs)):
        data = scene_loader.get_scene(scene_idx)
<<<<<<< HEAD
        frames = data["frame"].to(device)
        gt_shape = data["shape_params"].mean(axis=0)

        optimizer.zero_grad()
        predicted_shape_mean = model(frames)
        # print('\nGt_shape: ', gt_shape, '\t Predicted: ', predicted_shape_mean, '\n')
=======
        frames = data['frame']
        if args.conv_dims > 2:
            frames = frames.reshape(1, 20, 256, 256).unsqueeze(0)
                
        gt_shape = data['shape_params'].mean(axis=0)
     
        optimizer.zero_grad()
        predicted_shape_dist = model(frames)
        # print('\nGt_shape: ', gt_shape, '\t Predicted: ', predicted_shape_mean, '\n') 
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1
        # mean_target = torch.mean(target, dim=0)
        # shape_pred = torch.mean(shape_pred, dim=0)
        # shape_pred = torch.sigmoid(shape_pred) * 4  # shape exponents are in the range [0, 4] -> this enforces a non-negativity

<<<<<<< HEAD
        loss = model.prob_loss(
            gt_shape, predicted_shape_mean
        )  # F.mse_loss(shape_pred, mean_target)
=======
        loss = model.prob_loss(gt_shape, predicted_shape_dist) # F.mse_loss(shape_pred, mean_target)
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            frames_completed = batch_idx * args.n_frames
            total_frames = len(scene_loader.train_idxs) * args.n_frames
            percent_complete = 100 * batch_idx / len(scene_loader.train_idxs)

<<<<<<< HEAD
            print(
                f"Train Epoch: {epoch} [{frames_completed}/{total_frames} frames ({percent_complete:.0f}%)]\tMSE: {running_mse / (batch_idx + 1):.6f}"
            )

    running_mse /= len(scene_loader.train_idxs)
    wandb.log({"Train MSE": running_mse, "epoch": epoch})

=======
            print(f'Train Epoch: {epoch} [{frames_completed}/{total_frames} frames ({percent_complete:.0f}%)]\tLoss: {running_loss / (batch_idx + 1):.6f}')

    running_loss /=  len(scene_loader.train_idxs)
    wandb.log({"Train Log Likelihood": running_loss})
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1

def test_shapenet(args, model, device, scene_loader, epoch, best_test_score=10000):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, scene_idx in enumerate(scene_loader.test_idxs):
            data = scene_loader.get_scene(scene_idx)
<<<<<<< HEAD
            frames = data["frame"].to(device)
            gt_shape = data["shape_params"]
=======
            frames = data['frame'].to(device)
            if args.conv_dims > 2:
                frames = frames.reshape(1, 20, 256, 256).unsqueeze(0)

            gt_shape = data['shape_params']
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1

            predicted_shape_dist = model(frames)

            loss = model.prob_loss(gt_shape, predicted_shape_dist)
            test_loss += loss

    test_loss /= len(scene_loader.test_idxs)

<<<<<<< HEAD
    print(f"\nTest set: Average MSE: {test_loss_mse:.4f}\n")

    wandb.log({"Test MSE": test_loss_mse, "epoch": epoch})

    if test_loss_mse < best_test_score:
        torch.save(model.state_dict(), args.model_save_path)
        best_test_score = test_loss_mse

    return best_test_score

=======
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

    wandb.log({"Test Log Likelihood": test_loss})

    if test_loss < best_test_score:
            torch.save(model.state_dict(), args.model_save_path)
            best_test_score = test_loss
    
    return best_test_score 
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1

def train_posenet(args, model, device, scene_loader, optimizer, epoch):
    model.train()
    running_mse = 0
    running_geodesic = 0
    running_chance = 0
    for batch_idx, scene_idx in enumerate(tqdm(scene_loader.train_idxs)):
        data = scene_loader.get_scene(scene_idx)
        frames = data["frame"].to(device)

        if batch_idx == 0:
            fig = plt.figure(figsize=(16, 12))
            img = frames[0, 0]
            img = img.cpu().detach().numpy()
            plt.matshow(img, cmap="gray")
            plt.savefig("example_frame.png")

        target_key = "rotation"
        if data["rotation"][0].shape == (3, 3):
            target_key = "quaternion"

        target = data[target_key].to(device)
        optimizer.zero_grad()
        pred = model(frames)

        loss_dict = model.loss(pred, target)
        mse = loss_dict["mse"]
        geodesic = loss_dict["geodesic"]
        chance = loss_dict["chance"]

        mse.backward()
        optimizer.step()
        running_mse += mse.item()
        running_geodesic += geodesic.item()
        running_chance += chance.item()

        if batch_idx % args.log_interval == 0:
<<<<<<< HEAD
            print(
                "Train Epoch: {} [{}/{} frames ({:.0f}%)]\tMSE: {:.6f}\tGeodesic: {:.6f}".format(
                    epoch,
                    batch_idx * len(frames),
                    len(scene_loader.train_idxs) * len(frames),
                    100.0 * batch_idx / len(scene_loader.train_idxs),
                    mse.item(),
                    geodesic.item(),
                )
            )

    running_mse /= len(scene_loader.train_idxs)
    running_geodesic /= len(scene_loader.train_idxs)
    running_chance /= len(scene_loader.train_idxs)

    wandb.log(
        {
            "Train MSE": running_mse,
            "Train Geodesic": running_geodesic,
            "Chance MSE (train)": running_chance,
        }
    )

=======
            print('Train Epoch: {} [{}/{} frames ({:.0f}%)]\tloss: {:.6f}\tGeodesic: {:.6f}'.format(
                epoch, batch_idx * len(frames), len(scene_loader.train_idxs) * len(frames),
                100. * batch_idx / len(scene_loader.train_idxs), mse.item(), geodesic.item()))

    running_mse /=  len(scene_loader.train_idxs)
    running_geodesic /= len(scene_loader.train_idxs)
    running_chance /= len(scene_loader.train_idxs)

    wandb.log({"Train loss": running_mse, "Train Geodesic": running_geodesic, "Chance loss (train)": running_chance})
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1

def test_posenet(args, model, device, scene_loader):
    model.eval()
    test_loss_mse = 0
    test_loss_geodesic = 0

    example_images = []
    with torch.no_grad():
        for batch_idx, scene_idx in enumerate(scene_loader.test_idxs):
            data = scene_loader.get_scene(scene_idx)
            frames = data["frame"].to(device)

            target_key = "rotation"
            if data["rotation"][0].shape == (3, 3):
                target_key = "quaternion"

            target = data[target_key].to(device)

            output = model(frames)
            # sum up batch loss
            losses = model.loss(output, target)

            test_loss_mse += losses["mse"]
            test_loss_geodesic += losses["geodesic"]

    test_loss_mse /= len(scene_loader.test_idxs)
    test_loss_geodesic /= len(scene_loader.test_idxs)

<<<<<<< HEAD
    print(
        "\nTest set: Average MSE: {:.4f}, Geodesic: {:.4f}\n".format(
            test_loss_mse, test_loss_geodesic
        )
    )
=======
    print('\nTest set: Average loss: {:.4f}, Geodesic: {:.4f}\n'.format(test_loss_mse, test_loss_geodesic))
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1

    wandb.log({"Test loss": test_loss_mse, "Test Geodesic": test_loss_geodesic})

    torch.save(model.state_dict(), args.model_save_path)

<<<<<<< HEAD

def main():
    wandb.init(settings=wandb.Settings(start_method="fork"))
=======
def train(model, 
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1

def main():
    # Training settings
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="CommonFate State Inference")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="scenes/data",
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
        "--model_save_path",
        type=str,
        default="saved_models/shapenet.pt",
        metavar="P",
        help="path to save model",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
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
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--pred_type",
        type=str,
        default="shape",
        metavar="P",
        help="Predicting either shape or rotation",
    )
=======
    parser = argparse.ArgumentParser(description='CommonFate State Inference')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--scene_dir', type=str, default='scenes/data', nargs='+',
                    metavar='DIR', help='directory in which to look for scenes')
    parser.add_argument('--run_name', type=str, default='shapenet', metavar='NAME',
                        help='name to log run with in wandb')
    parser.add_argument('--n_scenes', type=int, default=1000, metavar='N',
                        help='Total number of scenes to train on')
    parser.add_argument('--n_frames', type=int, default=20, metavar='N',
                        help='Total number of frames per scene')
    parser.add_argument('--model_save_path', type=str, default='saved_models/shapenet.pt', metavar='P',
                        help='path to save model')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='M',
                        help='L2 norm (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--pred_shape', type=int, default=4, metavar='P',
                        help='Output size of neural network')
    parser.add_argument('--conv_dims', type=int, default=3, metavar='D',
                    help='2D or 3D conv net')
    parser.add_argument('--load_existing', action='store_true',
                    help='Begin training from an existing model')
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    resume_wandb = True if args.load_existing else False
    wandb.init(settings=wandb.Settings(start_method='fork'), resume=resume_wandb)
    if args.run_name:
        wandb.run.name = args.run_name
    wandb.config.update(args)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

<<<<<<< HEAD
    wargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    out_size = 2 if args.pred_type == "shape" else 6
    model = cnn.ShapeNet(out_size=out_size).to(device)  # cnn.ResNet().to(device)
    scene_loader = SceneLoader(
        root_dir=args.scene_dir,
        n_scenes=args.n_scenes,
        img_size=256,
        device=device,
        as_rgb=False,
        transforms=model.get_transforms(),
        seed=args.seed,
    )
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
=======
    wargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = cnn.ShapeNet(out_size=args.pred_shape, conv_dims=args.conv_dims).to(device) 
    if args.load_existing:
        model.load_state_dict(torch.load(args.model_save_path))

    scene_loader = SceneLoader(root_dirs=args.scene_dir, n_scenes=args.n_scenes, 
                    n_frames=args.n_frames, img_size=256, device=device, 
                    as_rgb=False, transforms=model.get_transforms(), seed=args.seed)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1
    wandb.watch(model)

    print("Initialized model and data loader, beginning training...")

    for epoch in range(1, args.epochs + 1):
        best_test = 100000
<<<<<<< HEAD
        if args.pred_type == "shape":
            train_shapenet(args, model, device, scene_loader, optimizer, epoch)
            best_test = test_shapenet(
                args, model, device, scene_loader, epoch, best_test
            )
        else:
            train_posenet(args, model, device, scene_loader, optimizer, epoch)
            test_posenet(args, model, device, scene_loader)
=======
        train_shapenet(args, model, device, scene_loader, optimizer, epoch)
        best_test = test_shapenet(args, model, device, scene_loader, epoch, best_test)
>>>>>>> e37947e815337b9b92bf785b1b13d70a247da0c1


if __name__ == "__main__":
    main()
