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
    running_mse = 0
    for batch_idx, scene_idx in enumerate(tqdm(scene_loader.train_idxs)):
        data = scene_loader.get_scene(scene_idx)
        frames = data['frame'].to(device)
        target = data['shape_params'][0, :2].to(device)

        optimizer.zero_grad()
        pred = model(frames)
        shape_pred = torch.mean(pred, dim=0)
        
        loss = F.mse_loss(shape_pred, target)
        loss.backward()
        optimizer.step()

        running_mse += loss.item()

        if batch_idx % args.log_interval == 0:
            frames_completed = batch_idx * len(frames)
            total_frames = len(scene_loader.train_idxs) * len(frames)
            percent_complete = 100 * batch_idx / len(scene_loader.train_idxs)

            print(f'Train Epoch: {epoch} [{frames_completed}/{total_frames} frames ({percent_complete:.0f}%)]\tMSE: {running_mse / (batch_idx + 1):.6f}')
        
    running_mse /=  len(scene_loader.train_idxs)
    wandb.log({"Train MSE": running_mse})

def test_shapenet(args, model, device, scene_loader):
    model.eval()
    test_loss_mse = 0

    with torch.no_grad():
        for batch_idx, scene_idx in enumerate(scene_loader.test_idxs):
            data = scene_loader.get_scene(scene_idx)
            frames = data['frame'].to(device)
            target = data['shape_params'][0, :2].to(device)

            output = model(frames)
            pred_shape = torch.mean(output, dim=0)
            loss = F.mse_loss(pred_shape, target)
            test_loss_mse += loss

    test_loss_mse /= len(scene_loader.test_idxs) 

    print('\nTest set: Average MSE: {:.4f}\n'.format(test_loss_mse, test_loss_geodesic))

    wandb.log({"Test MSE": test_loss_mse})

    torch.save(model.state_dict(), args.model_save_path)


def train(args, model, device, scene_loader, optimizer, epoch):
    model.train()
    running_mse = 0
    running_geodesic = 0
    running_chance = 0
    for batch_idx, scene_idx in enumerate(tqdm(scene_loader.train_idxs)):
        data = scene_loader.get_scene(scene_idx)
        frames = data['frame'].to(device)

        if batch_idx == 0:
            fig = plt.figure(figsize=(16,12))
            img = frames[0, 0]
            img = img.cpu().detach().numpy()
            plt.matshow(img, cmap='gray')
            plt.savefig('example_frame.png')

        target = data['rotation'].to(device)
        optimizer.zero_grad()
        pred = model(frames)

        loss_dict = model.loss(pred, target)
        mse = loss_dict['mse']
        geodesic = loss_dict['geodesic']
        chance = loss_dict['chance']

        mse.backward()
        optimizer.step()
        running_mse += mse.item()
        running_geodesic += geodesic.item()
        running_chance += chance.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} frames ({:.0f}%)]\tMSE: {:.6f}\tGeodesic: {:.6f}'.format(
                epoch, batch_idx * len(frames), len(scene_loader.train_idxs) * len(frames),
                100. * batch_idx / len(scene_loader.train_idxs), mse.item(), geodesic.item()))

    running_mse /=  len(scene_loader.train_idxs)
    running_geodesic /= len(scene_loader.train_idxs)
    running_chance /= len(scene_loader.train_idxs)

    wandb.log({"Train MSE": running_mse, "Train Geodesic": running_geodesic, "Chance MSE (train)": running_chance})

def test(args, model, device, scene_loader):
    model.eval()
    test_loss_mse = 0
    test_loss_geodesic = 0

    example_images = []
    with torch.no_grad():
        for batch_idx, scene_idx in enumerate(scene_loader.test_idxs):
            data = scene_loader.get_scene(scene_idx)
            frames = data['frame'].to(device)
            target = data['rotation'].to(device)

            output = model(frames)
            # sum up batch loss
            losses = model.loss(output, target)

            test_loss_mse += losses['mse']
            test_loss_geodesic += losses['geodesic']

    test_loss_mse /= len(scene_loader.test_idxs) 
    test_loss_geodesic /= len(scene_loader.test_idxs) 

    print('\nTest set: Average MSE: {:.4f}, Geodesic: {:.4f}\n'.format(test_loss_mse, test_loss_geodesic))

    wandb.log({"Test MSE": test_loss_mse, "Test Geodesic": test_loss_geodesic})

    torch.save(model.state_dict(), args.model_save_path)

def main():
    wandb.init()
    # Training settings
    parser = argparse.ArgumentParser(description='CommonFate State Inference')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--scene_dir', type=str, default='scenes', metavar='DIR',
                        help='directory in which to look for scenes')
    parser.add_argument('--n_scenes', type=int, default=400, metavar='N',
                        help='Total number of scenes to train on')
    parser.add_argument('--model_save_path', type=str, default='saved_models/shapenet.pt', metavar='P',
                        help='path to save model')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='M',
                        help='L2 norm (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--pred_type', type=str, default='shape', metavar='P',
                        help='Predicting either shape or rotation')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    wandb.config.update(args)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    wargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = None
    if args.pred_type == 'shape':
        model = cnn.SimpleCNN(out_size=2).to(device) #cnn.ResNet().to(device)
    else:
        model = cnn.SimpleCNN(out_size=6).to(device) #cnn.ResNet().to(device)

    scene_loader = SceneLoader(root_dir=args.scene_dir, n_scenes=args.n_scenes, img_size=256, device=device,
                               as_rgb=False, transforms=model.get_transforms())
    #model = cnn.ResNet(Bottleneck, [2, 2, 2, 2], num_classes=4).to(device)
    # model = cnn.SimpleCNN().to(device)

    print('Initialized model and data loader, beginning training...')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    wandb.watch(model)

    for epoch in range(1, args.epochs + 1):
        if args.pred_type == 'shape':
            train_shapenet(args, model, device, scene_loader, optimizer, epoch)
            test_shapenet(args, model, device, scene_loader)
        else:
            train(args, model, device, scene_loader, optimizer, epoch)
            test(args, model, device, scene_loader)


if __name__ == '__main__':
    main()
