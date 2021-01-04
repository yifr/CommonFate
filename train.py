import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms as T
from models import cnn
from data_loader import SceneLoader
import wandb



def train(args, model, device, scene_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    running_geodesic = 0
    for batch_idx, scene_idx in enumerate(scene_loader.train_idxs):
        data = scene_loader.get_scene(scene_idx)
        frames = data['frame'].to(device)

        target = data['rotation'].to(device)
        optimizer.zero_grad()
        pred = model(frames)

        loss_dict = model.loss(pred, target)
        mse = loss_dict['mse']
        geodesic = loss_dict['geodesic']

        mse.backward()
        optimizer.step()
        running_mse += mse.item()
        running_geodesic += geodesic.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} frames ({:.0f}%)]\tMSE: {:.6f}\tGeodesic: {:.6f}'.format(
                epoch, batch_idx * len(frames), len(scene_loader.train_idxs) * len(frames),
                100. * batch_idx / len(scene_loader.train_idxs), mse.item(), geodesic.item()))

    running_mse /= (len(frames) * len(scene_loader.train_idxs))
    running_geodesic /= (len(frames) * len(scene_loader.train_idxs))
    wandb.log({"Train MSE": running_mse, "Train Geodesic": running_geodesic})

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

    test_loss_mse /= len(scene_loader.test_idxs) * len(frames)
    test_loss_geodesic /= len(scene_loader.test_idxs) * len(frames)

    print('\nTest set: Average MSE: {:.4f}, Geodesic: {:.4f}\n'.format(test_loss_mse, test_loss_geodesic))

    wandb.log({"Test MSE": test_loss_mse, "Test Geodesic": test_loss_geodesic})


def main():
    wandb.init()
    # Training settings
    parser = argparse.ArgumentParser(description='CommonFate State Inference')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--scene_dir', type=str, default='scenes', metavar='DIR',
                        help='directory in which to look for scenes')
    parser.add_argument('--n_scenes', type=int, default=142, metavar='N',
                        help='Total number of scenes to train on')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    wandb.config.update(args)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    wargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('Initialized Data Loader')
    scene_loader = SceneLoader(root_dir=args.scene_dir, n_scenes=args.n_scenes, img_size=1024, device=device)
    #model = cnn.ResNet(Bottleneck, [2, 2, 2, 2], num_classes=4).to(device)
    print('Initializing Model...')
    model = models.SimpleCNN()

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    wandb.watch(model)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, scene_loader, optimizer, epoch)
        test(args, model, device, scene_loader)

    torch.save(model.state_dict(), 'model.pt')
if __name__ == '__main__':
    main()
