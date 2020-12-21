import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.resnet import Bottleneck
from models import cnn
from data_loader import SceneLoader
import wandb


class Net(nn.Module):
    def __init__(self, out_size=4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(40 * 30 * 30, 50)
        self.fc2 = nn.Linear(50, out_size)

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Conv 2
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Feedforward
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def train(args, model, device, scene_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    for batch_idx, scene_idx in enumerate(scene_loader.train_idxs):
        data = scene_loader.get_scene(scene_idx)
        frames = data['frame'].to(device)
        target = data['rotation'].to(device)
        optimizer.zero_grad()
        pred = model(frames)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} frames ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(frames), len(scene_loader.train_idxs) * len(frames),
                100. * batch_idx / len(scene_loader.train_idxs), loss.item()))

    running_loss /= (len(frames) * len(scene_loader.train_idxs))
    wandb.log({"Train Loss": running_loss})

def test(args, model, device, scene_loader):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for batch_idx, scene_idx in enumerate(scene_loader.test_idxs):
            data = scene_loader.get_scene(scene_idx)
            frames = data['frame'].to(device)
            target = data['rotation'].to(device)

            output = model(frames)
            # sum up batch loss
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            """
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))
            """
    test_loss /= len(scene_loader.test_idxs) * len(frames)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    wandb.log({"Test Loss": test_loss})


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

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    scene_loader = SceneLoader(root_dir=args.scene_dir, n_scenes=args.n_scenes, device=device)
    model = cnn.ResNet(Bottleneck, [2, 2, 2, 2], num_classes=4).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    wandb.watch(model)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, scene_loader, optimizer, epoch)
        test(args, model, device, scene_loader)

    torch.save(model.state_dict(), 'model.pt')
if __name__ == '__main__':
    main()
