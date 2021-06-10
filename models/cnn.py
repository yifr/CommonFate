import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
from . import loss

class ShapeNet(nn.Module):
    def __init__(self, img_size=256, out_size=5, conv_dims=2):
        super(ShapeNet, self).__init__()
        self.img_size = img_size
        self.out_size = out_size
        self.feature_extractor = nn.Sequential(
            self.conv_layer(1, 32, conv_dims),
            self.conv_layer(32, 64, conv_dims),
        )

        conv_out = int(img_size / 4 - 2) ** 2
        if conv_dims > 2:
            n_frames = 20
            conv_out *= int(n_frames / 4 - 2)

        self.fc1 = nn.Linear(64 * conv_out, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, out_size)
        self._transforms = T.Compose([T.Resize(img_size), T.ToTensor()])

    def conv_layer(self, in_channels, out_channels, dims):
        if dims == 2:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3)),
                nn.MaxPool3d((2, 2, 2)),
                nn.ReLU(),
            )
        return layer

    def get_transforms(self):
        return self._transforms

    def forward(self, x, transform=True):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        x = torch.mean(x, dim=0)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        if transform:
            x = torch.sigmoid(x) * 4

        x = x.reshape(-1, 2)
        return x

    def inverse_transform(self, shape_params, eps=1e-4):
        """
        Transform shape parameters (defined in the range [0, 4]) to
        model space defined between -1 and 1. Do so by taking inverse
        of sigmoid and dividing by 4
        """
        out = shape_params / 4
        out = torch.clamp(out, eps, 1 - eps)
        out = torch.log(out / (1 - out))
        return out

    def shape_transform(self, model_params):
        return F.sigmoid(model_params) * 4

    def get_shape_dist(self, predicted_shape_dist):
        predicted_mean = predicted_shape_dist[..., 0]
        predicted_std = torch.clamp(predicted_shape_dist[..., 1], min=0.0001)
        return torch.distributions.Independent(
            torch.distributions.Normal(predicted_mean, predicted_std),
            reinterpreted_batch_ndims=1,
        )

    def prob_loss(self, gt_shape, predicted_shape_mean):
        # gt_shape = self.inverse_transform(gt_shape)
        loss = -self.get_shape_dist(predicted_shape_mean).log_prob(gt_shape).mean()
        return loss


class SimpleCNN(nn.Module):
    """
    4 Layer feedforward CNN with dropout after convolutional layers
    """

    def __init__(
        self, img_size=256, out_size=6, loss=loss.PoseLoss(break_symmetry=True)
    ):
        super(SimpleCNN, self).__init__()
        self.img_size = img_size
        self._loss = loss
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        conv_out = int(img_size / 4 - 2)
        self.fc1 = nn.Linear(40 * conv_out * conv_out, 50)
        self.fc2 = nn.Linear(50, out_size)

        self._transforms = T.Compose([T.Resize(img_size), T.ToTensor()])

    def get_transforms(self):
        return self._transforms

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

        x = torch.mean(x, dim=0)
        x = torch.sigmoid(x) * 4

        return x

    def loss(self, pred, gt):
        return self._loss.compute_loss(pred, gt)

    @property
    def device(self):
        return next(self.parameters()).device

    def __str__(self):
        return "SimpleCNN"


class ResNet(nn.Module):
    def __init__(
        self, pretrained=True, out_size=6, loss=loss.PoseLoss(break_symmetry=True)
    ):
        super(ResNet, self).__init__()
        self._model = models.resnet18(pretrained=pretrained)
        self._transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Resize(256),
                T.CenterCrop(224),
            ]
        )
        # Replace model fc head
        n_inputs = self._model.fc.in_features
        self._model.fc = nn.Sequential(
            nn.Linear(n_inputs, 100), nn.ReLU(), nn.Linear(100, out_size)
        )

        self._loss = loss

    def get_transforms(self):
        return self._transforms

    def forward(self, x):
        out = self._model(x)
        return out

    def loss(self, pred, gt):
        return self._loss.compute_loss(pred, gt)

    def load_trained(self, trained_model, device="cuda"):
        print(f"Loading trained weights from {trained_model}")
        state_dict = torch.load(trained_model, map_location=torch.device(device))
        # match up state dict keys with model keys:
        state_dict = {k.partition("_model.")[2]: v for k, v in state_dict.items()}
        res = self._model.load_state_dict(state_dict)
        print("Result: ", res)
        return res

    @property
    def device(self):
        return next(self.parameters()).device

    def __str__(self):
        return "ResNet"


class ResNet1ch(models.resnet.ResNet):
    """
    Overrides ResNet architecture for a 1 channel input
    """

    def __init__(self, block, layers, num_classes=4):
        super(ResNet, self).__init__(block, layers, num_classes=4)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1, bias=False)
