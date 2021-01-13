import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T

class Loss:
    def __init__(self, break_symmetry=False):
        self.break_sym = break_symmetry

    def compute_loss(self, pred_6d, quat_gt):
        """
        Compute loss from predicted 6d rotation and ground truth quaternions
        converts both representations into rotation matrices
        if self.break_sym == True, constrains quaternion to one hemisphere

        Params
        ---------
        pred: predicted 6d representation
        quat_gt: ground truth quaternion rotation
        """
        if self.break_sym:
            quat_gt = torch.abs(quat_gt)

        gt_rmat = self.rotation_matrix_from_quaternion(quat_gt)
        pred_rmat = self.rotation_matrix_from_6d(pred_6d)

        mse = F.mse_loss(gt_rmat, pred_rmat)
        geodesic = torch.mean(self.geodesic_dist(gt_rmat.double(), pred_rmat.double()))

        mean_train = torch.mean(gt_rmat, axis=0)
        chance_mse = torch.mean((mean_train - gt_rmat) ** 2)

        return {'mse': mse, 'geodesic': geodesic, 'chance': chance_mse}

    def geodesic_dist(self, m1, m2, cos_angle=False):
        """
        Calculates angles (in radians) of a batch of rotation matrices.
        Adapted from PyTorch3d:
        https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/transforms/so3.py
        Args:
            m1, m2: rotation matrices
            cos_angle: if True, returns cosine of rotation angles instead of angle itself
        """
        m = torch.bmm(m1, m2.permute(0, 2, 1))
        batch, d1, d2 = m.shape
        if d1 != 3 or d2 != 3:
            raise ValueError('Geodesic distance only implemented for batches of 3x3 Tensors')

        rotation_trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        rotation_trace = torch.clamp(rotation_trace, -1.0, 3.0)

        # rotation angle
        phi = 0.5 * (rotation_trace - 1.0)

        if cos_angle:
            return phi
        else:
            return phi.acos()

    def rotation_matrix_from_6d(self, ortho6d):
        """
        Computes rotation matrix from 6d representation.
        Implements eq. (15) from: https://arxiv.org/pdf/1812.07035.pdf
        """
        x_raw = ortho6d[:, :3]
        y_raw = ortho6d[:, 3:]
        x = F.normalize(x_raw, p=2, dim=1) # batch * 3
        y = F.normalize(y_raw - torch.sum(x * y_raw, dim=-1, keepdim=True) * x, p=2, dim=1) # batch * 3
        z = torch.cross(x, y, dim=1)

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        m = torch.cat((x, y, z), 2) # batch * 3 * 3

        return m


    def rotation_matrix_from_quaternion(self, quaternions):
        """
        Convert rotations given as quaternions to rotation matrices.
        Source: https://www.weizmann.ac.il/sci-tea/benari/sites/sci-tea.benari/files/uploads/softwareAndLearningMaterials/quaternion-tutorial-2-0-1.pdf
                https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/transforms/rotation_conversions.py
        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).
        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),

            ),
            -1,
        )

        return o.reshape(quaternions.shape[:-1] + (3, 3))

class SimpleCNN(nn.Module):
    """
    4 Layer feedforward CNN with dropout after convolutional layers
    """
    def __init__(self, img_size=256, out_size=6):
        super(SimpleCNN, self).__init__()
        self.img_size = img_size
        self._loss = Loss(break_symmetry=True)
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        conv_out = int(img_size / 4 - 2)
        self.fc1 = nn.Linear(40 * conv_out * conv_out, 50)
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

    def loss(self, pred, gt):
        return self._loss.compute_loss(pred, gt)

    @property
    def device(self):
        return next(self.parameters()).device

class ResNet(nn.Module):
    def __init__(self, pretrained=True, out_size=6):
        super(ResNet, self).__init__()
        self._model = models.resnet18(pretrained=pretrained)
        self._transforms = T.Compose([T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      T.Resize(256),
                                      T.CenterCrop(224)]
                                    )
        # Replace model fc head
        n_inputs = self._model.fc.in_features
        self._model.fc = nn.Sequential(nn.Linear(n_inputs, 100),
                                       nn.ReLU(),
                                       nn.Linear(100, out_size))

        self._loss = Loss(break_symmetry=True)

    def get_transforms(self):
        return self._transforms

    def forward(self, x):
        out = self._model(x)
        return out

    def loss(self, pred, gt):
        return self._loss.compute_loss(pred, gt)

    @property
    def device(self):
        return next(self.parameters()).device

class ResNet1ch(models.resnet.ResNet):
    """
    Overrides ResNet architecture for a 1 channel input
    """
    def __init__(self, block, layers, num_classes=4):
        super(ResNet, self).__init__(block, layers, num_classes=4)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1,
                               bias=False)
