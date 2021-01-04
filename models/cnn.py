import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
        pred_rmat = self.rotation_matrix_from_6d(pred)

        mse = F.mse(gt_mat, pred_mat)
        geodesic = torch.mean(self.geodesic_dist(gt_mat, pred_mat))

        return {'mse': mse, 'geodesic': geodesic}

    def geodesic_dist(self, m1, m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2)) # batch * 3 * 3

        cos = ( m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1  ) / 2
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )

        theta = torch.acos(cos)

        return theta

    def rotation_matrix_from_6d(self, ortho6d):
        """
        Computes rotation matrix from 6d representation. 
        Implements eq. (15) from: https://arxiv.org/pdf/1812.07035.pdf
        """
        x = F.normalize(ortho6d[:, :3], p=2, dim=1) # batch * 3
        y_raw = ortho6d[:, 3:]
        y = F.normalize(y_raw - torch.dot(x, y_raw) * x, p=2, dim=1) # batch * 3 
        z = torch.cross(x, y, dim=1)

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        m = torch.cat((x, y, z), 2) # batch * 3 * 3

        return m

    def rotation_matrix_from_quaternion(self, q):
        """
        Convert rotations given as quaternions to rotation matrices. 
        Source: https://www.weizmann.ac.il/sci-tea/benari/sites/sci-tea.benari/files/uploads/softwareAndLearningMaterials/quaternion-tutorial-2-0-1.pdf
                https://github.com/moble/quaternion/blob/master/src/quaternion/__init__.py
        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).
        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
    n = F.normalize(q, p=2, dim=1)
    m = torch.empty(q.shape + (3, 3))

    m[..., 0, 0] = 1.0 - 2*(q[..., 2]**2 + q[..., 3]**2)/n
    m[..., 0, 1] = 2*(q[..., 1]*q[..., 2] - q[..., 3]*q[..., 0])/n
    m[..., 0, 2] = 2*(q[..., 1]*q[..., 3] + q[..., 2]*q[..., 0])/n
    m[..., 1, 0] = 2*(q[..., 1]*q[..., 2] + q[..., 3]*q[..., 0])/n
    m[..., 1, 1] = 1.0 - 2*(q[..., 1]**2 + q[..., 3]**2)/n
    m[..., 1, 2] = 2*(q[..., 2]*q[..., 3] - q[..., 1]*q[..., 0])/n
    m[..., 2, 0] = 2*(q[..., 1]*q[..., 3] - q[..., 2]*q[..., 0])/n
    m[..., 2, 1] = 2*(q[..., 2]*q[..., 3] + q[..., 1]*q[..., 0])/n
    m[..., 2, 2] = 1.0 - 2*(q[..., 1]**2 + q[..., 2]**2)/n

    return m

    

class SimpleCNN(nn.Module):
    """
    Predicts 4-D quaternion pose given an input image
    """
    def __init__(self, out_size=6):
        super(Net, self).__init__()
        self.loss = Loss(break_symmetry=True)
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

    def loss(self, pred, gt):
        return self.loss.compute_loss(pred, gt)

    @property
    def device(self):
        return next(self.parameters()).device


class ResNet(models.resnet.ResNet):
    """
    Overrides ResNet architecture for a 1 channel input
    """
    def __init__(self, block, layers, num_classes=4):
        super(ResNet, self).__init__(block, layers, num_classes=4)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1,
                               bias=False)
