import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Loss:
    def __init__(self, break_symmetry=False):
        self.break_sym = break_symmetry

    def compute_loss(self, pred, quat_gt):
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

        gt_rmat = self.rmat_from_quaternion(quat_gt)
        pred_rmat = self.rmat_from_6d(pred)

        mse = F.mse(gt_mat, pred_mat)
        geodesic = self.geodesic_dist(gt_mat, pred_mat)
        geodesic_mean = torch.mean(geodeisc)

        return {'mse': mse, 'geodesic': geodesic_mean}

    def geodesic_dist(self, m1, m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2)) # batch * 3 * 3

        cos = ( m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1  ) / 2
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )

        theta = torch.acos(cos)

        return theta

    def normalize_vector(self, v):
        batch= v.shape[0]
        v_m = torch.sqrt(v.pow(2).sum(1)) # batch
        v_m = torch.max(v_m, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_m = v_m.view(batch, 1).expand(batch, v.shape[1])
        v = v/v_m
        return v

    def cross_product(self, u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1) # batch * 3

        return out

    def rmat_from_6d(self, ortho6d):
        """
        Computes rotation matrix from 6d representation
        """
        x_raw = ortho6d[:, 0:3] # batch * 3
        y_raw = ortho6d[:, 3:6] # batch * 3

        x = self.normalize_vector(x_raw) # batch *3
        z = self.cross_product(x,y_raw) # batch * 3
        z = self.normalize_vector(z) #batch * 3
        y = self.cross_product(z, x) # batch * 3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrix = torch.cat((x, y, z), 2) # batch * 3 * 3

        return matrix

    def rmat_from_quaternion(self, quaternion):
        batch=quaternion.shape[0]


        quat = normalize_vector(quaternion).contiguous()

        qw = quat[...,0].contiguous().view(batch, 1)
        qx = quat[...,1].contiguous().view(batch, 1)
        qy = quat[...,2].contiguous().view(batch, 1)
        qz = quat[...,3].contiguous().view(batch, 1)

        # Unit quaternion rotation matrices computatation
        xx = qx*qx
        yy = qy*qy
        zz = qz*qz
        xy = qx*qy
        xz = qx*qz
        yz = qy*qz
        xw = qx*qw
        yw = qy*qw
        zw = qz*qw

        row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
        row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
        row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3

        matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3

        return matrix

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
