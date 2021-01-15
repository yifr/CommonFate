import torch
import torch.nn.functional as F
from pyquaternion import Quaternion


class PoseLoss:
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
        batch_size = ortho6d.shape[0]
        x_raw = ortho6d[:, :3]
        y_raw = ortho6d[:, 3:]
        x = F.normalize(x_raw, p=2, dim=1) # batch * 3
        batch_dot = torch.bmm(x.view(batch_size, 1, -1), y_raw.view(batch_size, -1, 1)).view(batch_size, 1)
        y = F.normalize(y_raw - batch_dot * x, p=2, dim=1) # batch * 3
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

