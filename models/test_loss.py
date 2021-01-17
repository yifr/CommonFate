from loss import PoseLoss
import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

ls = PoseLoss(break_symmetry=True)
ortho6d = torch.rand(100, 6)
data = np.load('scenes/scene_000/data.npy', allow_pickle=True).item()
rmat_gt = np.zeros(shape=(100, 3,3))
quat_gt = data['quaternion']

print(data['rotation'][0].shape == (3,3))
rmat_gt = torch.from_numpy(rmat_gt)
quat_gt = torch.from_numpy(quat_gt)

#loss = ls.compute_loss(ortho6d, quat_gt)

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).

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


def get_rotations(quats):
    rotations = torch.zeros((quats.shape[0], 3, 3))
    for i, q in enumerate(quats):
        rotations[i] = torch.from_numpy(Quaternion(q.detach().numpy()).rotation_matrix)
    return rotations

m1 = quaternion_to_matrix(quat_gt)
m2 = get_rotations(quat_gt)
m3 = ls.rotation_matrix_from_quaternion(quat_gt)

idx = 49
print('FB:\n', m1[idx], '\nPyQuaternion\n', m2[idx], '\nNP\n', m3[idx])
