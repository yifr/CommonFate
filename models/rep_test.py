import torch
import numpy as np
from pyquaternion import Quaternion
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

a = torch.rand(100, 6)


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cpu()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat(
        (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1
    )  # batch*3
    return out


def rotation_matrix_from_6d(six_d):
    x_raw = a[:, :3]
    y_raw = a[:, 3:]

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3

    return matrix.numpy()


def is_valid_rotation_matrix(M):
    I = np.eye(3)
    print("\nnp.matmul(M, M.T): \n", np.matmul(M, M.T))
    print("\nDeterminant of input: \n", np.linalg.det(M))
    if np.all((np.matmul(M, M.T)) == I) and (np.linalg.det(M) == 1):
        return True
    else:
        return False


example_6d = torch.rand(10, 6)  # example batch size x 6
rotation_matrices = rotation_matrix_from_6d(
    example_6d
)  # Implemented using the og paper author's code
test_mat = rotation_matrices[0]
q = Quaternion(matrix=test_mat, atol=0.0001)

r = R.from_matrix(test_mat)

print("Scipy: \n", r.as_quat())
print("pyquaternion: \n", q.elements)
