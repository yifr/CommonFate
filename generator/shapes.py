from operator import is_
import os
import numpy as np
from scipy.spatial import Delaunay


class SuperQuadric:
    def __init__(
        self,
        shape_type,
        shape_params,
        scaling_params,
        n_points=50,
        parent_id=False,
        is_parent=False,
    ):
        self.shape_type = shape_type
        self.shape_params = shape_params
        self.scaling_params = scaling_params
        self.n_points = n_points
        self.parent_id = parent_id
        self.is_parent = is_parent
        self._id = None

        if self.shape_type == "superellipsoid":
            self.u_bounds = (-np.pi / 2, np.pi / 2)
            self.v_bounds = (-np.pi, np.pi)

        elif self.shape_type == "supertoroid":
            self.u_bounds = (-np.pi, np.pi)
            self.v_bounds = (-np.pi, np.pi)

        else:
            raise Exception(
                f"Invalid shape type '{self.shape_type}': currently the only supported \
                values for shape_type are: 'supertoroid' and 'superellipsoid'"
            )

        self._points = self.compute_points()
        self._verts = self.get_verts()
        self._faces = self.compute_faces()

    @property
    def points(self):
        return self._points

    @property
    def verts(self):
        return self._verts

    @property
    def faces(self):
        return self._faces

    def compute_points(self):
        """
        Compute x, y, z coordinates of shapes
        """
        u = np.linspace(self.u_bounds[0], self.u_bounds[1], self.n_points)
        v = np.linspace(self.v_bounds[0], self.v_bounds[1], self.n_points)
        u, v = np.meshgrid(u, v)

        u = u.flatten()
        v = v.flatten()

        s, t = self.shape_params[0], self.shape_params[1]

        def fexp(func, w, exponent):
            """
            Signed exponentiation for superquadrics.
            Params:
                func: function to apply
                w: np.meshgrid: region where shape is defined
                m: float: exponent
            """
            return np.sign(func(w)) * np.abs(func(w)) ** exponent

        a1, a2, a3 = self.scaling_params[:3]
        if self.shape_type == "supertoroid":
            a4 = self.scaling_params[-1]
            x = (a4 + a1 * fexp(np.cos, u, s)) * fexp(np.cos, v, t)
            y = (a4 + a2 * fexp(np.cos, u, s)) * fexp(np.sin, v, t)
            z = a3 * fexp(np.sin, u, s)

        elif self.shape_type == "superellipsoid":
            a1, a2, a3 = self.scaling_params[:3]
            x = a1 * fexp(np.cos, u, s) * fexp(np.cos, v, t)
            y = a2 * fexp(np.cos, u, s) * fexp(np.sin, v, t)
            z = a3 * fexp(np.sin, u, s)

        return x, y, z

    def compute_faces(self):
        u = np.linspace(self.u_bounds[0], self.u_bounds[1], self.n_points)
        v = np.linspace(self.v_bounds[0], self.v_bounds[1], self.n_points)
        u, v = np.meshgrid(u, v)

        u = u.flatten()
        v = v.flatten()

        points2d = np.vstack([u, v]).T
        triangulation = Delaunay(points2d)
        return list(triangulation.simplices)

    def get_verts(self, eps=0.001):
        """
        Return vertices of shape
        if parent shape, remove vertices that are closer than `eps`
        """
        x, y, z = self.points
        verts = []

        def dist(a, b):
            return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

        tmp_verts = [(x[i], y[i], z[i]) for i in range(x.shape[0])]

        if not self.is_parent:
            return np.array(tmp_verts)

        for candidate in tmp_verts:
            if not verts:
                verts.append(candidate)
            filter = False
            for existing in verts:
                if dist(candidate, existing) < eps:
                    filter = True
                    break

            if not filter:
                verts.append(candidate)

        return np.array(verts)

    def sample_points(self, n):
        verts = self.verts
        return np.random.choice(verts, n, replace=False)

    def __str__(self):
        return f"{self.shape_type}: \n\tParams={self.shape_params}\n\tScaling={self.scaling_params}"


############################################################
#                    Utility functions                     #
############################################################
def random_parameter_assignment(parameter, **kwargs):
    if parameter == "shape_params":
        return np.random.uniform(0.01, 4)
    if parameter == "shape_type":
        return np.random.choice(["superellipsoid", "supertoroid"])
    if parameter == "scaling_params":
        if kwargs.get("is_parent"):
            return [10, 10, 10, 10]
        else:
            return [1, 1, 1, 1]


def create_shape(
    shape_type, shape_params, scaling_params=None, is_parent=False, n_points=50
):
    if shape_type == "random":
        shape_type = np.random.choice(["superellipsoid", "supertoroid"])

    if type(shape_params) == str and shape_params == "random":
        shape_params = np.random.uniform(0.01, 4.0, 3)
        if is_parent:
            scaling_params = [10, 10, 10]
        else:
            scaling_params = [1, 1, 1]

    elif type(shape_params) == dict:
        shape_params = shape_params.get("shape_params")
        if shape_params == "random":
            shape_params = random_parameter_assignment("shape_params")
        scaling_params = shape_params.get("scaling_params")
        if scaling_params == "random":
            scaling_params = random_parameter_assignment("scaling_params")
        if shape_type == "random":
            shape_type = random_parameter_assignment("shape_type")

    shape = SuperQuadric(shape_type, shape_params, scaling_params, n_points, is_parent)

    return shape


if __name__ == "__main__":
    s = create_shape("superellipsoid", "random", scaling_params=[10, 10, 10])
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    idxs = np.random.choice(len(s.points[0]), 100, replace=False)
    ax.scatter3D(s.points[0][idxs], s.points[1][idxs], s.points[2][idxs])
    plt.show()
