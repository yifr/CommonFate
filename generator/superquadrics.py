"""
This file generates superquadrics.
Ref1: https://cse.buffalo.edu/~jryde/cse673/files/superquadrics.pdf
Ref2: https://en.wikipedia.org/wiki/Superquadrics
Author: Huayi Zeng
"""
import os
import numpy as np
from scipy.spatial import Delaunay


def fexp(func, w, exponent):
    """
    Signed exponentiation for superquadrics.
    Params:
        func: function to apply
        w: np.meshgrid: region where shape is defined
        m: float: exponent
    """
    return np.sign(func(w)) * np.abs(func(w)) ** exponent


class SuperToroid:
    def __init__(self, epsilon, a, n=50):
        self.epsilon = epsilon
        self.a = a
        self.n = n
        self.points = self.get_points()
        self.verts = self.get_verts()
        self.faces = self.get_faces()

    def get_points(self):
        """
        Compute x, y, z coordinates of shapes
        """
        u = np.linspace(-np.pi, np.pi, self.n)
        v = np.linspace(-np.pi, np.pi, self.n)
        u, v = np.meshgrid(u, v)

        u = u.flatten()
        v = v.flatten()

        s, t = self.epsilon[0], self.epsilon[1]
        x = (self.a[0] + fexp(np.cos, u, s)) * fexp(np.cos, v, t)
        y = (self.a[0] + fexp(np.cos, u, s)) * fexp(np.sin, v, t)
        z = fexp(np.sin, u, s)
        return x, y, z

    def get_faces(self):
        u = np.linspace(-np.pi, np.pi, self.n)
        v = np.linspace(-np.pi, np.pi, self.n)
        u, v = np.meshgrid(u, v)

        u = u.flatten()
        v = v.flatten()

        points2d = np.vstack([u, v]).T
        triangulation = Delaunay(points2d)
        return list(triangulation.simplices)

    def get_verts(self):
        x, y, z = self.points
        return [(x[i], y[i], z[i]) for i in range(x.shape[0])]


class SuperEllipsoid:
    def __init__(self, epsilon, a, n=50):
        self.epsilon = epsilon
        self.a = a
        self.n = n
        self.points = self.get_points()
        self.verts = self.get_verts()
        self.faces = self.get_faces()

    def get_points(self):
        u = np.linspace(-np.pi / 2, np.pi / 2, self.n)
        v = np.linspace(-np.pi, np.pi, self.n)
        u, v = np.meshgrid(u, v)

        u = u.flatten()
        v = v.flatten()

        s, t = self.epsilon[0], self.epsilon[1]
        x = self.a[0] * fexp(np.cos, u, s) * fexp(np.cos, v, t)
        y = self.a[1] * fexp(np.cos, u, s) * fexp(np.sin, v, t)
        z = self.a[2] * fexp(np.sin, u, s)
        return x, y, z

    def get_faces(self):
        u = np.linspace(-np.pi / 2, np.pi / 2, self.n)
        v = np.linspace(-np.pi, np.pi, self.n)
        u, v = np.meshgrid(u, v)

        u = u.flatten()
        v = v.flatten()

        points2d = np.vstack([u, v]).T
        triangulation = Delaunay(points2d)
        return list(triangulation.simplices)

    def get_verts(self):
        x, y, z = self.points
        return [(x[i], y[i], z[i]) for i in range(x.shape[0])]


def save_pts(path_save, x, y, z):
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    pts = np.zeros((x.shape[0], 3))
    pts[:, 0], pts[:, 1], pts[:, 2] = x, y, z
    np.savetxt(path_save, pts, fmt="%1.3f")


def save_obj(path_save, x, y, z, threshold=-1):
    hei, wid = x.shape[0], x.shape[1]
    with open(path_save, "w+") as fout:
        for i in range(hei):
            for j in range(wid):
                if threshold > 0:
                    if (
                        np.abs(x[i, j]) > threshold
                        or np.abs(y[i, j]) > threshold
                        or np.abs(z[i, j]) > threshold
                    ):
                        continue
                    fout.write("v %.3f %.3f %.3f\n" % (x[i, j], y[i, j], z[i, j]))
                else:
                    fout.write("v %.3f %.3f %.3f\n" % (x[i, j], y[i, j], z[i, j]))
        # Write face: we could not write face when we filter out vertices by threshold
        if threshold < 0:
            for i in range(hei - 1):
                for j in range(wid - 1):
                    fout.write(
                        "f %d %d %d\n"
                        % ((i + 1) * wid + j + 1, i * wid + j + 1 + 1, i * wid + j + 1)
                    )
                    fout.write(
                        "f %d %d %d\n"
                        % (
                            (i + 1) * wid + j + 1 + 1,
                            i * wid + j + 1 + 1,
                            (i + 1) * wid + j + 1,
                        )
                    )

    # def get_faces_and_verts(x, y, z, threshold=-1):
    hei, wid = x.shape[0], x.shape[1]
    verts = []
    faces = []
    for i in range(hei):
        for j in range(wid):
            if threshold > 0:
                if (
                    np.abs(x[i, j]) > threshold
                    or np.abs(y[i, j]) > threshold
                    or np.abs(z[i, j]) > threshold
                ):
                    continue
                verts.append((x[i, j], y[i, j], z[i, j]))
            else:
                verts.append((x[i, j], y[i, j], z[i, j]))
        # Write face: we could not write face when we filter out vertices by threshold
        if threshold < 0:
            for i in range(hei - 1):
                for j in range(wid - 1):
                    faces.append(
                        ((i + 1) * wid + j + 1, i * wid + j + 1 + 1, i * wid + j + 1)
                    )
                    faces.append(
                        (
                            (i + 1) * wid + j + 1 + 1,
                            i * wid + j + 1 + 1,
                            (i + 1) * wid + j + 1,
                        )
                    )

    return faces, verts


def save_obj_not_overlap(path_save, x, y, z):
    """
    This function saves super-ellpsoid w/o overlap: the rightmost vertices are not coincide with the leftmost points,
    and only one vertex at the top and bottom
    """
    x = np.transpose(x, (1, 0))
    y = np.transpose(y, (1, 0))
    z = np.transpose(z, (1, 0))
    hei, wid = x.shape[0], x.shape[1]
    count = 0
    with open(path_save, "w+") as fout:
        for i in range(1, hei - 1):
            for j in range(0, wid - 1):
                fout.write("v %.3f %.3f %.3f\n" % (x[i, j], y[i, j], z[i, j]))
                count += 1
        for i in range(0, hei - 3):
            for j in range(0, wid - 2):
                fout.write(
                    "f %d %d %d %d\n"
                    % (
                        i * (wid - 1) + j + 1,
                        i * (wid - 1) + j + 2,
                        (i + 1) * (wid - 1) + j + 2,
                        (i + 1) * (wid - 1) + j + 1,
                    )
                )
        for i in range(0, hei - 3):
            fout.write(
                "f %d %d %d %d\n"
                % (
                    i * (wid - 1) + wid - 2 + 1,
                    i * (wid - 1) + 0 + 1,
                    (i + 1) * (wid - 1) + 0 + 1,
                    (i + 1) * (wid - 1) + wid - 2 + 1,
                )
            )
        fout.write("v %.3f %.3f %.3f\n" % (x[0, 0], y[0, 0], z[0, 0]))
        fout.write("v %.3f %.3f %.3f\n" % (x[-1, -1], y[-1, -1], z[-1, -1]))
        for j in range(0, wid - 2):
            fout.write("f %d %d %d\n" % (count + 1, j + 2, j + 1))
        fout.write("f %d %d %d\n" % (count + 1, 1, wid - 2 + 1))
        for j in range(0, wid - 2):
            fout.write(
                "f %d %d %d\n"
                % (
                    count + 2,
                    (hei - 3) * (wid - 1) + j + 1,
                    (hei - 3) * (wid - 1) + j + 2,
                )
            )
        fout.write(
            "f %d %d %d\n"
            % (
                count + 2,
                (hei - 3) * (wid - 1) + wid - 3 + 2,
                (hei - 3) * (wid - 1) + 1,
            )
        )


def get_faces_and_verts(x, y, z, use_existing_faces=True):
    """
    This function returns faces adn vertices for super-ellpsoid w/o overlap: the rightmost vertices are not coincide with the leftmost points,
    and only one vertex at the top and bottom
    """
    x = np.transpose(x, (1, 0))
    y = np.transpose(y, (1, 0))
    z = np.transpose(z, (1, 0))
    hei, wid = x.shape[0], x.shape[1]
    count = 0
    faces = []
    verts = []
    for i in range(hei):
        for j in range(wid):
            verts.append((x[i, j], y[i, j], z[i, j]))

    for i in range(hei - 1):
        for j in range(wid - 1):
            faces.append(((i + 1) * wid + j + 1, i * wid + j + 1 + 1, i * wid + j + 1))
            faces.append(
                ((i + 1) * wid + j + 1 + 1, i * wid + j + 1 + 1, (i + 1) * wid + j + 1)
            )

    if use_existing_faces:
        current_path = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        try:
            faces = list(
                np.load(os.path.join(current_path, "faces.npy"), allow_pickle=True)
            )
        except:
            print(
                f'Canonical face indexes could not be found at path: {os.path.join(current_path, "faces.npy")}. Falling back on default faces - this may result in unwanted errors. \
                To correct this, extract the face indexes from an existing .obj file and store them in the same directory as superquadrics.py in a faces.npy file.'
            )


#     return faces, verts


def sgn(x):
    return np.sign(x)


def signed_sin(w, m):
    return sgn(np.sin(w)) * np.power(np.abs(np.sin(w)), m)


def signed_cos(w, m):
    return sgn(np.cos(w)) * np.power(np.abs(np.cos(w)), m)


def signed_tan(w, m):
    return sgn(np.tan(w)) * np.power(np.abs(np.tan(w)), m)


def signed_sec(w, m):
    return sgn(np.cos(w)) * np.power(np.abs(1 / np.cos(w)), m)


def superellipsoid(epsilon, a, n):
    """
    We follow https://en.wikipedia.org/wiki/Superquadrics, (code there should be superellipsoid)
    a is a 3-element vector: A, B, C
    epsilon is a 2-element vector: 2/r, 2/s
    """

    eta = np.linspace(-np.pi / 2, np.pi / 2, n)
    w = np.linspace(-np.pi, np.pi, n)
    eta, w = np.meshgrid(eta, w)

    x = a[0] * signed_cos(eta, epsilon[0]) * signed_cos(w, epsilon[1])
    y = a[1] * signed_cos(eta, epsilon[0]) * signed_sin(w, epsilon[1])
    z = a[2] * signed_sin(eta, epsilon[0])
    return x, y, z


def supertoroid(epsilon, a, n):
    """
    https://cse.buffalo.edu/~jryde/cse673/files/superquadrics.pdf EQ 2.22
    """
    eta = np.linspace(-np.pi, np.pi, n)
    w = np.linspace(-np.pi, np.pi, n)
    eta, w = np.meshgrid(eta, w)

    x = a[0] * (a[3] + signed_cos(eta, epsilon[0])) * signed_cos(w, epsilon[1])
    y = a[1] * (a[3] + signed_cos(eta, epsilon[0])) * signed_sin(w, epsilon[1])
    z = a[2] * signed_sin(eta, epsilon[0])
    return x, y, z


if __name__ == "__main__":
    path_save = "temp.obj"

    (
        x,
        y,
        z,
    ) = superellipsoid([3.74, 0.05], [1, 1, 1, 2], 100)
    save_obj_not_overlap(path_save, x, y, z)
