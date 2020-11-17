from itertools import permutations
import math

import numpy as np
from scipy.spatial import ConvexHull
import trimesh
from pointcloud import Pointcloud


class Shape(object):
    """A wrapper class for shapes"""
    def __init__(self, points, face_idxs):
        self._points = points
        self._faces_idxs = face_idxs

    @property
    def points(self):
        if self._points is None:
            raise NotImplementedError()
        return self._points

    @property
    def faces_idxs(self):
        if self._faces_idxs is None:
            raise NotImplementedError()
        return self._faces_idxs

    def save_as_mesh(self, filename, format="ply"):
        m = trimesh.Trimesh(vertices=self.points.T, faces=self.faces_idxs)
        # m.vertices = self.points.T
        print('Generated / Trimesh faces: {}, {} ... Generated / Trimesh Vertices: {}, {} '.format(
            self.faces_idxs.shape, m.faces.shape, self.points.shape, m.vertices.shape))
        # Make sure that the face orinetations are ok
        trimesh.repair.fix_normals(m, multibody=True)
        trimesh.repair.fix_winding(m)
        assert m.is_winding_consistent == True
        m.export(filename)        
    

    def sample_faces(self, N=1000):
        m = trimesh.Trimesh(vertices=self.points.T, faces=self.faces_idxs)
        # Make sure that the face orinetations are ok
        trimesh.repair.fix_normals(m, multibody=True)
        trimesh.repair.fix_winding(m)
        assert m.is_winding_consistent == True
        P, t = trimesh.sample.sample_surface(m, N)
        return np.hstack([
            P, m.face_normals[t, :]
        ])

    def rotate(self, R):
        """ 3x3 rotation matrix that will rotate the points
        """
        # Make sure that the rotation matrix has the right shape
        assert R.shape == (3, 3)
        self._points = R.T.dot(self.points)

        return self

    def translate(self, t):
        # Make sure thate everything has the right shape
        assert t.shape[0] == 3
        assert t.shape[1] == 1
        self._points = self.points + t

        return self
        
    @staticmethod
    def get_orientation_of_face(points, face_idxs):
        # face_idxs corresponds to the indices of a single face
        assert len(face_idxs.shape) == 1
        assert face_idxs.shape[0] == 3

        x = np.vstack([
            points.T[face_idxs, 0].T,
            points.T[face_idxs, 1].T,
            points.T[face_idxs, 2].T
        ]).T

        # Based on the Wikipedia article
        # https://en.wikipedia.org/wiki/Curve_orientation
        # If the determinant is negative, then the polygon is oriented
        # clockwise. If the determinant is positive, the polygon is oriented
        # counterclockwise
        return np.linalg.det(x)

    @staticmethod
    def fix_orientation_of_face(points, face_idxs):
        # face_idxs corresponds to the indices of a single face
        assert len(face_idxs.shape) == 1
        assert face_idxs.shape[0] == 3

        # Iterate over all possible permutations
        for item in permutations(face_idxs, face_idxs.shape[0]):
            t = np.array(item)
            orientation = Shape.get_orientation_of_face(points, t)
            if orientation < 0:
                pass
            else:
                return t


class ConvexShape(Shape):
    """A wrapper class for convex shapes"""
    def __init__(self, points):
        self._points = points

        # Contains the convexhull of the set of points (see cv property)
        self._cv = None
        # Contains the faces_idxs (see face_idxs Shape property)
        self._faces_idxs = None

    @property
    def points(self):
        return self._points

    @property
    def cv(self):
        if self._cv is None:
            self._cv = ConvexHull(self.points.T)
        return self._cv

    @property
    def faces_idxs(self):
        if self._faces_idxs is None:
            #mesh = trimesh.load('reference_faces.ply')
            self._faces_idxs = np.array(self.cv.simplices) # np.array(mesh.faces)
            self._make_consistent_orientation_of_faces()
        return self._faces_idxs

    def _make_consistent_orientation_of_faces(self):
        for i, face_idxs in zip(range(self.faces_idxs.shape[0]), self.faces_idxs):
            # Compute the orientation for the current face
            orientation = Shape.get_orientation_of_face(self.points, face_idxs)
            if orientation < 0:
                # if the orientation is negative, permute the face_idxs to make
                # it positive
                self._faces_idxs[i] =\
                    Shape.fix_orientation_of_face(self.points, face_idxs)


def signed_exp(x, n):
    return np.sign(x) * (np.abs(x) ** n)

class Ellipsoid(ConvexShape):
    def __init__(self, x_scale=1, y_scale=1, z_scale=1, epsilons=[1, 1]):
        """
        Params:
            yscale [0, 2]
            xscale [0, 2]
            height [0, 5]
            epsilons: [e1: vertical squareness, e2: horizontal squareness] [0.25, 1]
        """
        super(Ellipsoid, self).__init__(
            self._create_points(x_scale, y_scale, z_scale, epsilons)
        )

    def _create_points(self, x_scale, y_scale, z_scale, eps):
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        phi = np.linspace(-np.pi, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)

        x = x_scale * signed_exp(np.cos(theta), eps[0]) * signed_exp(np.cos(phi), eps[1])
        y = y_scale * signed_exp(np.cos(theta), eps[0]) * signed_exp(np.sin(phi), eps[1])
        z = z_scale * signed_exp(np.sin(theta), eps[0])
        points = np.stack([x, y, z]).reshape(3, -1)

        return points

class Toroid(ConvexShape):
    def __init__(self, x_scale=1, y_scale=1, z_scale=1, inner_radius=0.5, epsilons=[0.5, 0.5]):
        """
        Params:
            yscale [0, 2]
            xscale [0, 2]
            height [0, 5]
            inner radius [0, 2]
            epsilons: [e1: vertical squareness, e2: horizontal squareness] [0.25, 1]
        """
        super(Toroid, self).__init__(
            self._create_points(x_scale, y_scale, z_scale, inner_radius, epsilons)
        )

    def _create_points(self, x_scale, y_scale, z_scale, inner_radius, eps):
        theta = np.linspace(-np.pi, np.pi, 100)
        phi = np.linspace(-np.pi, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        x = x_scale * (inner_radius + signed_exp(np.cos(theta), eps[0])) * signed_exp(np.cos(phi), eps[1])
        y = y_scale * (inner_radius + signed_exp(np.cos(theta), eps[0])) * signed_exp(np.sin(phi), eps[1])
        z = z_scale * signed_exp(np.sin(theta), eps[1])
        points = np.stack([x, y, z]).reshape(3, -1)

        return points

