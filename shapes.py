from operator import is_
import os
import numpy as np
from scipy.spatial import Delaunay

try:
    import bpy
except:
    pass


class SuperQuadric:
    def __init__(
        self,
        shape_type,
        shape_params,
        scaling_params,
        n_points=50,
        parent_id=False,
        is_parent=False,
        obj_id="",
        collection_id=None,
    ):
        self.shape_type = shape_type
        self.shape_params = shape_params
        self.scaling_params = scaling_params
        self.n_points = n_points
        self.parent_id = parent_id
        self.is_parent = is_parent
        self.obj_id = obj_id
        self._class = "SuperQuadric"
        self.collection_id = collection_id

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

    def add_mesh(self):
        obj = _add_mesh(self.obj_id, self.verts, self.faces, self.collection_id)
        return obj

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

    def sample_points(self, n, min_dist=None):

        print("Sampling points")

        def dist(a, b):
            return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

        points = []
        starting_idx = np.random.choice(range(len(self.verts)))
        starting_point = self.verts[starting_idx]
        points.append(starting_point)
        for i in range(1, n):
            if dist is None:
                idx = np.random.choice(range(len(self.verts)), n, replace=False)
                point = self.verts[idx]
                points.append(point)
            else:
                reject = True
                while reject:
                    proposal_idx = np.random.choice(range(len(self.verts)))
                    proposal = self.verts[proposal_idx]

                    # Calculate distance to all existing sampled points
                    reject = False
                    for point in points:
                        prop_dist = dist(point, proposal)
                        if prop_dist < min_dist:
                            print(f"Rejecting point {proposal}")
                            reject = True
                            break

                    if not reject:
                        points.append(proposal)
                        break

        return np.array(points)

    def __str__(self):
        return f"{self.shape_type}: \n\tParams={self.shape_params}\n\tScaling={self.scaling_params}"


class ShapeGenerator:
    """
    Wrapper for ShapeGenerator plug-in: https://blendermarket.com/products/shape-generator (unfortunately not free)
    """

    def __init__(
        self,
        seed: int = None,
        n_extrusions: int = 4,
        bevel: bool = False,
        subsurf: bool = True,
        big_shapes: int = 1,
        medium_shapes: int = 3,
        small_shapes: int = 5,
        mirror_x: bool = False,
        mirror_dims: tuple = (True, True, True),
        scaling_params=(1, 1, 1),
        name="ShapeGenerator",
        collection_id=None,
    ):
        """
        Params:
            Seed: int: seed for random number generator
            n_extrusions: int: number of extrusion steps
            bevel: bool: whether to bevel the shape
            subsurf: bool: whether to subsurf the shape
            big_shapes: int: number of big shapes to generate
            medium_shapes: int: number of medium shapes to generate
            small_shapes: int: number of small shapes to generate
            mirror_dims: tuple: Whether to mirror along x, y, z
            name: str: name of the shape generator object
        Returns:
            Mesh for shape
        """
        if seed is None:
            seed = np.random.randint(0, 100000)

        self.seed = seed
        self.n_extrusions = n_extrusions
        self.bevel = bevel
        self.subsurf = subsurf
        self.big_shapes = big_shapes
        self.medium_shapes = medium_shapes
        self.small_shapes = small_shapes
        self.mirror_dims = mirror_dims
        self.name = name
        self._class = "ShapeGenerator"
        self.shape_type = "ShapeGenerator"
        self.collection_id = collection_id

        self.scaling_params = scaling_params

    def add_mesh(self):
        obj = self.generate()
        self.obj = obj
        return obj

    @property
    def shape_params(self):
        return {
            "seed": self.seed,
            "n_extrusions": self.n_extrusions,
            "bevel": self.bevel,
            "subsurf": self.subsurf,
            "big_shapes": self.big_shapes,
            "medium_shapes": self.medium_shapes,
            "small_shapes": self.small_shapes,
            "mirror_dims": self.mirror_dims,
        }

    def generate(self):
        bpy.ops.mesh.shape_generator(
            random_seed=self.seed,
            amount=self.n_extrusions,
            is_bevel=self.bevel,
            is_subsurf=self.subsurf,
            big_shape_num=self.big_shapes,
            medium_shape_num=self.medium_shapes,
            small_shape_num=self.small_shapes,
            mirror_x=self.mirror_dims[0],
            mirror_y=self.mirror_dims[1],
            mirror_z=self.mirror_dims[2],
        )
        # Bake mesh
        bpy.ops.object.join()

        # Rename
        bpy.context.object.name = self.name
        obj = bpy.context.object
        obj.scale = self.scaling_params[:3]

        return obj

    def get_verts(self):
        verts = []
        for vert in self.obj.data.vertices:
            verts.append(vert.co[:])

        self.verts = verts

        return verts

    def __str__(self):
        return "ShapeGenerator"


def _add_mesh(id, verts, faces, collection=None):
    """
    Adds a mesh to Blender with specified id and collection (if specified)
    Params:
        id: str: name of mesh
        verts: list: list of vertices
        faces: list: list of faces
    """
    print("Adding mesh", id, len(verts), len(faces), collection)
    mesh = bpy.data.meshes.new(id)
    obj = bpy.data.objects.new(mesh.name, mesh)

    # Objects are assigned to unique collections if they belong
    # to a larger shape hierarchy
    if collection:
        col = bpy.data.collections.get(collection)
        if not col:
            col = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(col)
    else:
        col = bpy.data.collections.get("Collection")

    col.objects.link(obj)
    # self.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, [], faces)
    return obj


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
            return [5, 5, 5, 10]
        else:
            return [1, 1, 1, 2]


def create_shape(
    shape_type,
    shape_params,
    scaling_params=None,
    is_parent=False,
    n_points=50,
    object_id="",
):
    if shape_type == "random":
        shape_type = np.random.choice(["superquadric", "shapegenerator"])

    if shape_type == "shapegenerator":
        if type(shape_params) == str and shape_params == "random":
            shape_params = {}
        n_extrusions = np.random.randint(2, 5)
        bevel = np.random.choice([True, False])
        subsurf = np.random.choice([True, False])
        big_shapes = np.random.randint(1, 5)
        medium_shapes = np.random.randint(1, 5)
        small_shapes = np.random.randint(1, 5)
        mirror_dims = [np.random.choice([True, False]) for _ in range(3)]

        if scaling_params is None:
            scaling_params = [1, 1, 1]

        return ShapeGenerator(
            seed=shape_params.get("seed", None),
            n_extrusions=shape_params.get("n_extrusions", n_extrusions),
            bevel=shape_params.get("bevel", bevel),
            subsurf=shape_params.get("subsurf", subsurf),
            big_shapes=shape_params.get("big_shapes", big_shapes),
            medium_shapes=shape_params.get("medium_shapes", medium_shapes),
            small_shapes=shape_params.get("small_shapes", small_shapes),
            mirror_dims=shape_params.get("mirror_dims", mirror_dims),
            scaling_params=scaling_params,
            name=object_id,
        )

    else:
        subtype = np.random.choice(["superellipsoid", "supertoroid"])

        if type(shape_params) == str and shape_params == "random":
            shape_params = np.random.uniform(0.01, 4.0, 3)
            if not scaling_params:
                if is_parent:
                    scaling_params = [5, 5, 5, 10]
                else:
                    scaling_params = [1, 1, 1, 2]

        elif type(shape_params) == dict:
            shape_params = shape_params.get("shape_params")
            if shape_params == "random":
                shape_params = random_parameter_assignment("shape_params")
            scaling_params = shape_params.get("scaling_params")
            if scaling_params == "random":
                scaling_params = random_parameter_assignment(
                    "scaling_params", is_parent=is_parent
                )
            if shape_type == "random":
                shape_type = random_parameter_assignment("shape_type")

        shape = SuperQuadric(
            shape_type=subtype,
            shape_params=shape_params,
            scaling_params=scaling_params,
            n_points=n_points,
            is_parent=is_parent,
            obj_id=object_id,
        )

        return shape


if __name__ == "__main__":
    s = ShapeGenerator()
#    s = create_shape("superellipsoid", "random", scaling_params=[5, 5, 5, 10])
#    from mpl_toolkits import mplot3d
#    import matplotlib.pyplot as plt

#    fig = plt.figure()
#    ax = plt.axes(projection="3d")
#    idxs = np.random.choice(len(s.points[0]), 100, replace=False)
#    ax.scatter3D(s.points[0][idxs], s.points[1][idxs], s.points[2][idxs])
#    plt.savefig("figure.png")
