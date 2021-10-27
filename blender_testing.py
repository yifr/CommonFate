import sys

sys.path.append("/Users/yoni/Projects/CommonFate")
import bpy
import numpy as np
import shapes
from scipy.spatial import Delaunay
from pyquaternion import Quaternion

# from render_scenes import delete_all

# delete_all('MESH')

# x, y, z = superquadrics.superellipsoid([0.2232, 0.2232], [1, 1, 1, 2], 10)

# faces, verts = superquadrics.get_faces_and_verts(x, y, z)
# print(np.array(verts).shape)
# faces = Delaunay(np.array(verts).T, qhull_options="Tv").simplices


def add_mesh(name, verts, faces, edges=None, col_name="Collection"):
    if edges is None:
        edges = []
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)


# add_mesh("meshtest", verts, faces)

# u=np.linspace(-np.pi,np.pi, 50)
# v=np.linspace(-np.pi,np.pi, 50)
# u,v=np.meshgrid(u,v)
# u=u.flatten()
# v=v.flatten()


# def fexp(func, w, m):
#    return np.sign(func(w)) * np.abs(func(w)) ** m

# eps = [1, 2.5, 2]
# x = (2 + fexp(np.cos, u, eps[0])) * fexp(np.cos, v, eps[1])
# y = (2 + fexp(np.cos, u, eps[0])) * fexp(np.sin, v, eps[1])
# z = fexp(np.sin, u, eps[0])

##define 2D points, as input data for the Delaunay triangulation of U
# points2D=np.vstack([u,v]).T
# tri = Delaunay(points2D)#triangulate the rectangle U

# verts = [(x[i], y[i], z[i]) for i in range(x.shape[0])]
# print(len(verts), tri.simplices.shape)

# shape = superquadrics.SuperEllipsoid([0.5, 2.5], [1])
# add_mesh("test", shape.verts, shape.faces)
#shape = shapes.SuperQuadric("supertoroid", shape_params=[1.59864339, 3.80445551, 0.83916191], scaling_params=[1,1,1,2])
#add_mesh("test", verts=shape.verts, faces=shape.faces)

def rotate(obj):
    obj.rotation_mode = "QUATERNION"
    degrees = np.linspace(0, 360, 100)
    rotation_axis = np.random.uniform(-1, 1, 3)
    for frame, degree in enumerate(degrees):
            q = Quaternion(axis=rotation_axis, degrees=degree)
            obj.rotation_quaternion = q.elements
            obj.keyframe_insert("rotation_quaternion", frame=frame+1)
            
def add_shapegenerator_shapes():
    x, y, z = 0, 0, 1
    for extrusions in range(7, 12):
        seed = np.random.randint(0, 100000)
        for slide in np.linspace(1, 3, 5):
            subsurf = y > 20 # Make some shapes with rounder edges
            bpy.ops.mesh.shape_generator(random_seed=seed, 
                                         amount=extrusions, 
                                         max_slide=slide,
                                         big_shape_num=np.random.randint(1, 3),
                                         mirror_x=False, 
                                         is_subsurf=subsurf)
                                         
            collection = bpy.data.collections[-1]
 
            for obj in collection.all_objects:
                obj.select_set(True)
            bpy.ops.object.join()
            
            # adjust location coordinates to tile the shapes 
            obj = bpy.context.active_object
            obj.location = (x, y, z)
            rotate(obj)
            
            x += 10
            if x % 50 == 0:
                x = 0
                y += 10 
            
add_shapegenerator_shapes()

"""
dx, dy = 6, 6

s = np.linspace(0.25, 4, 10)
t = np.linspace(0.25, 4, 10)
for i in range(20):
    for j in range(10):
        if i >= 10:
            shape = superquadrics.SuperEllipsoid([s[i - 10], t[j - 10]], [2, 2, 2])
        else:
            shape = superquadrics.SuperToroid([s[i], t[j]], [2], n=25)
               
        add_mesh("test8_%d_%d"%(i, j), shape.verts, shape.faces)
        bpy.data.objects["test8_%d_%d"%(i, j)].location = ((i - 10) * dx, j * dy, 0)
"""
"""
shape = shapes.SuperQuadric("superellipsoid", [1, 2], [10, 10, 10], n_points=5)
verts = shape.get_verts()
shapename = "tt1"
for i, vert in enumerate(verts):
    temp = shapes.SuperQuadric("superellipsoid", [1, 2], [1, 1, 1])
    add_mesh(f"{shapename}_{i}", temp.verts, temp.faces)
    bpy.data.objects[f"{shapename}_{i}"].location = vert
    bpy.data.objects[f"{shapename}_{i}"].keyframe_insert(data_path="location", frame=1)


from pyquaternion import Quaternion

n = 5
rotation_axis = [0, 1, 0]
degrees = np.linspace(0, 120, 120)
verts = np.array(verts)
idxs = np.random.choice(verts, n, replace=False)

print(verts.shape, verts)
for frame, degree in enumerate(degrees):
    q = Quaternion(axis=rotation_axis, degrees=degree)
    rmat = q.rotation_matrix

    verts = np.matmul(verts, rmat)
    for i in idxs:
        bpy.data.objects[f"{shapename}_{i}"].location = verts[i, :]
        bpy.data.objects[f"{shapename}_{i}"].keyframe_insert(
            data_path="location", frame=frame
        )
"""