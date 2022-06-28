import sys


sys.path.append("/Users/yoni/Projects/CommonFate")
import bpy
import numpy as np
import shapes
from scipy.spatial import Delaunay
from pyquaternion import Quaternion

import imp
imp.reload(shapes)

def add_mesh(name, verts, faces, edges=None, col_name="Collection"):
    if edges is None:
        edges = []
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)

def render_single_shape(shape_type, shape_params):
    temp = shapes.SuperQuadric(shape_type, shape_params, [1, 1, 1, 2])
    add_mesh("superquadric", temp.verts, temp.faces)
    material = bpy.data.materials.new(name="Material")
    material.use_nodes = True
    color = list(np.random.uniform(0, 1, 3))
    color.append(1)

    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color

    ob = bpy.context.active_object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = material
    else:
        # no slots
        ob.data.materials.append(material)


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
            
#add_shapegenerator_shapes()

shape = shapes.SuperQuadric("supertoroid", (1, 0.01), [1,1,1,2])
add_mesh("shape", shape.verts, shape.faces)

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