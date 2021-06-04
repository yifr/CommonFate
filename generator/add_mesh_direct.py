import sys

sys.path.append("/Users/yoni/Projects/CommonFate/generator")
import bpy
import numpy as np
import superquadrics

# from render_scenes import delete_all

# delete_all('MESH')

x, y, z = superquadrics.supertoroids([0.2232, 0.2232], [1, 1, 1, 2], 100)
faces, verts = superquadrics.get_faces_and_verts(x, y, z)


def add_mesh(name, verts, faces, edges=None, col_name="Collection"):
    if edges is None:
        edges = []
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)


add_mesh("meshtest", verts, faces)
