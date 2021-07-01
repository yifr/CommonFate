import sys

sys.path.append("/Users/yoni/Projects/CommonFate/generator")
import bpy
import numpy as np
import superquadrics
from scipy.spatial import Delaunay

# from render_scenes import delete_all

# delete_all('MESH')

#x, y, z = superquadrics.superellipsoid([0.2232, 0.2232], [1, 1, 1, 2], 10)

#faces, verts = superquadrics.get_faces_and_verts(x, y, z)
#print(np.array(verts).shape)
#faces = Delaunay(np.array(verts).T, qhull_options="Tv").simplices

def add_mesh(name, verts, faces, edges=None, col_name="Collection"):
    if edges is None:
        edges = []
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)


#add_mesh("meshtest", verts, faces)

#u=np.linspace(-np.pi,np.pi, 50)
#v=np.linspace(-np.pi,np.pi, 50)
#u,v=np.meshgrid(u,v)
#u=u.flatten()
#v=v.flatten()


#def fexp(func, w, m):
#    return np.sign(func(w)) * np.abs(func(w)) ** m

#eps = [1, 2.5, 2]
#x = (2 + fexp(np.cos, u, eps[0])) * fexp(np.cos, v, eps[1])
#y = (2 + fexp(np.cos, u, eps[0])) * fexp(np.sin, v, eps[1])
#z = fexp(np.sin, u, eps[0])

##define 2D points, as input data for the Delaunay triangulation of U
#points2D=np.vstack([u,v]).T
#tri = Delaunay(points2D)#triangulate the rectangle U

#verts = [(x[i], y[i], z[i]) for i in range(x.shape[0])]
#print(len(verts), tri.simplices.shape)

#shape = superquadrics.SuperEllipsoid([0.5, 2.5], [1])
#add_mesh("test", shape.verts, shape.faces)

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