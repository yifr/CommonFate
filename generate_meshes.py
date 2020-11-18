import numpy as np
import trimesh
import shapes
import os


from texture_creation import random_dots

x_scale = 1
y_scale = 1
z_scale = 1

i = 0
root_dir = 'objects/'

for e2 in np.arange(0, 4.25, 0.25):
    for e1 in np.arange(e2, 4, 0.25):
        scene_dir = os.path.join(root_dir, 'scene_%d'%i)
        if not os.path.exists(scene_dir):
            os.mkdir(scene_dir)
        
        print(e1, e2, 'Generating mesh...')
        epsilons = [e1, e2]
        shape = shapes.Ellipsoid(x_scale, y_scale, z_scale, epsilons)
        fname = os.path.join(scene_dir, 'mesh.ply')
        shape.save_as_mesh(fname)

        with open(os.path.join(scene_dir, 'params.txt'), 'w') as f:
            f.write('e1: {}\ne2: {}\n'.format(e1, e2))
        
        print('Generating texture...')
        random_dots(output=os.path.join(scene_dir, 'texture.jpg'))
        
        i += 1
