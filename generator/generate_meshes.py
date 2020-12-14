import numpy as np
import trimesh
import os

import logging
import superquadrics
from random_dots import random_dots

root_dir = 'objects/'

if not os.path.exists(root_dir):
    os.mkdir(root_dir)

scene_num = 0

# Shape parameters
n = 64           # Number points to generate per shape
a = [1, 1, 1, 1] # [x_scale, y_scale, z_scale, toroid_inner_radius]

# Range of exponents:
eps = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]

for e1 in eps:
    for e2 in eps:
        scene_dir = os.path.join(root_dir, 'scene_%03d'%scene_num)
        logging.info('SCENE: ', scene_dir)
        if not os.path.exists(scene_dir):
            os.mkdir(scene_dir)

        logging.info(e1, e2, 'Generating mesh...')

        epsilons = [e1, e2, e1]
        x, y, z = superquadrics.superellipsoid(epsilons, a, n)
        fname = os.path.join(scene_dir, 'mesh.obj')
        superquadrics.save_obj_not_overlap(fname, x, y, z)

        with open(os.path.join(scene_dir, 'params.txt'), 'w') as f:
            f.write('e1: {}\ne2: {}\ne3: {}'.format(e1, e2, e1))

        logging.info('Generating texture...')
        random_dots(output=os.path.join(scene_dir, 'texture.jpg'))

        scene_num += 1
