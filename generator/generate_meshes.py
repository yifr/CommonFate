import numpy as np
import trimesh
import os
from tqdm import tqdm

import logging
import superquadrics
from textures import dot_texture

logging.basicConfig(level=logging.INFO)
root_dir = '/home/yyf/CommonFate/scenes/'

if not os.path.exists(root_dir):
    os.mkdir(root_dir)

scene_num = 0
n_scenes = 4000

# Shape parameters
n = 100           # Number points to generate per shape
a = [1, 1, 1, 1] # [x_scale, y_scale, z_scale, toroid_inner_radius]

# Range of exponents:

points = np.linspace(0, )
for i in tqdm(range(n_scenes)):
    scene_dir = os.path.join(root_dir, 'scene_%03d'%scene_num)
    # logging.info('SCENE: %s' % scene_dir)
    if not os.path.exists(scene_dir):
        os.mkdir(scene_dir)

    epsilons = np.random.uniform(0, 5, 3)
    epsilons[2] = epsilons[0]

    #logging.info('[SCENE: %s] Generating mesh with params: (e1=%.3f, e2=%.3f, e3=%.3f)...' %(scene_dir, epsilons[0], epsilons[1], epsilons[2]))

    x, y, z = superquadrics.superellipsoid(epsilons, a, n)
    fname = os.path.join(scene_dir, 'mesh.obj')
    superquadrics.save_obj_not_overlap(fname, x, y, z)

    with open(os.path.join(scene_dir, 'params.txt'), 'w') as f:
        f.write('e1: {:3f}\ne2: {:3f}\ne3: {:3f}'.format(epsilons[0], epsilons[1], epsilons[2]))

    scene_num += 1
