import os
import json
import logging
import numpy as np
import superquadrics
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
root_dir = '/om2/user/yyf/CommonFate/data'

if not os.path.exists(root_dir):
    os.mkdir(root_dir)

# Shape parameters
n = 100           # Number points to generate per shape

# Range of exponents:
exp_range = np.arange(0, 4.1, 0.25)

scene_num = 0
print('Generating meshes...')
for r in exp_range:
    for t in exp_range:
        for a in range(3):
            scene_dir = os.path.join(root_dir, 'scene_%03d'%scene_num)

            # logging.info('SCENE: %s' % scene_dir)
            if not os.path.exists(scene_dir):
                os.mkdir(scene_dir)

            epsilons = [r, t, r]

            print('[SCENE: %s] Generating mesh with params: (e1=%.3f, e2=%.3f, e3=%.3f)...' %(scene_dir, epsilons[0], epsilons[1], epsilons[2]))

            scaling = np.random.uniform(0.5, 2.1, 4).round(decimals=2)
            x, y, z = superquadrics.superellipsoid(epsilons, scaling, n)
            fname = os.path.join(scene_dir, 'mesh.obj')
            superquadrics.save_obj_not_overlap(fname, x, y, z)

            with open(os.path.join(scene_dir, 'params.json'), 'w') as f:
                data = {'exponents': epsilons, 'scaling': list(scaling)}
                json.dump(data, f)

            scene_num += 1

print(f'Done generating {scene_num} scenes')
