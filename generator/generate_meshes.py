import os
import json
import logging
import numpy as np
import superquadrics
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
root_dir = '/Users/yoni/Projects/CommonFate/scenes'

if not os.path.exists(root_dir):
    os.mkdir(root_dir)

n_meshes = 2

# Shape parameters
n = 100           # Number points to generate per shape

print('Generating meshes...')
for i in range(100):
        for a in range(3):
            scene_dir = os.path.join(root_dir, 'scene_%03d'%i)

            # logging.info('SCENE: %s' % scene_dir)
            if not os.path.exists(scene_dir):
                os.mkdir(scene_dir)

            mesh_data = {}
            for m in range(n_meshes):
                r, t = np.random.uniform(0, 4.00001, 2).round(decimals=2)
                scaling = np.random.uniform(0.5, 2.1, 4).round(decimals=2)

                epsilons = [r, t, r]

                print('[SCENE: %s] Generating mesh with params: (e1=%.3f, e2=%.3f, e3=%.3f)...' %(scene_dir, epsilons[0], epsilons[1], epsilons[2]))

                x, y, z = superquadrics.superellipsoid(epsilons, scaling, n)
                fname = os.path.join(scene_dir, f'mesh_{m}.obj')
                superquadrics.save_obj_not_overlap(fname, x, y, z)
            
            mesh_data[f'mesh_{m}'] = {'exponents': epsilons, 'scaling': list(scaling)}
            with open(os.path.join(scene_dir, 'params.json'), 'w') as f:
                json.dump(mesh_data, f)

print(f'Done generating {scene_num} scenes')
