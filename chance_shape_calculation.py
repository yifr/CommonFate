import json
import numpy as np

params = []
for scene in range(650):
    param_file = f'scenes/scene_{scene:03d}/params.json'
    print(param_file)
    with open(param_file, 'r') as f:
        data = json.load(f)
    exponents = data['exponents'][:-1]
    scaling = data['scaling'][:-1]
    scene_params = np.concatenate([exponents, scaling], axis=0)
    params.append(scene_params)

params = np.array(params)
print(np.mean(params))
