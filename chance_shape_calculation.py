import json
import numpy as np

params = []
for scene in range(650):
    param_file = f"scenes/data/scene_{scene:03d}/params.json"
    print(param_file)
    with open(param_file, "r") as f:
        data = json.load(f)
    exponents = data["mesh_0"]["exponents"]
    params.append(exponents)

params = np.array(params)
print(np.mean(params))
