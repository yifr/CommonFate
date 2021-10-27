import pprint
import os
import pickle
import json
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy

parser = ArgumentParser()
parser.add_argument("--scene_dir", type=str, default="scenes", help="Path to scene directory")
parser.add_argument("--scene_num", type=int, default=0, help="which scene to print config for")
parser.add_argument("--config_extension", type=str, default=".pkl", help="filetype for config")
parser.add_argument("--print_nparrays", action="store_true", help="will also print out numerical info like rotation and angle")
parser.add_argument("--save_path", default="", type=str, help="Save config to location")
args = parser.parse_args()

path = os.path.join(args.scene_dir, f"scene_{args.scene_num:03d}", "scene_config" + args.config_extension)

with open(path, "rb") as f:
    if args.config_extension == ".pkl":
        config = pickle.load(f)
    else:
        config = json.load(f)


def delete_numerical(config):
    new_config = deepcopy(config)
    for obj in config["objects"]:
        obj_conf = config["objects"][obj]
        for key in obj_conf:
            if type(obj_conf[key]) == np.ndarray:
                if key == "shape_params":
                    new_config["objects"][obj]["shape_params"] = list(obj_conf[key])
                else:
                    del new_config["objects"][obj][key]

    return new_config

config = delete_numerical(config)

pprint.pprint(config)

if args.save_path:
    with open(args.save_path, "w") as f:
        json.dump(config, f)

