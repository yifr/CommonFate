import os
import json
import logging
import numpy as np
import superquadrics
from tqdm import tqdm
import itertools
import argparse

parser = argparse.ArgumentParser()
logging.basicConfig(level=logging.INFO)


def generate(args):
    root_dir = args.root_dir
    for i in range(args.start_scene, args.start_scene + args.n_scenes):
        scene_dir = os.path.join(root_dir, "scene_%03d" % i)

        # logging.info('SCENE: %s' % scene_dir)
        if not os.path.exists(scene_dir):
            os.mkdir(scene_dir)

        mesh_data = {}

        for m in range(args.n_shapes):

            e1, e2 = np.random.uniform(0, 4.00001, 2).round(decimals=2)
            epsilons = [e1, e2]
            scaling = [1, 1, 1, 2]

            print(f"[SCENE: {i}] Generating mesh with params: (e1={e1}, e2={e2})...")

            curr_shape = ""
            if args.shape_type == "superellipsoid":
                curr_shape = "superellipsoid"
                x, y, z = superquadrics.superellipsoid(epsilons, scaling, 64)
            elif args.shape_type == "supertoroid":
                curr_shape = "supertoroid"
                x, y, z = superquadrics.supertoroid(epsilons, scaling, 64)
            elif args.shape_type == "mixed":
                if np.random.rand() > 0.5:
                    curr_shape = "superellipsoid"
                    x, y, z = superquadrics.superellipsoid(epsilons, scaling, 64)
                else:
                    curr_shape = "supertoroid"
                    x, y, z = superquadrics.supertoroid(epsilons, scaling, 64)
            else:
                raise ValueError(
                    f'Shape type: {args.shape_type} is not currently supported. Options are: "superellipsoid" | "supertoroid" | "mixed""'
                )

            fname = os.path.join(scene_dir, f"mesh_{m}.obj")
            if curr_shape == "superellipsoid":
                superquadrics.save_obj_not_overlap(fname, x, y, z)
            else:
                superquadrics.save_obj(fname, x, y, z)

            mesh_data[f"mesh_{m}"] = {
                "exponents": epsilons,
                "scaling": list(scaling),
                "type": curr_shape,
            }

        param_file = os.path.join(scene_dir, "params.json")
        if os.path.exists(param_file) and not args.overwrite_old_shapes:
            with open(param_file, "r") as f:
                data = json.load(f)
                mesh_data.update(data)

        with open(param_file, "w") as f:
            json.dump(mesh_data, f)


if __name__ == "__main__":
    parser.add_argument(
        "--root_dir", type=str, default="/Users/yoni/Projects/CommonFate/scenes"
    )
    parser.add_argument("--n_shapes", type=int, default=1)
    parser.add_argument(
        "--shape_type",
        type=str,
        default="superellipsoids",
        help="Options: superellipsoids | supertoroids | mixed",
    )
    parser.add_argument("--start_scene", type=int, default=0)
    parser.add_argument("--n_scenes", type=int, default=1000)
    parser.add_argument(
        "--overwrite_old_shapes",
        action="store_true",
        help="Overwrite any old params.json files",
    )

    args = parser.parse_args()
    if not os.path.exists(args.root_dir):
        os.mkdir(args.root_dir)

    generate(args)
