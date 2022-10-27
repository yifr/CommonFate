import os
import sys
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--glob_dir", type=str, default="scenes")
parser.add_argument("--n_videos", type=int, default="1")
parser.add_argument("--views", type=str, default="all")
parser.add_argument("--output_dir", type=str, default="movies")
args = parser.parse_args()

scenes = glob(args.glob_dir)
for i, scene_path in enumerate(scenes):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(scene_path)
    if args.views == "all":
        for img_pass in ["normals", "noise", "voronoi", "wave", "depth", "untextured"]:
            output_path = os.path.join(args.output_dir, f"{i:03d}_{img_pass}.mp4")
            pass_path = os.path.join(scene_path, img_pass)
            if not os.path.exists(os.path.join(pass_path, "Image0001.png")):
                print("Could not find pass: ", pass_path)
                continue

            print(output_path)
            try:
                os.system(
                    f"ffmpeg -y -framerate 15 -i {pass_path}/Image%04d.png -pix_fmt yuv420p -c:v libx264 {output_path}"
                )
            except:
                print("Couldn't generate video")