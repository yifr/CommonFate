import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="scenes")
parser.add_argument("--n_videos", type=int, default="1")
parser.add_argument("--views", type=str, default="all")
parser.add_argument("--start_scene", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="movies")
args = parser.parse_args()

for i in range(args.start_scene, args.start_scene + args.n_videos):
    path = f"{args.root_dir}/scene_{i:03d}/images"

    os.system(
        f"ffmpeg -y -framerate 15 -i {path}/Image%04d.png -pix_fmt yuv420p -c:v libx264 movies/{i:03d}.mp4"
    )

    """
    if args.views == "all":
        try:
            os.system(
                f"ffmpeg -y -framerate 15 -i {path}/gt_%04d.png -pix_fmt yuv420p -c:v libx264 movies/scene_{i:03d}_gt.mp4"
            )
        except:
            print("Could not find ground truth videos")
    """
