import os
import sys

root_dir = sys.argv[1]
n_videos = sys.argv[2]
for i in range(int(n_videos)):
    path = f"{root_dir}/scene_{i:03d}/images"
    os.system(f'ffmpeg -y -framerate 10 -i {path}/img_%04d.png -pix_fmt yuv420p -c:v libx264 {root_dir}/scene_{i:03d}/scene_{i:03d}.mkv')
