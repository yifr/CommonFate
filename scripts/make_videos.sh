#!/bin/bash

root_dir=$1
n_scenes=$2

for ((i=0; i < n_scenes; i++ )); do
    scene_num=$(printf "%03d" $i)

    ffmpeg -y -framerate 20 -i $root_dir/scene_$scene_num/images/img_%04d.png -vcodec libx264 -pix_fmt yuv420p /Users/yoni/Projects/yifr.github.io/CommonFate/videos/scene_$scene_num.mp4
done

