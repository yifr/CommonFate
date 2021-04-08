#!/bin/bash

root_dir=$1
n_scenes=$2
for ((i=0; i < n_scenes; i++ )); do
    scene_num=$(printf "%03d" $i)
    ffmpeg -y -framerate 10 -i $root_dir/scene_$scene_num/images/img_%04d.png -vf format=yuv420p vid_$scene_num.mp4
done

