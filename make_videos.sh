#!/bin/bash

root_dir=$1
for scene in $root_dir/*; do
    ffmpeg -y -framerate 25 -i $scene/images/img_%04d_sm.png -vf format=yuv420p $scene/vid_128x128.mp4
done

