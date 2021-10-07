#!/bin/sh

for i in $(seq 1 10); do
    y=$(printf "%03d" $i)
    scp -r yyf@openmind.mit.edu:/om2/user/yyf/CommonFate/scenes/gestalt_v2/scene_$y scenes
done
