#!/bin/bash
#SBATCH --job-name blender_rendering
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 24:15:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH -p normal
#SBATCH --mem=5G

Blender/blender -b -P generator/render_scenes.py -- --root_dir /om2/user/yyf/CommonFate/data/ --n_scenes 1000 --start_scene 0 --render_size 256 --scene_type default --n_shapes 1 --background_style white --n_frames 20 --experiment_name galaxy_scene_v1
