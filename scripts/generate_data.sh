#!/bin/bash
#SBATCH --job-name blender_rendering
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 24:15:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH -p normal
#SBATCH --mem=5G

Blender/blender -b -P generator/scenes.py -- --root_dir /om2/user/yyf/CommonFate/scenes/gestalt_v2_low_res/ --n_scenes 20 --start_scene 0 --render_size 64  --background_style white --n_frames 120 --device CUDA --scene_config configs/scene_config_3.json
