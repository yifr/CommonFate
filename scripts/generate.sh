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

Blender/blender -b -P scenes.py -- --root_dir /om2/user/yyf/CommonFate/gestalt_v3/ --n_scenes 1000 --start_scene 0 --render_size 512 --render_views all --scene_config configs/multi_shape_deformed_background.json --engine CYCLES --device CUDA


