#!/bin/bash
#SBATCH --job-name blender_rendering
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 48:15:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH -p tenenbaum
#SBATCH --mem=5G

Blender/blender -b -noaudio -P generate_scenes.py -- --root_dir /om2/user/yyf/CommonFate/scenes/voronoi_single_shape/ --n_scenes 10000 --start_scene 280 --render_size 512 --render_views masks --scene_config configs/1_shape_textured_background.json --engine CYCLES --n_frames 64 --samples 64 --device CUDA


