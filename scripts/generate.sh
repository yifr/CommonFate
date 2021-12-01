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

Blender/blender -b -noaudio -P generate_scenes.py -- --root_dir /om2/user/yyf/CommonFate/scenes/gestalt_masks_multiscene/ --n_scenes 10000 --start_scene 381 --render_size 512 --render_views masks --scene_config configs/multi_shape_textured_background.json --engine CYCLES --n_frames 50 --samples 128 --device CUDA


