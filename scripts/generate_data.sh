#!/bin/bash
#SBATCH --job-name blender_rendering
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH -p normal
#SBATCH --mem=5G

Blender/blender -b -P generator/render_scenes.py -- --root_dir /om2/user/yyf/CommonFate/data/ --n_scenes 1 --render_size 1024
