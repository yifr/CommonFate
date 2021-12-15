#!/bin/bash
#SBATCH --job-name voronoi_scenes
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 94:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=12GB
#SBATCH -p tenenbaum
#SBATCH --mem=5G
#SBATCH --array=0-15
#SBATCH --output=/om/user/yyf/CommonFate/%x.%A_%a.log


IDX=$((SLURM_ARRAY_TASK_ID / 4 + 1))
START_SCENE=$((2500 * (($SLURM_ARRAY_TASK_ID % 4))))
echo $IDX
echo $START_SCENE

Blender/blender -b -noaudio -P generate_scenes.py -- \
    --root_dir /om/user/yyf/CommonFate/scenes/voronoi/superquadric_${IDX} \
    --scene_config formats/voronoi_${IDX}_shape.json \
    --n_scenes 2500 \
    --start_scene $START_SCENE \
    --render_size 512 \
    --render_views masks \
    --engine CYCLES \
    --n_frames 64 \
    --samples 64 \
    --device CUDA


