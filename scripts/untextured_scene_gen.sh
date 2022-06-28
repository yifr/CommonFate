#!/bin/bash
#SBATCH --job-name gt_test_scenes
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 94:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=12GB
#SBATCH -p tenenbaum
#SBATCH --mem=5G
#SBATCH --output=%x.%A_%a.log


START_SCENE=0 #$((2500 * (($SLURM_ARRAY_TASK_ID % 4))))
echo $IDX
echo $START_SCENE
echo ${1}

Blender/blender -b -noaudio -P generate_scenes.py -- \
    --root_dir /om/user/yyf/CommonFate/scenes/test_${1}/superquadric_${IDX} \
    --scene_config formats/${1}_1_shape.json \
    --n_scenes 10 \
    --start_scene $START_SCENE \
    --render_size 512 \
    --render_views ground_truth \
    --engine CYCLES \
    --n_frames 64 \
    --samples 64 \
    --device CUDA \
    --save_config \
    --save_blendfile \


