#!/bin/bash
#SBATCH --job-name test_scenes
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 00:12:00
#SBATCH -N 1
#SBATCH --constraint=12GB
#SBATCH -p tenenbaum
#SBATCH --mem=5G
#SBATCH --array=1-4
#SBATCH --output=%x.%A_%a.log


IDX=$SLURM_ARRAY_TASK_ID
START_SCENE=0 #$((2500 * (($SLURM_ARRAY_TASK_ID % 4))))
echo $IDX
echo $START_SCENE
echo ${1}

python scripts/make_videos.py --root_dir /om/user/yyf/CommonFate/scenes/test_${1}/superquadric_${IDX} --n_videos 48 --output_dir /om/user/yyf/CommonFate/test_set/${1}/superquadric_${IDX}/


