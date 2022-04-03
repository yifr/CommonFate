#!/bin/bash
#SBATCH --job-name noise_normals
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 94:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -p normal
#SBATCH --mem=5G
#SBATCH --array=1-4
#SBATCH --output=%x.%A_%a.log

python utils/exr_to_normals.py --trgt_path /om2/user/yyf/CommonFate/scenes/test_noise/superquadric_${SLURM_ARRAY_TASK_ID} --images_as_RGB --reverse_normal_sphere

