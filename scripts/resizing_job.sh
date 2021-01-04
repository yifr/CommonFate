#!/bin/bash
#SBATCH --job-name blender_rendering
#SBATCH -t 00:25:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --constraint=any-gpu
#SBATCH -p normal
#SBATCH --mem=1G

./resize.sh /om2/user/yyf/CommonFate/data/
