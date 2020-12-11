#!/bin/bash
#SBATCH --job-name blender_rendering
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:5
#SBATCH --constraint=any-gpu
#SBATCH -p normal
#SBATCH --mem=2G

Blender/blender -b --python blender_qrotation.py
