#!/bin/bash
#SBATCH --job-name blender_rendering
#SBATCH -t 03:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH -p normal
#SBATCH --mem=1G

Blender/blender -b --python blender_qrotation.py
