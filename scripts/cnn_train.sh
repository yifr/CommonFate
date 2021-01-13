#!/bin/bash
#SBATCH --job-name common_fate_inference
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 10:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --constraint=high-capacity
#SBATCH -p normal
#SBATCH --mem=4G

python train.py
