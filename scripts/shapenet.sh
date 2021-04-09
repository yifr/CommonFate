#!/bin/bash
#SBATCH --job-name common_fate_inference
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 12:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH -p tenenbaum
#SBATCH --mem=4G

python train.py --model_save_path saved_models/shapenet_prob_loss_v1.pt --run_name shapenet_prob_loss_v1 --epochs 1000 --scene_dir scenes/single_shape_plain
