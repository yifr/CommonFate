#!/bin/bash
#SBATCH --job-name common_fate_inference
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --constraint=high-capacity
#SBATCH -p normal
#SBATCH --mem=4G

python train.py \
--model_save_path saved_models/06-15-21-experiment-baselines/ \
--model_save_name 3DCNN_plain_background.pt \
--run_name 3DCNN_plain_background \
--scene_dir scenes/single_shape_plain \
--conv_dims 3 \
--epochs 500 \
--n_scenes 1000 \
