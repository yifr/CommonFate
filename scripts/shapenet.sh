#!/bin/bash
#SBATCH --job-name common_fate_inference
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --constraint=high-capacity
#SBATCH -p normal
#SBATCH --mem=4G

python train.py \
--model_save_path saved_models/vaes/ \
--model_save_name vae.pt \
--run_name vae_textured_background \
--scene_dir scenes/single_shape_textured_v2 \
--conv_dims 3 \
--epochs 100 \
--n_scenes 1000 \
