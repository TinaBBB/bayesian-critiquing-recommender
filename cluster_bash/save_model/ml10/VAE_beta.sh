#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/bayesian-critiquing-recommender

#cd ~/code/vae-pe
python model_save.py --model_name VAE --data_name ml10 --data_dir fold0_valid/fold0 --log_dir VAE_beta_fold0valid --conf VAE_beta.config
