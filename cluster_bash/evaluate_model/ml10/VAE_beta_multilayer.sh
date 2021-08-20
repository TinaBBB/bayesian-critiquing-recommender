#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/bayesian-critiquing-recommender

python model_evaluate.py --model_name VAEmultilayer --data_name ml10 --log_dir VAE_beta_multilayer --conf VAE_beta_multilayer.config
