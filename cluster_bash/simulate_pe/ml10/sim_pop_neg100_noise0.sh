#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/bayesian-critiquing-recommender

#cd ~/code/vae-pe
python simulate.py --saved_model VAE_beta_multilayer.pt --data_name ml10 --data_dir fold0_valid/fold0 --conf sim_pop_neg100_noise0.config
