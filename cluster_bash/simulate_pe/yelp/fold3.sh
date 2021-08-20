#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/bayesian-critiquing-recommender
#cd ~/code/vae-pe
python simulate_yelp.py --saved_model VAE_beta_multilayer.pt --data_name yelp --data_dir fold3 --conf sim_abs_diff_neg1_noise0.config
python simulate_yelp.py --saved_model VAE_beta_multilayer.pt --data_name yelp --data_dir fold3 --conf sim_abs_diff_neg1_noise3.config
python simulate_yelp.py --saved_model VAE_beta_multilayer.pt --data_name yelp --data_dir fold3 --conf sim_abs_diff_neg1_noise5.config
