#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/bayesian-critiquing-recommender

#cd ~/code/vae-pe
python model_save.py --model_name VAEsigma --data_name ml10 --data_dir fold0_valid/fold0 --log_dir VAEsigma_optimal_nb_fold0valid --conf VAEsigma_optimal_nb.config
