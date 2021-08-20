#!/usr/bin/env bash
module load python/3.7
source ~/ENV/bin/activate

cd /home/hojin/projects/def-ssanner/hojin/vae-pe
#cd ~/code/vae-pe
python simulate.py --saved_model VAE_beta_multilayer.pt --data_name ml10 --data_dir fold0 --conf sim_abs_diff_neg1_noise10.config
