#!/bin/bash
#PBS -N run_vif
#PBS -m ae
#PBS -M mengyuan.mu@unsw.edu.au
#PBS -j oe
#PBS -l walltime=2:00:00
#PBS -l ncpus=1
#PBS -l mem=20GB

# ====== Environment ======
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_xgboost_gpu_shap

# ====== Run ======
cd /srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP

python calc_plot_vif.py

