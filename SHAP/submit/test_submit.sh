#!/bin/bash

#PBS -m ae
#PBS -l walltime=6:00:00
#PBS -l mem=124GB
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -j oe
#PBS -M mengyuan.mu@unsw.edu.au
#PBS -m ae

# Load CUDA (very important)
module load cuda/11.8.0

# Enable conda (THIS IS THE KEY PART)
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the correct environment
conda activate env_xgboost_gpu_shap

cd /srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP
python test.py