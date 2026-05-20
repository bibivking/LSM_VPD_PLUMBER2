#!/bin/bash

#PBS -m ae
#PBS -l walltime=2:00:00
#PBS -l ncpus=1
#PBS -l mem=20GB
#PBS -M mengyuan.mu@unsw.edu.au
#PBS -j oe

# ====== Execute ======
# Load CUDA
module load cuda/11.8.0

# Enable conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate env_xgboost_gpu_shap

# Change to script directory
cd "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP"

# Run Python script
python plot_SHAP_selected_periods.py --select_model "MATSIRO" --folder_path "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP" --score_metric "kge_scorer" "--test"

