#!/bin/bash

#PBS -m ae
#PBS -l walltime=12:00:00
#PBS -l ncpus=20
#PBS -l mem=180GB
#PBS -l ngpus=1
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
python XGBoost_SHAP_per_site.py --select_model "obs" --select_site "AU-Dry" --folder_path "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP" --batch_size "5000" --GridSearchCV_score "kge_scorer"

