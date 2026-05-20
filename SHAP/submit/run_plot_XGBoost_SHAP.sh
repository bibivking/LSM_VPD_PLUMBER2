#!/bin/bash

#PBS -m ae
#PBS -l walltime=2:00:00
#PBS -l ncpus=1
#PBS -l mem=50GB
#PBS -M mengyuan.mu@unsw.edu.au
#PBS -j oe

# ====== Execute ======
score_metric='kge_scorer'  # Change to 'r2' if needed
test_message='--test'
# Enable conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate env_xgboost_gpu_shap

# Change to script directory
cd /srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP


models=("obs" "CABLE" "CABLE-POP-CN" "CHTESSEL_Ref_exp1" "CLM5a" "GFDL" "JULES_GL9" "MATSIRO" "MuSICA" "NoahMPv401" "ORC2_r6593" "ORC3_r8120" "STEMMUS-SCOPE")  # # Add all required models

for model in "${models[@]}"; do
python plot_SHAP_selected_periods.py --select_model "$model" --folder_path "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP" --select_site "AU-Tum" --score_metric "$score_metric" "$test_message"
done