#!/bin/bash
#PBS -N run_model_site
#PBS -m ae
#PBS -M mengyuan.mu@unsw.edu.au
#PBS -j oe
#PBS -l walltime=2:00:00
#PBS -l ncpus=1
#PBS -l mem=180GB
#PBS -l ngpus=1

# ====== Environment ======
module load cuda/11.8.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_xgboost_gpu_shap

# ====== Run ======
cd /srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP

python XGBoost_SHAP.py \
  --select_model obs \
  --folder_path /srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP \
  --batch_size 5000 \
  --score_metric kge_scorer \
  --select_site AU-Tum 


# #!/bin/bash
# #PBS -m ae
# #PBS -l walltime=2:00:00
# #PBS -l ncpus=20
# #PBS -l mem=180GB
# #PBS -l ngpus=1
# #PBS -M mengyuan.mu@unsw.edu.au
# #PBS -j oe
# #PBS -m ae

# # ====== Execute ======
# # Load CUDA
# module load cuda/11.8.0

# # Enable conda
# source ~/miniconda3/etc/profile.d/conda.sh

# # Activate environment
# conda activate env_xgboost_gpu_shap

# # Change to script directory
# cd /srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP

# # Run Python script
# python XGBoost_SHAP.py --select_model "obs" --folder_path "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP" --batch_size 5000 --score_metric "kge_scorer" --select_site "AU-Tum"

# EOF
