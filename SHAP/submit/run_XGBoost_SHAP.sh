#!/bin/bash

# ====== Preset ======
# Define an array of models
models=("obs" "CABLE" "CABLE-POP-CN" "CHTESSEL_Ref_exp1" "CLM5a" "GFDL" "JULES_GL9" "MATSIRO" "MuSICA" "NoahMPv401" "ORC2_r6593" "ORC3_r8120" "STEMMUS-SCOPE")  # # Add all required models

folder_path='/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP'
batch_size=5000
score_metric='kge_scorer'  # Change to 'r2' if needed
test=true
# Loop through each model and site
for model in "${models[@]}"; do
    if [ "$test" = true ]; then
        job_script="XGBoost_SHAP_${model}_all_${score_metric}_test.sh"
        test_message="--test"
    else
        job_script="XGBoost_SHAP_${model}_all_${score_metric}.sh"
        test_message=""
    fi
    cat > "$job_script" <<EOF
#!/bin/bash

#PBS -m ae
#PBS -l walltime=12:00:00
#PBS -l ncpus=1
#PBS -l mem=20GB
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
cd "$folder_path"

# Run Python script
python XGBoost_SHAP.py --select_model "$model" --folder_path "$folder_path" --batch_size "$batch_size" --score_metric "$score_metric" "$test_message"
python plot_XGBoost_SHAP.py --select_model "$model" --folder_path "$folder_path" --score_metric "$score_metric" "$test_message"

EOF

    # Make the job script executable
    chmod +x "$job_script"
    echo "Generated: $job_script"

    # Submit the job script
    qsub "$job_script"
done




