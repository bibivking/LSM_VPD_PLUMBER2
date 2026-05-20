#!/bin/bash

### It is hard to use GH200 since it doesn't have a lot of module such as cuda/11.8.0, and it cannot access to the conda I installed.

# ====== Preset ======
# Define an array of models
models=("CHTESSEL_Ref_exp1" "GFDL" "CLM5a" "JULES_GL9")

# models=("obs" "CABLE" "CABLE-POP-CN" "CHTESSEL_Ref_exp1" "CLM5a" "GFDL" 
#         "JULES_GL9" "MATSIRO" "MuSICA" "NoahMPv401"
#         "ORC2_r6593" "ORC3_r8120" "STEMMUS-SCOPE")  # Add all required models
### "JULES_GL9_withLAI"

folder_path='/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP'
batch_size=5000
score_metric='r2'

# Loop through each model and create a separate bash script
for model in "${models[@]}"; do
    job_script="XGBoost_SHAP_${model}.sh"

    cat > "$job_script" <<EOF
#!/bin/bash

#PBS -m ae
#PBS -l walltime=12:00:00
#PBS -l select=1:cpu_arch=aarch64:ngpus=1:ncpus=72:mem=585505mb
#PBS -M mengyuan.mu@unsw.edu.au
#PBS -j oe
#PBS -m ae

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
python XGBoost_SHAP.py --select_model "$model" --folder_path "$folder_path" --batch_size "$batch_size" --score_metric "$score_metric"

EOF

    # Make the job script executable
    chmod +x "$job_script"
    echo "Generated: $job_script"
    qsub "$job_script"
    
done

