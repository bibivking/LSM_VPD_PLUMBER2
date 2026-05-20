#!/bin/bash

# ====== Preset ======
# Define an array of models
models=("obs" "CABLE" "CABLE-POP-CN" "CHTESSEL_Ref_exp1" "CLM5a" "GFDL" "JULES_GL9" "MATSIRO" "MuSICA" "NoahMPv401" "ORC2_r6593" "ORC3_r8120" "STEMMUS-SCOPE")  # # Add all required models
### "CABLE" "CABLE-POP-CN" "CHTESSEL_Ref_exp1" "CLM5a" "GFDL" "JULES_GL9" "MATSIRO" "MuSICA" "NoahMPv401" "ORC2_r6593" "ORC3_r8120" "STEMMUS-SCOPE"

# List all sites
sites=('AU-Tum')
        # 'AR-SLu' 'BW-Ma1' 'DE-Meh' 'ID-Pag' 'PT-Mi1' 'US-MMS' 
        # 'AT-Neu' 'CA-NS1' 'DE-Obe' 'IE-Ca1' 'PT-Mi2' 'US-MOz' 
        # 'AU-ASM' 'CA-NS2' 'DE-Seh' 'IE-Dri' 'RU-Che' 'US-Myb' 
        # 'AU-Cow' 'CA-NS4' 'DE-SfN' 'IT-Amp' 'RU-Fyo' 'US-Ne1' 
        # 'AU-Cpr' 'CA-NS5' 'DE-Tha' 'IT-BCi' 'RU-Zot' 'US-Ne2' 
        # 'AU-Ctr' 'CA-NS6' 'DE-Wet' 'IT-CA1' 'SD-Dem' 'US-Ne3' 
        # 'AU-Cum' 'CA-NS7' 'DK-Fou' 'IT-CA2' 'SE-Deg' 'US-NR1' 
        # 'AU-DaP' 'CA-Qcu' 'DK-Lva' 'IT-CA3' 'UK-Gri' 'US-PFa' 
        # 'AU-DaS' 'CA-Qfo' 'DK-Ris' 'IT-Col' 'UK-Ham' 'US-Prr' 
        # 'AU-Dry' 'CA-SF1' 'DK-Sor' 'IT-Cpz' 'UK-PL3' 'US-SP1' 
        # 'AU-Emr' 'CA-SF2' 'DK-ZaH' 'IT-Isp' 'US-AR1' 'US-SP2' 
        # 'AU-Gin' 'CA-SF3' 'ES-ES1' 'IT-Lav' 'US-AR2' 'US-SP3' 
        # 'AU-GWW' 'CH-Cha' 'ES-ES2' 'IT-LMa' 'US-ARM' 'US-SRG' 
        # 'AU-How' 'CH-Dav' 'ES-LgS' 'IT-Mal' 'US-Aud' 'US-SRM' 
        # 'AU-Lit' 'CH-Fru' 'ES-LMa' 'IT-MBo' 'US-Bar' 'US-Syv' 
        # 'AU-Otw' 'CH-Oe1' 'ES-VDA' 'IT-Noe' 'US-Bkg' 'US-Ton' 
        # 'AU-Rig' 'CN-Cha' 'FI-Hyy' 'IT-Non' 'US-Blo' 'US-Tw4' 
        # 'AU-Rob' 'CN-Cng' 'FI-Kaa' 'IT-PT1' 'US-Bo1' 'US-Twt' 
        # 'AU-Sam' 'CN-Dan' 'FI-Lom' 'IT-Ren' 'US-Cop' 'US-UMB' 
        # 'AU-Stp' 'CN-Din' 'FI-Sod' 'IT-Ro1' 'US-FPe' 'US-Var' 
        # 'AU-TTE' 'CN-Du2' 'FR-Fon' 'IT-Ro2' 'US-GLE' 'US-WCr' 
        # 'AU-Tum' 'CN-HaM' 'FR-Gri' 'IT-SR2' 'US-Goo' 'US-Whs' 
        # 'AU-Whr' 'CN-Qia' 'FR-Hes' 'IT-SRo' 'US-Ha1' 'US-Wkg' 
        # 'AU-Wrr' 'CZ-wet' 'FR-LBr' 'JP-SMF' 'US-Ho1' 'ZA-Kru' 
        # 'AU-Ync' 'DE-Bay' 'FR-Lq1' 'NL-Ca1' 'US-KS2' 'ZM-Mon' 
        # 'BE-Bra' 'DE-Geb' 'FR-Lq2' 'NL-Hor' 'US-Los' 
        # 'BE-Lon' 'DE-Gri' 'FR-Pue' 'NL-Loo' 'US-Me2' 
        # 'BE-Vie' 'DE-Hai' 'GF-Guy' 'PL-wet' 'US-Me4' 
        # 'BR-Sa3' 'DE-Kli' 'HU-Bug' 'PT-Esp' 'US-Me6')

folder_path='/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP'
batch_size=5000
score_metric='kge_scorer'  # Change to 'r2' if needed
test=false
# Loop through each model and site
for model in "${models[@]}"; do
    for site in "${sites[@]}"; do
        if [ "$test" = true ]; then
            job_script="XGBoost_SHAP_${model}_${site}_${score_metric}_test.sh"
            test_message="--test"
        else
            job_script="XGBoost_SHAP_${model}_${site}_${score_metric}.sh"
            test_message=""
        fi
        cat > "$job_script" <<EOF
#!/bin/bash

#PBS -m ae
#PBS -l walltime=4:00:00
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
python XGBoost_SHAP.py --select_model "$model" --select_site  "$site" --folder_path "$folder_path" --batch_size "$batch_size" --score_metric "$score_metric" "$test_message"
python plot_XGBoost_SHAP.py --select_model "$model" --select_site  "$site" --folder_path "$folder_path" --score_metric "$score_metric" "$test_message"

EOF

        # Make the job script executable
        chmod +x "$job_script"
        echo "Generated: $job_script"

        # Submit the job script
        qsub "$job_script"
    done
done
