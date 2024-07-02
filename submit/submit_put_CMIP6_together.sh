#!/bin/bash

region='north_Am' #'west_EU' #'east_AU' #'global'
model_list=( 'CABLE' 'CABLE-POP-CN' 'CHTESSEL_Ref_exp1' 'CLM5a' 'GFDL' \
             'JULES_GL9' 'JULES_GL9_withLAI' 'MATSIRO' 'MuSICA' 'NASAEnt' \
             'NoahMPv401' 'ORC2_r6593' 'ORC3_r8120' 'QUINCY' 'STEMMUS-SCOPE' 'obs')

dist_type='Gamma'

for model_in in ${model_list[@]}; do

cat > put_CMIP6_together_${model_in}_${region}.sh << EOF_put_CMIP6

#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalsr
#PBS -l walltime=5:00:00
#PBS -l mem=80GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

cd /scratch/w97/mm3972/script_data/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6

CMIP6_model_list=('ACCESS-CM2' 'BCC-CSM2-MR' 'CMCC-CM2-SR5' 'CMCC-ESM2' 'EC-Earth3' 'KACE-1-0-G' 'MIROC6' 'MIROC-ES2L' 'MPI-ESM1-2-HR' 'MPI-ESM1-2-LR' 'MRI-ESM2-0')

# 'historical'
for CMIP6_model in \${CMIP6_model_list[@]}; do
    sed 1d predicted_CMIP6_DT_Qle_historical_\${CMIP6_model}_${model_in}_${region}_${dist_type}.csv > predicted_CMIP6_DT_Qle_historical_\${CMIP6_model}_${model_in}_${region}_${dist_type}_no_head.csv
done
find . -name 'predicted_CMIP6_DT_Qle_historical_*_${model_in}_${region}_${dist_type}_no_head.csv' -print | xargs -I {}  sh -c 'cat "{}" >> "predicted_CMIP6_DT_Qle_historical_${model_in}_${region}_${dist_type}_no_head.csv"'
rm predicted_CMIP6_DT_Qle_historical_*_${model_in}_${region}_${dist_type}_no_head.csv

# 'ssp245'
for CMIP6_model in \${CMIP6_model_list[@]}; do
    sed 1d predicted_CMIP6_DT_Qle_ssp245_\${CMIP6_model}_${model_in}_${region}_${dist_type}.csv > predicted_CMIP6_DT_Qle_ssp245_\${CMIP6_model}_${model_in}_${region}_${dist_type}_no_head.csv
done
# find . -name 'predicted_CMIP6_DT_Qle_ssp245_*_${model_in}_${region}_${dist_type}_no_head.csv' | xargs -I {} cat {} >> predicted_CMIP6_DT_Qle_ssp245_${model_in}_${region}_${dist_type}_no_head.csv
find . -name 'predicted_CMIP6_DT_Qle_ssp245_*_${model_in}_${region}_${dist_type}_no_head.csv' -print | xargs -I {}  sh -c 'cat "{}" >> "predicted_CMIP6_DT_Qle_ssp245_${model_in}_${region}_${dist_type}_no_head.csv"'
rm predicted_CMIP6_DT_Qle_ssp245_*_${model_in}_${region}_${dist_type}_no_head.csv


EOF_put_CMIP6

qsub put_CMIP6_together_${model_in}_${region}.sh

done
