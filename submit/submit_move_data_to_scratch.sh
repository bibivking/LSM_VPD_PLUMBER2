#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q copyq
#PBS -l walltime=2:30:00
#PBS -l mem=50GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

# mv /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/CMIP6_DT_Qle_historical_global.csv /scratch/w97/mm3972/script_data/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6
# mv /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/CMIP6_DT_Qle_ssp245_global.csv /scratch/w97/mm3972/script_data/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6
# mv /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/CMIP6_Qle_historical_global.csv /scratch/w97/mm3972/script_data/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6
#mv /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/CMIP6_Qle_*_* /scratch/w97/mm3972/script_data/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6

mv /scratch/w97/mm3972/script_data/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/*no_head.csv /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt_tmp
