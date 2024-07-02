#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q hugemem
#PBS -l walltime=10:00:00
#PBS -l mem=1470GB
#PBS -l ncpus=16
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97+gdata/oi10+gdata/fs38

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2
python process_CMIP6_step2_save_1d_csv.py
