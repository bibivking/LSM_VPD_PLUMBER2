#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q megamem
#PBS -l walltime=14:00:00
#PBS -l mem=3000GB
#PBS -l ncpus=28
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+gdata/fs38+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2
python calc_LSM_lead_CMIP6_uncertaity.py
