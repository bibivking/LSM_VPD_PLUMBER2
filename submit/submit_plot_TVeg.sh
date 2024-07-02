#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalbw
#PBS -l walltime=12:00:00
#PBS -l mem=20GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-22.01
cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2
python plot_ET_vs_VPD_wet_dry_TVeg.py
