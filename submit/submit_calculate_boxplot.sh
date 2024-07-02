#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q expressbw
#PBS -l walltime=1:00:00
#PBS -l mem=200GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-22.01
cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2
python calculate_boxplot.py
