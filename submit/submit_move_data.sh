#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=192GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
mv /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_3hourly/* /scratch/w97/mm3972/script_data/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_3hourly

