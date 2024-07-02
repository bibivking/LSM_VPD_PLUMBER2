#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalbw
#PBS -l walltime=2:00:00
#PBS -l mem=256GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output
diff Qle_all_sites_4sigma.csv Qle_all_sites_filter_4sigma_check.csv

