#!/bin/bash

# Set the path to search
PLUMBER2_met_path="/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
IGBP_types=('GRA' 'OSH' 'SAV' 'WSA' 'CSH' 'DBF' 'ENF' 'EBF' 'MF' 'WET' 'CRO')
clim_types=('Af' 'Am' 'Aw' 'BSh' 'BSk' 'BWh' 'BWk' 'Cfa' 'Cfb' 'Csa' 'Csb' 'Cwa' 'Dfa' 'Dfb' 'Dfc' 'Dsb' 'Dsc' 'Dwa' 'Dwb' 'ET')

# Loop through all files in the path
for IGBP_type in ${IGBP_types[@]}; do

  # Print the site name to the console
  echo "$IGBP_type"

  # Set the parameters
  case_name="TVeg_SM_per_all_models_${IGBP_type}_data_selected_STD_annual_model"
  data_selection='True'
  add_aridity_index='True'
  var_name='"TVeg"'
  standardize='"STD_annual_model"' # 'None' #
  selected_by='"SM_per_all_models"'
  add_Xday_mean_EF='None'
  low_bound='[0,15]'
  high_bound='[85,100]'
  select_site="None"
  middle_day='False'
  LAI_range='None'
  IGBP_type="'${IGBP_type}'"
  add_LAI='True'
  add_qc='True'
  add_SMtopXm='"0.5"'
  add_Rnet_caused_ratio='True'
  add_normalized_SMtopXm='None'
  quality_ctrl='True'
  day_time='True'
  time_scale='"hourly"'
  region_name='"global"'
  veg_fraction='None'
  Tair_constrain='None'
  VPD_sensitive='False'

  # Change directory to the script location
  cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2

  # Create the Python script
  cat > "process_step3_raw_data_for_curve_${case_name}.py" << EOF_process_step3
#!/usr/bin/env python

import os
import gc
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from quality_control import *
from PLUMBER2_VPD_common_utils import *
from process_step3_raw_data_for_curve import *

# Path of PLUMBER 2 dataset
PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
site_names, IGBP_types, clim_types, model_names = load_default_list()

# set cause
var_name       = ${var_name}
data_selection = ${data_selection}
add_aridity_index=${add_aridity_index}
selected_by    = ${selected_by}
standardize    = ${standardize}
time_scale     = ${time_scale}
add_Xday_mean_EF=${add_Xday_mean_EF}
select_site    =${select_site}
low_bound      =${low_bound}
high_bound     =${high_bound}

add_LAI        =${add_LAI}
add_qc         =${add_qc}
add_SMtopXm    =${add_SMtopXm}
add_normalized_SMtopXm = ${add_normalized_SMtopXm}
add_Rnet_caused_ratio=${add_Rnet_caused_ratio}
quality_ctrl   =${quality_ctrl}

region_name    =${region_name}
veg_fraction   =${veg_fraction}
LAI_range      =${LAI_range}
IGBP_type      =${IGBP_type}
middle_day     =${middle_day}
Tair_constrain =${Tair_constrain}
VPD_sensitive  =${VPD_sensitive}

# default setting
day_time       = ${day_time}
output_2d_grids_only = False
clarify_site   = {'opt': True,
                  'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6', # rainfall problems
                                  'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1',
                                  'AU-Wrr','CN-Din','US-WCr','ZM-Mon' # models miss the simulations of them
                                 ]}
models_calc_LAI   = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
energy_cor        = False
country_code      = None #'AU'

if country_code != None:
    site_names = load_sites_in_country_list(country_code)

# whether only provide 2d_grid csv data and stop the script
if region_name == 'global':
    region = {'name':'global', 'lat':None, 'lon':None}
    regional_sites   = None
elif region_name == 'east_AU':
    region = {'name':'east_AU', 'lat':[-44.5,-10], 'lon':[129,155]}
    regional_sites   = get_regional_site_list(region)
elif region_name == 'west_EU':
    region = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
    regional_sites   = get_regional_site_list(region)
elif region_name == 'north_Am':
    region = {'name':'north_Am', 'lat':[25,52], 'lon':[-125,-65]}
    regional_sites   = get_regional_site_list(region)

write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
                high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
                models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
                add_SMtopXm=add_SMtopXm, add_normalized_SMtopXm=add_normalized_SMtopXm,
                add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,  IGBP_type=IGBP_type,
                country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
                energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
                middle_day=middle_day, VPD_sensitive=VPD_sensitive, Tair_constrain=Tair_constrain,
                add_Xday_mean_EF=add_Xday_mean_EF, select_site=select_site,
                data_selection=data_selection, add_aridity_index=add_aridity_index)

EOF_process_step3

  # Change directory to the submit location
  cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/submit

  # Create the PBS submit script
  cat > "process_step3_raw_data_for_curve_${case_name}.sh" << EOF_submit

#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalbw
#PBS -l walltime=0:30:00
#PBS -l mem=150GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2

python process_step3_raw_data_for_curve_${case_name}.py

EOF_submit

  # Submit the PBS job
  qsub "process_step3_raw_data_for_curve_${case_name}.sh"

done
