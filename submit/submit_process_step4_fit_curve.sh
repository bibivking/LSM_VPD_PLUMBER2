#!/bin/bash

# Set the path to search
PLUMBER2_met_path="/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

# Loop through all files in the path
for file in $(find "$PLUMBER2_met_path" -type f -name "*.nc"); do

  # Extract the file name
  file_name=$(basename "$file")

  # Extract the site name from the file name
  site_name="${file_name%%_*}"

  # site_name="AU-Tum"
  # Print the site name to the console
  echo "$site_name"

  # Set the path to search
  var_name2s=('Qle') # 'Qle_VPD_caused' 'LAI' 'SMtop0.3m' 'SWdown')

  # Loop through all variables in the array
  for var_name2 in "${var_name2s[@]}"; do

    echo "$var_name2"
    case_name="${var_name2}_SM_per_all_models_0-15th_${site_name}_data_selected_STD_annual_model"

    # Print the script name to the console
    echo "process_step4_fit_curve_${case_name}.py"

    # Set the variables
    var_name='"Qle"'
    selected_by='"SM_per_all_models"'
    data_selection='"True"'
    add_Xday_mean_EF='None'
    bounds='[0,15]'
    middle_day='False'
    select_site="'${site_name}'"
    VPD_num_threshold='10' #'200'

    method='"CRV_bins"'
    uncertain_type='"UCRTN_bootstrap"'
    standardize='"STD_annual_model"' #'None' #
    dist_type='None'
    vpd_top_type='"sample_larger_200"'

    day_time='True'
    time_scale='"hourly"'
    veg_fraction='None'
    LAI_range='None'
    IGBP_type='None'

    cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2

    cat > "process_step4_fit_curve_${case_name}.py" << EOF_process_step4
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
from process_step4_fit_curve import *

# Path of PLUMBER 2 dataset
PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
site_names, IGBP_types, clim_types, model_names = load_default_list()

# ======================= Default setting (don't change) =======================
var_name       = ${var_name}
var_name2      = '${var_name2}'
bounds         = ${bounds}
time_scale     = ${time_scale}
selected_by    = ${selected_by}
method         = ${method}
standardize    = ${standardize}
add_Xday_mean_EF=${add_Xday_mean_EF}
select_site    = ${select_site}
data_selection = ${data_selection}

dist_type      = ${dist_type}
VPD_num_threshold = ${VPD_num_threshold}
vpd_top_type   = ${vpd_top_type}

uncertain_type = ${uncertain_type}
day_time       = ${day_time}
IGBP_type      = ${IGBP_type}
LAI_range      = ${LAI_range}
veg_fraction   = ${veg_fraction}
middle_day     = ${middle_day}

# default setting
energy_cor     = False
country_code   = None
clarify_site   = {'opt': True,
                  'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                                  'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1',
                                  'AU-Wrr','CN-Din','US-WCr','ZM-Mon']}
models_calc_LAI = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
if middle_day:
    message_midday = '_midday'
else:
    message_midday = ''

folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                            standardize=standardize, country_code=country_code,
                                            selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                            LAI_range=LAI_range, clarify_site=clarify_site, add_Xday_mean_EF=add_Xday_mean_EF,
                                            data_selection=data_selection)
if select_site == None:
    file_input     = 'raw_data_'+var_name+'_VPD'+file_message+message_midday+'.csv'
else:
    file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'_'+select_site+message_midday+'.csv'

write_var_VPD_parallel(var_name2, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
                       bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
                       standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
                       models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
                       country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type,
                       add_Xday_mean_EF=add_Xday_mean_EF, select_site=select_site, data_selection=data_selection)
gc.collect()

EOF_process_step4

    cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/submit

    cat > "process_step4_fit_curve_${case_name}.sh" << EOF_submit

#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalbw
#PBS -l walltime=0:10:00
#PBS -l mem=50GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2
python process_step4_fit_curve_${case_name}.py

EOF_submit

    qsub "process_step4_fit_curve_${case_name}.sh"

  done
done
