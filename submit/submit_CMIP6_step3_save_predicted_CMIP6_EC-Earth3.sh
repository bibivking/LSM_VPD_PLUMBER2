#!/bin/bash

# CMIP6 model list
CMIP6_model='EC-Earth3'
scenario='historical' # 'ssp245' #'historical'
model_list=('CABLE' 'CABLE-POP-CN' 'CHTESSEL_Ref_exp1' 'CLM5a' 'GFDL' 'JULES_GL9' 'JULES_GL9_withLAI' 'MATSIRO' 'MuSICA' 'NASAEnt' 'NoahMPv401' 'ORC2_r6593' 'ORC3_r8120' 'QUINCY' 'STEMMUS-SCOPE' 'obs')

echo $model_list

for model_in in ${model_list[@]}; do

cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2

# Print the CMIP6 model name
echo "$model_in"

cat > save_predicted_CMIP6_3hourly_${scenario}_${CMIP6_model}_${model_in}.py << EOF_predicted_CMIP6
#!/usr/bin/env python
__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from matplotlib.patches import Polygon
import copy
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *
import multiprocessing

def get_PLUMBER2_curve_names(bounds):

    # Path of PLUMBER 2 dataset
    PLUMBER2_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    var_name       = 'Qle'      #'TVeg'
    time_scale     = 'hourly'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'

    method         = 'CRV_fit_GAM_complex' # 'CRV_bins'
                                # 'CRV_fit_GAM_simple'
                                # 'CRV_fit_GAM_complex'

    standardize    = None       # 'None'
                                # 'STD_LAI'
                                # 'STD_annual_obs'
                                # 'STD_monthly_obs'
                                # 'STD_monthly_model'
                                # 'STD_daily_obs'
    LAI_range      = None
    veg_fraction   = None   #[0.7,1]

    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}

    day_time       = True  # False for daily
                            # True for half-hour or hourly
    if var_name == 'Gs' and time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # ================ 1D curve ================
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code, method=method,
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site) #

    return folder_name, file_message

def calc_predicted_CMIP6_each_model(CMIP6_txt_path, scenario, CMIP6_model, model_in, var_name='Qle',
                                    day_time=False, region={'name':'global','lat':None, 'lon':None},
                                    dist_type=None):
    # Read data
    if day_time:
        var_output = pd.read_csv(f'{CMIP6_txt_path}/CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{region["name"]}.csv', na_values=[''])
    else:
        var_output = pd.read_csv(f'{CMIP6_txt_path}/CMIP6_{var_name}_{scenario}_{CMIP6_model}_{region["name"]}.csv', na_values=[''])

    print('model', model_in)
    print('Check var_output', var_output)

    # divide Qle_1d, VPD_1d and EF_1d by EF_1d values
    EF         = var_output['EF'][:]
    Qle        = var_output['Qle'][:]
    VPD        = var_output['VPD'][:]
    EF_02_mask = (EF >= 0)   & (EF < 0.2)
    EF_04_mask = (EF >= 0.2) & (EF < 0.4)
    EF_06_mask = (EF >= 0.4) & (EF < 0.6)
    EF_08_mask = (EF >= 0.6) & (EF < 0.8)
    EF_10_mask = (EF >= 0.8) & (EF <= 1.)

    VPD_02     = VPD[EF_02_mask]
    VPD_04     = VPD[EF_04_mask]
    VPD_06     = VPD[EF_06_mask]
    VPD_08     = VPD[EF_08_mask]
    VPD_10     = VPD[EF_10_mask]

    Qle_02     = Qle[EF_02_mask]
    Qle_04     = Qle[EF_04_mask]
    Qle_06     = Qle[EF_06_mask]
    Qle_08     = Qle[EF_08_mask]
    Qle_10     = Qle[EF_10_mask]
    #
    # EF_02      = EF[EF_02_mask]
    # EF_04      = EF[EF_04_mask]
    # EF_06      = EF[EF_06_mask]
    # EF_08      = EF[EF_08_mask]
    # EF_10      = EF[EF_10_mask]

    # put CMIP6 simulated Qle
    if model_in == 'CABLE':
        Qle_pred =  pd.DataFrame(np.concatenate((Qle_02,Qle_04,Qle_06,Qle_08,Qle_10)), columns=['CMIP6'])

    bounds = [0,0.2]
    folder_name, file_message = get_PLUMBER2_curve_names(bounds)
    Qle_pred_02 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_02, dist_type=dist_type)

    bounds = [0.2,0.4]
    folder_name, file_message = get_PLUMBER2_curve_names(bounds)
    Qle_pred_04 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_04, dist_type=dist_type)

    bounds = [0.4,0.6]
    folder_name, file_message = get_PLUMBER2_curve_names(bounds)
    Qle_pred_06 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_06, dist_type=dist_type)

    bounds = [0.6,0.8]
    folder_name, file_message = get_PLUMBER2_curve_names(bounds)
    Qle_pred_08 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_08, dist_type=dist_type)

    bounds = [0.8,1.]
    folder_name, file_message = get_PLUMBER2_curve_names(bounds)
    Qle_pred_10 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_10, dist_type=dist_type)
    if model_in == 'CABLE':
        Qle_pred[model_in] = np.concatenate((Qle_pred_02,Qle_pred_04,Qle_pred_06,Qle_pred_08,Qle_pred_10))
    else:
        Qle_pred =  pd.DataFrame(np.concatenate((Qle_pred_02,Qle_pred_04,Qle_pred_06,Qle_pred_08,Qle_pred_10)),
                    columns=[model_in])
    if day_time:
        if dist_type == None:
            Qle_pred.to_csv(f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv')
        else:
            Qle_pred.to_csv(f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv')
    else:
        if dist_type == None:
            Qle_pred.to_csv(f'{CMIP6_txt_path}/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv')
        else:
            Qle_pred.to_csv(f'{CMIP6_txt_path}/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv')
    gc.collect()

    return

if __name__ == "__main__":

    # Read files
    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/mm3972/data/CMIP6_data/Processed_CMIP6_data/"
    CMIP6_da_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_daily/"
    CMIP6_3h_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_3hourly/"
    CMIP6_txt_path    = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6'
    percent           = 15
    var_name          = 'Qle'
    day_time          = True
    region            = {'name':'global', 'lat':None, 'lon':None}

    # Get model lists
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    dist_type   = 'Gamma'
    scenario    = '${scenario}'
    CMIP6_model = '${CMIP6_model}'
    model_in    = '${model_in}'

    calc_predicted_CMIP6_each_model(CMIP6_txt_path, scenario, CMIP6_model, model_in, var_name=var_name,
                                    day_time=day_time, region=region, dist_type=dist_type)

EOF_predicted_CMIP6

cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/submit

cat > submit_predicted_CMIP6_${scenario}_${CMIP6_model}_${model_in}.sh << EOF_submit
#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalsr
#PBS -l walltime=2:00:00
#PBS -l mem=500GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+gdata/fs38+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2
python save_predicted_CMIP6_3hourly_${scenario}_${CMIP6_model}_${model_in}.py

EOF_submit

qsub submit_predicted_CMIP6_${scenario}_${CMIP6_model}_${model_in}.sh

done
