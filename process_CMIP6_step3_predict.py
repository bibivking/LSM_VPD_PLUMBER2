'''
Using GAM model to estimate LH via EF and VPD
Including:
    def calc_predicted_CMIP6_each_model
    def save_predicted_CMIP6_3hourly_parallel
'''

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

def calc_predicted_CMIP6_each_model(CMIP6_txt_path, scenario, CMIP6_model, model_in, var_name='Qle', day_time=False,
                                    region={'name':'global','lat':None, 'lon':None}, dist_type=None):

    # Read data
    if day_time:
        var_output = pd.read_csv(f'{CMIP6_txt_path}/CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{region["name"]}.csv', na_values=[''])
    else:
        var_output = pd.read_csv(f'{CMIP6_txt_path}/CMIP6_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{region["name"]}.csv', na_values=[''])

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
            Qle_pred.to_csv(f'{CMIP6_txt_path}/predicted_CMIP6_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv')
        else:
            Qle_pred.to_csv(f'{CMIP6_txt_path}/predicted_CMIP6_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv')

    gc.collect()

    return

def save_predicted_CMIP6_3hourly_parallel(CMIP6_txt_path, scenario, CMIP6_model, model_list, var_name='Qle',  day_time=False,
    region={'name':'global','lat':None, 'lon':None}, dist_type=None):

    with multiprocessing.Pool() as pool:
        pool.starmap(calc_predicted_CMIP6_each_model,
                     [(CMIP6_txt_path, scenario, CMIP6_model, model_in, var_name, day_time, region, dist_type)
                     for model_in in model_list])
    return

if __name__ == "__main__":

    # Read files
    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/mm3972/data/CMIP6_data/Processed_CMIP6_data/"
    CMIP6_da_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_daily/"
    CMIP6_3h_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_3hourly/"
    CMIP6_txt_path    = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6'


    # Set options
    scenarios         = ['historical','ssp245'] # ['historical','ssp126','ssp245','ssp370']
    percent           = 15
    var_name          = 'Qle'
    day_time          = True
    region_name       = 'east_AU' # 'west_EU', 'north_Am' 'east_AU'
    dist_type         = 'Gamma' #'Poisson' # 'Linear', None
    is_filter         = True

    if region_name == 'global':
        region = {'name':'global', 'lat':None, 'lon':None}
    elif region_name == 'east_AU':
        region = {'name':'east_AU', 'lat':[-44.5,-10], 'lon':[129,155]}
    elif region_name == 'west_EU':
        region = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
    elif region_name == 'north_Am':
        region = {'name':'north_Am', 'lat':[25,52], 'lon':[-125,-65]}

    # Get model lists
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list = model_names['model_select_new']

    for scenario in scenarios:

        # # Calculate predicted CMIP6
        CMIP6_model  =  ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6',
                         'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']

        save_predicted_CMIP6_3hourly_parallel(CMIP6_txt_path, scenario, CMIP6_model, model_list, var_name=var_name, region=region, dist_type=dist_type)
