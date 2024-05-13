'''
Bin the dataset by VPD (and EF) and save in process4_output
Including:
    def bin_Xvar
    def bin_Xvar_EF
    def write_var_Xvar
    def write_var_Xvar_EF
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import gc
import sys
import glob
import copy
import numpy as np
import pandas as pd
import netCDF4 as nc
import multiprocessing as mp
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *

def bin_X_Y(var_input, Xvar_name, Yvar_name, model_in, EF_bounds=None, uncertain_type='UCRTN_percentile'):

    print('model_in',model_in)

    if EF_bounds!=None:
        EF_mask   = (var_input[model_in+'_EF'] > EF_bounds[0]) & (var_input[model_in+'_EF'] < EF_bounds[1])
        var_input = var_input[EF_mask]

    print('var_input["obs_Tair"]',var_input["obs_Tair"])
    print('var_input["VPD"]',var_input["VPD"])

    # Set up the VPD bins
    if Xvar_name == 'VPD':
        x_top      = 10.05 #7.04
        x_bot      = 0.05 #0.02
        x_interval = 0.1 #0.04

    # Set up EF bins
    if Yvar_name == 'obs_Tair':
        y_top      = 51 #7.04
        y_bot      = -20 #0.02
        y_interval = 1. 
        var_input[Yvar_name] -= 273.15

    # Make x and y series
    x_series   = np.arange(x_bot, x_top, x_interval)
    y_series   = np.arange(y_bot, y_top, y_interval)

    # Set up the values need to draw
    x_tot      = len(x_series)
    y_tot      = len(y_series)

    # Initilize variables
    data_point_sum = np.zeros((x_tot, y_tot))
    var_vals       = np.zeros((x_tot, y_tot))
    var_vals_top   = np.zeros((x_tot, y_tot))
    var_vals_bot   = np.zeros((x_tot, y_tot))

    # Binned by VPD
    for i, x_val in enumerate(x_series):
        for j, y_val in enumerate(y_series):

            print('x_val',x_val,'y_val',y_val)
            
            if 'obs' in model_in:
                head = ''
            else:
                head = 'model_'

            mask_x    = (var_input[Xvar_name] > x_val-x_interval/2) & (var_input[Xvar_name] < x_val+x_interval/2)
            mask_y    = (var_input[Yvar_name] > y_val-y_interval/2) & (var_input[Yvar_name] < y_val+y_interval/2)
            mask_x_y  = (mask_x) & (mask_y)
            var_tmp   = var_input[head+model_in]

            if np.any(mask_x) and np.any(mask_y):

                print('x_val',x_val,'y_val',y_val, 'has data point')

                var_masked = var_tmp[mask_x_y]

                # calculate mean value
                var_vals[i,j]        = np.nanmean(var_masked)
                data_point_sum[i,j]  = np.sum(~np.isnan(var_masked))

                if uncertain_type=='UCRTN_one_std':
                    # using 1 std as the uncertainty
                    var_std           = np.nanstd(var_masked)
                    var_vals_top[i,j] = var_vals[i,j] + var_std
                    var_vals_bot[i,j] = var_vals[i,j] - var_std

                elif uncertain_type=='UCRTN_percentile':
                    # using percentile as the uncertainty
                    mask_nan          = ~ np.isnan(var_masked)
                    if np.any(mask_nan):
                        var_vals_top[i,j] = np.percentile(var_masked[mask_nan], 75)
                        var_vals_bot[i,j] = np.percentile(var_masked[mask_nan], 25)
                    else:
                        var_vals_top[i,j] = np.nan
                        var_vals_bot[i,j] = np.nan
                    # print(model_in, 'var_vals[i,:]', var_vals[i,:])
            else:
                print('In bin_X_Y, binned by '+Xvar_name+' & '+Yvar_name+', var_masked = np.nan. Please check why the code goes here')
                print('j=',j, ' x_val=',x_val)

                var_vals[i,j]        = np.nan
                data_point_sum[i,j]  = np.nan
                var_vals_top[i,j]    = np.nan
                var_vals_bot[i,j]    = np.nan

    return x_series, y_series, data_point_sum, var_vals, var_vals_top, var_vals_bot

def parallel_bin_process(var_input, Xvar_name, Yvar_name, EF_bounds, uncertain_type, 
                         sum_threshold, folder_name, file_message, model_in):
    
    """Bins data for a single model input in a separate process."""

    x_series, y_series, data_point_sum, var_vals, var_vals_top, var_vals_bot = \
        bin_X_Y(var_input, Xvar_name, Yvar_name, model_in, EF_bounds, uncertain_type)
    
    if sum_threshold:
        var_vals[data_point_sum < sum_threshold]       = np.nan
        var_vals_top[data_point_sum < sum_threshold]   = np.nan
        var_vals_bot[data_point_sum < sum_threshold]   = np.nan
        data_point_sum[data_point_sum < sum_threshold] = np.nan

    np.savetxt(f'./txt/process4_output/2d_grids/{folder_name}/num_{var_name}_{Xvar_name}_{Yvar_name}{file_message}_{model_in}.csv', data_point_sum)
    np.savetxt(f'./txt/process4_output/2d_grids/{folder_name}/val_{var_name}_{Xvar_name}_{Yvar_name}{file_message}_{model_in}.csv', var_vals)
    np.savetxt(f'./txt/process4_output/2d_grids/{folder_name}/top_{var_name}_{Xvar_name}_{Yvar_name}{file_message}_{model_in}.csv', var_vals_top)
    np.savetxt(f'./txt/process4_output/2d_grids/{folder_name}/bot_{var_name}_{Xvar_name}_{Yvar_name}{file_message}_{model_in}.csv', var_vals_bot)

    return

def save_2d_grids(var_name, Xvar_name, Yvar_name, EF_bounds, file_input, selected_by=None, 
                  day_time=False, summer_time=False, IGBP_type=None,
                  clim_type=None, energy_cor=False,sum_threshold=None,LAI_range=None,
                  uncertain_type='UCRTN_percentile', veg_fraction=None,
                  clarify_site={'opt':False,'remove_site':None}, standardize=None):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # save data
    if var_name == 'NEE':
        var_name = 'NEP'

    # Get model lists
    model_out_list = get_model_out_list(var_name)

    # Read in the selected raw data
    var_input      = pd.read_csv(f'./txt/process3_output/2d_grid/{file_input}',na_values=[''])

    # ========= Set output file namte =========
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                    IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                    country_code=country_code, selected_by=selected_by, bounds=EF_bounds, veg_fraction=veg_fraction,
                    uncertain_type=uncertain_type, method=method, LAI_range=LAI_range,
                    clarify_site=clarify_site)

    # Checks if a folder exists and creates it if it doesn't
    if not os.path.exists(f'./txt/process4_output/2d_grids/{folder_name}'):
        os.makedirs(f'./txt/process4_output/2d_grids/{folder_name}')

    with mp.Pool() as pool:
        pool.starmap(parallel_bin_process, [(var_input, Xvar_name, Yvar_name, EF_bounds, 
                                            uncertain_type, sum_threshold, folder_name, file_message, model_in)
                                            for model_in in model_out_list])

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_Xvar_PLUMBER2/nc_files/"

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # ======================= Default setting (dont change) =======================
    var_name       = 'Qle'       #'TVeg'
    time_scale     = 'hourly'   #'daily'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'
    method         = 'CRV_bins' # 'CRV_bins'
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

    clarify_site      = {'opt': True,
                         'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                         'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

    day_time       = False  # False for daily
                            # True for half-hour or hourly

    if time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # ====================== Custom setting ========================
    var_name         = 'nonTVeg'
    selected_by      = 'EF_model'
    method           = 'CRV_bins'
    sum_threshold    = 20

    Xvar_name        = 'VPD' #'obs_SWdown' # 'obs_Tair'# units: K #'VPD' # 
    Yvar_name        = 'obs_Tair'
    uncertain_type   = 'UCRTN_one_std'
                     # 'UCRTN_percentile'
                     # 'UCRTN_one_std'
    
    file_input       = 'raw_data_Qle_VPD_hourly_RM16_DT.csv'

    EF_bounds_all    = [ [0,   0.2],
                         [0.2, 0.4], ]                      
    # [0.4, 0.6],
    # [0.6, 0.8], 
    # [0.8, 1. ]] 

    for EF_bounds in EF_bounds_all:
        save_2d_grids(var_name, Xvar_name, Yvar_name, EF_bounds, file_input, selected_by=selected_by, 
                    day_time=day_time, energy_cor=energy_cor,sum_threshold=sum_threshold,
                    uncertain_type=uncertain_type,clarify_site=clarify_site)