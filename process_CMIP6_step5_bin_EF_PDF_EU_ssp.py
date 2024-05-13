'''
Bin the dataset by VPD (and EF) and save in process4_output
Including:
    def bin_VPD
    def bin_VPD_EF
    def write_var_VPD
    def write_var_VPD_EF
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

def bin_val_pdf(bin_input, bin_series, var_predict, uncertain_type='UCRTN_percentile'):

    '''
    bin_input, var_predict, bin_series: 1-d array
    '''

    # Set up the values need to draw
    bin_tot      = len(bin_series)
    bin_pdf      = np.zeros(bin_tot)
    var_vals     = np.zeros(bin_tot)
    var_vals_top = np.zeros(bin_tot)
    var_vals_bot = np.zeros(bin_tot)
    bin_interval = bin_series[2]-bin_series[1]
    var_tot      = len(var_predict)

    # loop each bin
    for i, bin_val in enumerate(bin_series):

        mask_EF = (bin_input > bin_val-bin_interval/2) & (bin_input <= bin_val+bin_interval/2)

        if np.any(mask_EF):

            var_masked = var_predict[mask_EF]

            # calculate mean value
            var_vals[i] = pd.Series(var_masked).mean(skipna=True)

            # calculate total num
            bin_pdf[i]  = np.sum(~np.isnan(var_masked))/var_tot

            if uncertain_type=='UCRTN_one_std':
                # using 1 std as the uncertainty
                var_std   = pd.Series(var_masked).std(skipna=True)
                var_vals_top[i] = var_vals[i] + var_std
                var_vals_bot[i] = var_vals[i] - var_std

            elif uncertain_type=='UCRTN_percentile':
                # using percentile as the uncertainty
                mask_temp = ~ np.isnan(var_masked)
                if np.any(mask_temp):
                    var_vals_top[i] = np.percentile(var_masked, 75)
                    var_vals_bot[i] = np.percentile(var_masked, 25)
                else:
                    var_vals_top[i] = np.nan
                    var_vals_bot[i] = np.nan

            elif uncertain_type=='UCRTN_bootstrap':
                # using bootstrap to get the confidence interval for the unknown distribution dataset
                # mask_temp = ~ np.isnan(var_masked)

                # Generate confidence intervals for the SAMPLE MEAN with bootstrapping:
                var_vals_bot[i], var_vals_top[i] = bootstrap_ci(var_masked, np.mean, n_samples=1000)

        else:
            print('In bin_EF, binned by EF, var_masked = np.nan. Please check why the code goes here')
            print('i=',i, ' bin_val=',bin_val)

            var_vals[i]     = np.nan
            bin_pdf[i]      = np.nan
            var_vals_top[i] = np.nan
            var_vals_bot[i] = np.nan

    return bin_pdf, var_vals, var_vals_top, var_vals_bot

def put_all_CMIP6_EF_VPD_together(CMIP6_path, CMIP6_list, scenario, region={'name':'global','lat':None, 'lon':None},
                                    dist_type=None):

    EF_annual_hist_all = []
    EF_all             = []
    VPD_all            = []

    for i, CMIP6_model in enumerate(CMIP6_list):

        # Read in the selected raw data

        EF_annual_hist = pd.read_csv(f'{CMIP6_path}/EF_annual_hist/CMIP6_DT_filtered_by_VPD_EF_annual_hist_{scenario}_{CMIP6_model}_{region["name"]}.csv',na_values=[''], usecols=['EF_annual_hist'])
        EF             = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{CMIP6_model}_{region["name"]}.csv',na_values=[''], usecols=['EF'])
        VPD            = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{CMIP6_model}_{region["name"]}.csv',na_values=[''],usecols=['VPD'])

        if i == 0:
            EF_annual_hist_all = EF_annual_hist.values
            EF_all             = EF.values
            VPD_all            = VPD.values
        else:
            EF_annual_hist_all = np.concatenate((EF_annual_hist_all, EF_annual_hist.values))
            EF_all             = np.concatenate((EF_all, EF.values))
            VPD_all            = np.concatenate((VPD_all, VPD.values))

    # Save output
    EF_annual_hist_out = pd.DataFrame(EF_annual_hist_all, columns=['EF_annual_hist'])
    EF_out             = pd.DataFrame(EF_all, columns=['EF'])
    VPD_out            = pd.DataFrame(VPD_all, columns=['VPD'])

    EF_annual_hist_out.to_csv(f'./txt/CMIP6/EF_annual_hist/CMIP6_DT_filtered_by_VPD_EF_annual_hist_{scenario}_{region["name"]}.csv')
    EF_out.to_csv(f'./txt/CMIP6/save_csv/CMIP6_DT_filtered_by_VPD_EF_{scenario}_{region["name"]}.csv')
    VPD_out.to_csv(f'./txt/CMIP6/save_csv/CMIP6_DT_filtered_by_VPD_VPD_{scenario}_{region["name"]}.csv')

    return

def put_all_CMIP6_predict_together(CMIP6_path, model_in, CMIP6_list, scenario, region={'name':'global','lat':None, 'lon':None},
                                    dist_type=None):

    var_predict_all = []

    for i, CMIP6_model in enumerate(CMIP6_list):

        # Read in the selected raw data
        var_predict = pd.read_csv(f'{CMIP6_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''], usecols=[model_in])

        if i == 0:
            var_predict_all = var_predict.values
        else:
            var_predict_all = np.concatenate((var_predict_all, var_predict.values))

    # Save output
    var_predict_out = pd.DataFrame(var_predict_all, columns=[model_in])
    var_predict_out.to_csv(f'./txt/CMIP6/predict/predicted_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv')

    return

def write_var_binned(CMIP6_path, scenario,  model_in=None, region={'name':'global','lat':None, 'lon':None},
                     dist_type=None, uncertain_type='UCRTN_bootstrap', bin_type='EF_annual_hist',
                     bin_method=None):

    # Read in the selected raw data
    if bin_type=='EF_annual_hist':
        var_bin      = pd.read_csv(f'{CMIP6_path}/EF_annual_hist/CMIP6_DT_filtered_by_VPD_EF_annual_hist_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['EF_annual_hist'])
        var_predict  = pd.read_csv(f'{CMIP6_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv',
                                   na_values=[''],usecols=[model_in])
        # bin_series   = np.arange(0.005, 1.005, 0.01)
        bin_series   = np.arange(0.05, 1.05, 0.1)

        if bin_method == 'bin':
            file_output  = f'bin_{bin_type}_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{model_in}_{region["name"]}_{dist_type}_coarse.csv'
        elif  bin_method == 'GAM':
            file_output  = f'GAM_{bin_type}_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv'

    elif bin_type=='EF':
        var_bin      = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['EF'])
        var_predict  = pd.read_csv(f'{CMIP6_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv',
                                   na_values=[''],usecols=[model_in])
        # bin_series   = np.arange(0.005, 1.005, 0.01)
        bin_series   = np.arange(0.05, 1.05, 0.1)
        file_output  = f'bin_{bin_type}_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{model_in}_{region["name"]}_{dist_type}_coarse.csv'

    elif bin_type=='VPD':
        var_bin      = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['VPD'])
        var_predict  = pd.read_csv(f'{CMIP6_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv',
                                   na_values=[''],usecols=[model_in])
        bin_series   = np.arange(0.05, 15.05, 0.1)
        file_output  = f'bin_{bin_type}_CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv'

    elif bin_type=='EF_bin_by_VPD':
        var_bin      = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_VPD_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['VPD'])
        var_predict  = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_EF_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['EF'])
        bin_series   = np.arange(0.05, 15.05, 0.1)
        file_output  = f'bin_{bin_type}_CMIP6_DT_filtered_by_VPD_EF_{scenario}_{region["name"]}_{dist_type}.csv'

    elif bin_type=='VPD_bin_by_EF':
        var_bin      = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_EF_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['EF'])
        var_predict  = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_VPD_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['VPD'])
        # bin_series   = np.arange(0.005, 1.005, 0.01)
        bin_series   = np.arange(0.05, 1.05, 0.1)
        file_output  = f'bin_{bin_type}_CMIP6_DT_filtered_by_VPD_VPD_{scenario}_{region["name"]}_{dist_type}_coarse.csv'

    elif bin_type=='VPD_bin_by_EF_annual_hist':
        var_bin      = pd.read_csv(f'{CMIP6_path}/EF_annual_hist/CMIP6_DT_filtered_by_VPD_EF_annual_hist_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['EF_annual_hist'])
        var_predict  = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_VPD_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['VPD'])
        # bin_series   = np.arange(0.005, 1.005, 0.01)
        bin_series   = np.arange(0.05, 1.05, 0.1)
        file_output  = f'bin_{bin_type}_CMIP6_DT_filtered_by_VPD_VPD_{scenario}_{region["name"]}_{dist_type}_coarse.csv'

    elif bin_type=='EF_bin_by_EF_annual_hist':
        var_bin      = pd.read_csv(f'{CMIP6_path}/EF_annual_hist/CMIP6_DT_filtered_by_VPD_EF_annual_hist_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['EF_annual_hist'])
        var_predict  = pd.read_csv(f'{CMIP6_path}/save_csv/CMIP6_DT_filtered_by_VPD_EF_{scenario}_{region["name"]}.csv',
                                   na_values=[''],usecols=['EF'])
        # bin_series   = np.arange(0.005, 1.005, 0.01)
        bin_series   = np.arange(0.05, 1.05, 0.1)
        file_output  = f'bin_{bin_type}_CMIP6_DT_filtered_by_VPD_EF_{scenario}_{region["name"]}_{dist_type}_coarse.csv'

    # if bin

    if bin_method == 'bin':
        # Set up the EF bins
        pdf, vals, vals_top, vals_bot = bin_val_pdf(var_bin.values, bin_series, var_predict.values, uncertain_type)
        # ============ Creat the output dataframe ============
        var_output             = pd.DataFrame(bin_series, columns=['bin_series'])
        var_output['pdf']      = pdf
        var_output['vals']     = vals
        var_output['vals_top'] = vals_top
        var_output['vals_bot'] = vals_bot
        var_output.to_csv(f'./txt/CMIP6/binned/{file_output}')

    elif bin_method == 'GAM':
        dist_type = dist_type#'Gamma'
        fit_GAM_CMIP6_predict(model_in, file_output, bin_series, var_bin.values, var_predict.values, dist_type=dist_type)

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    CMIP6_path     = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/'

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # ========================================= 1D curve ========================================
    model_list     = model_names['model_select_new']
    dist_type      = 'Poisson' #'Gamma' #'Poisson' # 'Linear', None
    uncertain_type = 'UCRTN_bootstrap'

    CMIP6_list   =  ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6',
                     'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']

    scenarios    = ['historical','ssp245']
    region_names = ['east_AU', 'west_EU', 'north_Am']

    # # ======= Get EF, VPD, EF_annual_hist all together =======
    # for region_name in region_names:
    #     region  = get_region_info(region_name)
    #     for scenario in scenarios:
    #         put_all_CMIP6_EF_VPD_together(CMIP6_path, CMIP6_list, scenario, region=region, dist_type=dist_type)

    # # ======= Get CMIP6 model predict all together =======
    # region_name = 'west_EU'
    # scenario    = 'ssp245'
    # region      = get_region_info(region_name)
    # with mp.Pool() as pool:
    #     pool.starmap(put_all_CMIP6_predict_together,
    #                  [(CMIP6_path, model_in, CMIP6_list, scenario, region, dist_type) for model_in in model_list])

    # for region_name in region_names:
    #     region  = get_region_info(region_name)
    #     for scenario in scenarios:
    #         with mp.Pool() as pool:
    #             pool.starmap(put_all_CMIP6_predict_together,
    #                          [(CMIP6_path, model_in, CMIP6_list, scenario, region, dist_type) for model_in in model_list])

    # # ======= binning =======
    # region_name = 'west_EU'
    # scenario    = 'ssp245'
    #
    # bin_method= 'bin'#'GAM'
    # dist_type = 'Poisson'
    # bin_type  = 'VPD_bin_by_EF_annual_hist'
    # model_in  = None
    # region    = get_region_info(region_name)
    # write_var_binned(CMIP6_path, scenario, model_in, region, dist_type, uncertain_type, bin_type, bin_method)
    #
    # bin_type ='EF_bin_by_EF_annual_hist'
    # model_in = None
    # region   = get_region_info(region_name)
    # write_var_binned(CMIP6_path, scenario, model_in, region, dist_type, uncertain_type, bin_type, bin_method)

    bin_type  = 'EF_annual_hist'
    bin_method= 'bin'#'GAM'
    dist_type = 'Poisson'
    region_name = 'west_EU'
    scenario    = 'ssp245'
    region      = get_region_info(region_name)
    with mp.Pool() as pool:
        pool.starmap(write_var_binned, [(CMIP6_path, scenario, model_in, region,
                     dist_type, uncertain_type, bin_type, bin_method) for model_in in model_list])
