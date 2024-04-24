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

def calc_stat(data_in, outlier_method='IQR', min_percentile=0.05, max_percentile=0.95):

    # Delete nan values
    notNan_mask = ~ np.isnan(data_in)
    data_in     = data_in[notNan_mask]

    # calculate statistics
    # Median      = pd.Series(data_in).median()
    print("max(data_in) ", max(data_in))
    print("min(data_in) ", min(data_in))

    Mean        = pd.Series(data_in).mean()
    P25         = pd.Series(data_in).quantile(0.25)
    P75         = pd.Series(data_in).quantile(0.75)
    IQR         = P75-P25
    if outlier_method=='IQR':
        Minimum     = P25 - 1.5*IQR # pd.Series(data_in).quantile(0.05) # # the lowest data point excluding any outliers.
        Maximum     = P75 + 1.5*IQR #pd.Series(data_in).quantile(0.95) # # the largest data point excluding any outliers. Ref: https://www.simplypsychology.org/boxplots.html#:~:text=When%20reviewing%20a%20box%20plot,whiskers%20of%20the%20box%20plot.&text=For%20example%2C%20outside%201.5%20times,Q3%20%2B%201.5%20*%20IQR).
    elif outlier_method=='percentile':
        Minimum     = pd.Series(data_in).quantile(min_percentile) # # the lowest data point excluding any outliers.
        Maximum     = pd.Series(data_in).quantile(max_percentile) # # the largest data point excluding any outliers. Ref: https://www.simplypsychology.org/boxplots.html#:~:text=When%20reviewing%20a%20box%20plot,whiskers%20of%20the%20box%20plot.&text=For%20example%2C%20outside%201.5%20times,Q3%20%2B%201.5%20*%20IQR).


    # print("P25 ", P25)
    # print("P75 ", P75)
    # print("IQR ", IQR)
    # print("Minimum ", Minimum)
    # print("Maximum ", Maximum)

    # return (Median, P25, P75, Minimum, Maximum)
    return (Mean, P25, P75, Minimum, Maximum)

def calc_stat_percentile(data_in):

    # Delete nan values
    print('data_in.shape',data_in.shape)
    data_in_tmp  = pd.Series(data_in)
    data_in      = data_in_tmp.dropna()

    # calculate statistics
    Mean           = data_in.mean()
    percentiles    = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #np.arange(0.01,1.00,0.01)
    val_percentile = np.zeros(len(percentiles)+1)

    for i, percentile in enumerate(percentiles):
        val_percentile[i] = data_in.quantile(percentile)

    val_percentile[-1] = Mean

    return val_percentile

def calc_predicted_CMIP6_metrics(CMIP6_txt_path, var_name, model_in, CMIP6_model_list, outlier_method='percentile',dist_type=None):

    # ============ Setting for plotting ============
    input_files = []
    var_input   = []

    # Read all CMIP6 dataset
    for i, CMIP6_model in enumerate(CMIP6_model_list):
        print(CMIP6_model)
        if day_time:
            if dist_type!=None:
                var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=[model_in])
            else:
                var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv', na_values=[''], usecols=[model_in])
        else:
            if dist_type!=None:
                var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=[model_in])
            else:
                var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv', na_values=[''], usecols=[model_in])

        # var_input.append(var_tmp[model_in])

        if i == 0:
            var_input = var_tmp[model_in].values #np.array(var_tmp.values)
        else:
            var_input = np.concatenate((var_input, var_tmp[model_in].values))#np.array(var_tmp.values))

    metrics  = pd.DataFrame(calc_stat(var_input, outlier_method=outlier_method), columns=[model_in])

    if day_time:
        if dist_type!=None:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_DT_{var_name}_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv')
        else:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_DT_{var_name}_{scenario}_{model_in}_{region["name"]}.csv')
    else:
        if dist_type!=None:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_{var_name}_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv')
        else:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_{var_name}_{scenario}_{model_in}_{region["name"]}.csv')


    # ===================== calc metrics for CMIP6 original var ===================

    # I save CMIP6 var in CABLE's csv files

    if model_in == 'CABLE':

        # Read all CMIP6 dataset
        for i, CMIP6_model in enumerate(CMIP6_model_list):

            print(CMIP6_model)
            if day_time:
                if dist_type !=None:
                    var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=['CMIP6'])
                else:
                    var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv', na_values=[''], usecols=['CMIP6'])
            else:
                if dist_type !=None:
                    var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=['CMIP6'])
                else:
                    var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv', na_values=[''], usecols=['CMIP6'])


            if i == 0:
                var_input = var_tmp['CMIP6'].values
            else:
                var_input = np.concatenate((var_input, var_tmp['CMIP6'].values))

        metrics  = pd.DataFrame(calc_stat(var_input, outlier_method=outlier_method), columns=['CMIP6'])

        if day_time:
            if dist_type !=None:
                metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_DT_{var_name}_{scenario}_CMIP6_{region["name"]}_{dist_type}.csv')
            else:
                metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_DT_{var_name}_{scenario}_CMIP6_{region["name"]}.csv')

        else:
            if dist_type !=None:
                metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_{var_name}_{scenario}_CMIP6_{region["name"]}_{dist_type}.csv')
            else:
                metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_{var_name}_{scenario}_CMIP6_{region["name"]}.csv')

    return

def calc_predicted_CMIP6_diff_metrics(CMIP6_txt_path, var_name, model_in, CMIP6_model_list, outlier_method='percentile',dist_type=None):

    # ============ Setting for plotting ============
    input_files = []
    var_input   = []

    # Read all CMIP6 dataset
    for i, CMIP6_model in enumerate(CMIP6_model_list):
        print(CMIP6_model)
        if day_time:
            if dist_type!=None:
                var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=[model_in])
                obs_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_obs_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=['obs'])
            else:
                var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv', na_values=[''], usecols=[model_in])
                obs_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_obs_{region["name"]}.csv', na_values=[''], usecols=['obs'])
        else:
            if dist_type!=None:
                var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=[model_in])
                obs_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_obs_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=['obs'])
            else:
                var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv', na_values=[''], usecols=[model_in])
                obs_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_obs_{region["name"]}.csv', na_values=[''], usecols=['obs'])

        # var_input.append(var_tmp[model_in])

        if i == 0:
            var_diff = var_tmp[model_in].values - obs_tmp['obs'].values
        else:
            var_diff = np.concatenate((var_diff, var_tmp[model_in].values - obs_tmp['obs'].values))

    metrics  = pd.DataFrame(calc_stat(var_diff, outlier_method=outlier_method), columns=[model_in])

    if day_time:
        if dist_type!=None:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_DT_{var_name}_diff_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv')
        else:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_DT_{var_name}_diff_{scenario}_{model_in}_{region["name"]}.csv')
    else:
        if dist_type!=None:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_{var_name}_diff_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv')
        else:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_{var_name}_diff_{scenario}_{model_in}_{region["name"]}.csv')

    return

def calc_predicted_CMIP6_percentiles(CMIP6_txt_path, var_name, model_in, CMIP6_model_list, dist_type=None):

    # ============ Setting for plotting ============
    input_files = []
    var_input   = []

    # Read all CMIP6 dataset
    for i, CMIP6_model in enumerate(CMIP6_model_list):
        print(CMIP6_model)
        if day_time:
            var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=[model_in])
        else:
            var_tmp = pd.read_csv(f'{CMIP6_txt_path}/predict/predicted_CMIP6_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''], usecols=[model_in])
        # var_input.append(var_tmp[model_in])

        if i == 0:
            var_input = var_tmp[model_in].values #np.array(var_tmp.values)
        else:
            var_input = np.concatenate((var_input, var_tmp[model_in].values))#np.array(var_tmp.values))

    val_percentile  = pd.DataFrame(calc_stat_percentile(var_input), columns=[model_in])

    if day_time:
        val_percentile.to_csv(f'{CMIP6_txt_path}/percentile_CMIP6_DT_{var_name}_{scenario}_{model_in}_{region["name"]}.csv')
    else:
        val_percentile.to_csv(f'{CMIP6_txt_path}/percentile_CMIP6_{var_name}_{scenario}_{model_in}_{region["name"]}.csv')

    return

if __name__ == "__main__":

    # Read files
    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/mm3972/data/CMIP6_data/Processed_CMIP6_data/"
    CMIP6_da_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_daily/"
    CMIP6_3h_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_3hourly/"
    CMIP6_txt_path    = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6'

    var_name          = 'Qle'
    day_time          = True
    # region          = {'name':'global', 'lat':None, 'lon':None}
    # region          = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
    # region          = {'name':'north_Am', 'lat':[25,58], 'lon':[-125,-65]}
    region            = {'name':'east_AU', 'lat':[-44.5,-22], 'lon':[138,155]}

    # Get model lists
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list        = model_names['model_select_new']

    CMIP6_model_list  = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6',
                         'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']

    scenario    = 'historical'
    model_in    = 'JULES_GL9'
    dist_type   = 'Gamma'

    # calc_predicted_CMIP6_percentiles(CMIP6_txt_path, var_name, model_in, CMIP6_model_list, dist_type=dist_type)
    # calc_predicted_CMIP6_metrics(CMIP6_txt_path, var_name, model_in, CMIP6_model_list, outlier_method='percentile',dist_type=dist_type)
    calc_predicted_CMIP6_diff_metrics(CMIP6_txt_path, var_name, model_in, CMIP6_model_list, outlier_method='percentile',dist_type=dist_type)

