#!/usr/bin/env python

"""
UNFINISH
"""

import os
import gc
import sys
import glob
import copy
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *
# from check_vars import check_variable_exists

def calc_stat(data_in, outlier_method='IQR', min_percentile=0.05, max_percentile=0.95):

    # Delete nan values
    notNan_mask = ~ np.isnan(data_in)
    data_in     = data_in[notNan_mask]

    # calculate statistics
    Median      = pd.Series(data_in).median()
    P25         = pd.Series(data_in).quantile(0.25)
    P75         = pd.Series(data_in).quantile(0.75)
    IQR         = P75-P25
    if outlier_method=='IQR':
        Minimum     = P25 - 1.5*IQR # pd.Series(data_in).quantile(0.05) # # the lowest data point excluding any outliers.
        Maximum     = P75 + 1.5*IQR #pd.Series(data_in).quantile(0.95) # # the largest data point excluding any outliers. Ref: https://www.simplypsychology.org/boxplots.html#:~:text=When%20reviewing%20a%20box%20plot,whiskers%20of%20the%20box%20plot.&text=For%20example%2C%20outside%201.5%20times,Q3%20%2B%201.5%20*%20IQR).
    elif outlier_method=='percentile':
        Minimum     = pd.Series(data_in).quantile(min_percentile) # # the lowest data point excluding any outliers.
        Maximum     = pd.Series(data_in).quantile(max_percentile) # # the largest data point excluding any outliers. Ref: https://www.simplypsychology.org/boxplots.html#:~:text=When%20reviewing%20a%20box%20plot,whiskers%20of%20the%20box%20plot.&text=For%20example%2C%20outside%201.5%20times,Q3%20%2B%201.5%20*%20IQR).

    # print("Median ", Median)
    # print("P25 ", P25)
    # print("P75 ", P75)
    # print("IQR ", IQR)
    # print("Minimum ", Minimum)
    # print("Maximum ", Maximum)

    return Median, P25, P75, Minimum, Maximum

def write_var_boxplot_metrics(var_name, site_names, PLUMBER2_path, bin_by=None, bounds=30,
                  outlier_method='IQR', min_percentile=0.05, max_percentile=0.95,
                  day_time=False, summer_time=False, IGBP_type=None,
                  clim_type=None, energy_cor=False,
                  hours_precip_free=None):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # ========== read the data ==========
    var_output    = pd.read_csv(f'./txt/{var_name}_all_sites.csv',na_values=[''])

    # Using AR-SLu.nc file to get the model namelist
    f             = nc.Dataset(PLUMBER2_path+"/AR-SLu.nc", mode='r')
    model_in_list = f.variables[var_name + '_models']
    ntime         = len(f.variables['CABLE_time'])
    model_out_list= []

    # Compare each model's output time interval with CABLE hourly interval
    # If the model has hourly output then use the model simulation
    for model_in in model_in_list:
        if len(f.variables[f"{model_in}_time"]) == ntime:
            model_out_list.append(model_in)

    # add obs to draw-out namelist
    if var_name in ['Qle','Qh','NEE']:
        model_out_list.append('obs')
        # model_out_list.append('obs_cor')

    # total site number
    site_num    = len(np.unique(var_output["site_name"]))
    print('Finish reading csv file')

    # ========== select data ==========
    # whether only considers the sites with energy budget corrected fluxs
    if var_name in ['Qle','Qh'] and energy_cor:
        check_obs_cor = var_output['obs_cor']
        check_obs_cor.to_csv(f'./txt/check_obs_cor.csv')

        cor_notNan_mask = ~ np.isnan(var_output['obs_cor'])
        var_output      = var_output[cor_notNan_mask]

    # whether only considers day time
    if day_time:
        # Use radiation as threshold
        day_mask    = (var_output['obs_SWdown'] >= 5)
        var_output  = var_output[day_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    # whether only considers summers
    if summer_time:
        summer_mask = (var_output['month'] > 11) | (var_output['month']< 3)
        # print('np.any(summer_mask)', np.any(summer_mask))
        var_output  = var_output[summer_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 3, site_num=',site_num)

    # whether only considers one type of IGBP
    if IGBP_type!=None:
        IGBP_mask   = (var_output['IGBP_type'] == IGBP_type)
        # print('np.any(IGBP_mask)', np.any(IGBP_mask))
        var_output  = var_output[IGBP_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 4, site_num=',site_num)

    # whether only considers one type of climate type
    if clim_type!=None:
        clim_mask   = (var_output['climate_type'] == clim_type)
        # print('np.any(clim_mask)', np.any(clim_mask))
        var_output  = var_output[clim_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 5, site_num=',site_num)

    # whether only considers observation without precipitation in hours_precip_free hours
    if hours_precip_free!=None:
        rain_mask   = (var_output['hrs_after_precip'] > hours_precip_free)
        var_output  = var_output[rain_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 6, site_num=',site_num)

    print('Finish selecting data')

    # ========== Divide dry and wet periods ==========

    # Calculate EF thresholds
    if bin_by == 'EF_obs':

        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        EF_notNan_mask = ~ np.isnan(var_output['obs_EF'])
        var_output     = var_output[EF_notNan_mask]

        # Select EF<low_bound and EF>high_bound for each site to make sure
        # that every site can contribute to the final VPD lines
        for site_name in site_names:

            # select data for this site
            site_mask       = (var_output['site_name'] == site_name)

            # calculate EF thresholds for this site
            try:
                bin_low  = np.percentile(var_output[site_mask]['obs_EF'], bounds[0])
                bin_high = np.percentile(var_output[site_mask]['obs_EF'], bounds[1])
            except:
                bin_low  = np.nan
                bin_high = np.nan

            # make the mask based on EF thresholds and append it to a full-site long logic array
            try:
                mask_keep = mask_keep.append((var_output[site_mask]['obs_EF'] > bin_low)
                                          & (var_output[site_mask]['obs_EF'] < bin_high))
            except:
                mask_keep = (var_output[site_mask]['obs_EF'] > bin_low) & (var_output[site_mask]['obs_EF'] < bin_high)

        # Mask out the time steps beyond the EF thresholds
        var_output = var_output[mask_keep]

        # free memory
        EF_notNan_mask=None

    elif bin_by == 'EF_model':

        var_output_dry = copy.deepcopy(var_output)

        print( 'Check point 6, np.any(~np.isnan(var_output_dry["model_CABLE"]))=',
               np.any(~np.isnan(var_output_dry["model_CABLE"])) )

        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        for i, model_out_name in enumerate(model_out_list):
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            if model_out_name == 'obs_cor':
                # Use Qle_obs and Qh_obs calculated EF to bin obs_cor, this method may introduce
                # some bias, but keep it for now
                EF_var_name = 'obs_EF'
            else:
                EF_var_name = model_out_name+'_EF'

            mask_keep  = (var_output[EF_var_name] > bounds[0]) & (var_output[EF_var_name] < bounds[1])
            var_output[head+model_out_name] = np.where(mask_keep, var_output[head+model_out_name], np.nan)

    print('Finish dividing dry and wet periods')

    # ============ Choosing fitting or binning ============
    for i, model_out_name in enumerate(model_out_list):
        if 'obs' in model_out_name:
            head = ''
        else:
            head = 'model_'
        if i == 0:
            box_metrics = pd.DataFrame({model_out_name: np.array(calc_stat(var_output[head+model_out_name], outlier_method=outlier_method))})
        else:
            box_metrics[model_out_name] = np.array(calc_stat(var_output[head+model_out_name], outlier_method=outlier_method))

    print(box_metrics)

    # ============ Set the output file name ============
    message = ''

    if day_time:
        message = message + '_daytime'

    if IGBP_type !=None:
        message = message + '_IGBP='+IGBP_type

    if clim_type !=None:
        message = message + '_clim='+clim_type

    # save data
    if bounds[1] > 1:
        box_metrics.to_csv(f'./txt/boxplot_metrics_{var_name}_VPD'+message+'_'+bin_by+'_outlier_by_'+outlier_method+'_'+str(bounds[0])+'-'+str(bounds[1])+'th_coarse.csv')
    else:
        box_metrics.to_csv(f'./txt/boxplot_metrics_{var_name}_VPD'+message+'_'+bin_by+'_outlier_by_'+outlier_method+'_'+str(bounds[0])+'-'+str(bounds[1])+'_coarse.csv')

    return

if __name__ == '__main__':

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
 
    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    outlier_method ='percentile'
    day_time       = True
    energy_cor     = False

    # if var_name == 'NEE':
    #     energy_cor     = False

    # ================== 0-0.4 ==================
    var_name    = 'Qle'  #'TVeg'
    bounds      = [0,0.2] #30

    # for IGBP_type in IGBP_types:
    for clim_type in clim_types:
        write_var_boxplot_metrics(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
                      outlier_method=outlier_method,
                      day_time=day_time, energy_cor=energy_cor,clim_type=clim_type) #IGBP_type=IGBP_type)
        gc.collect()

    var_name    = 'NEE'  #'TVeg'
    bounds      = [0,0.2] #30

    # for IGBP_type in IGBP_types:
    for clim_type in clim_types:
        write_var_boxplot_metrics(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
                      outlier_method=outlier_method,
                      day_time=day_time, energy_cor=energy_cor,clim_type=clim_type) #IGBP_type=IGBP_type)
        gc.collect()

    var_name    = 'Qle'  #'TVeg'
    bounds      = [0.8,1.0] #30

    # for IGBP_type in IGBP_types:
    for clim_type in clim_types:
        write_var_boxplot_metrics(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
                      outlier_method=outlier_method,
                      day_time=day_time, energy_cor=energy_cor,clim_type=clim_type) #IGBP_type=IGBP_type)
        gc.collect()

    var_name    = 'NEE'  #'TVeg'
    bounds      = [0.8,1.0] #30

    # for IGBP_type in IGBP_types:
    for clim_type in clim_types:
        write_var_boxplot_metrics(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
                      outlier_method=outlier_method,
                      day_time=day_time, energy_cor=energy_cor,clim_type=clim_type) #IGBP_type=IGBP_type)
        gc.collect()
