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

def bin_VPD(var_plot, model_out_list, uncertain_type='UCRTN_percentile', var_name='Qle'):

    # Set up the VPD bins
    vpd_top      = 10.001 #7.04
    vpd_bot      = 0.001 #0.02
    vpd_interval = 0.1 #0.04

    vpd_series   = np.arange(vpd_bot,vpd_top,vpd_interval)

    # Set up the values need to draw
    vpd_tot      = len(vpd_series)
    model_tot    = len(model_out_list)
    vpd_num      = np.zeros((model_tot, vpd_tot))
    var_vals     = np.zeros((model_tot, vpd_tot))
    var_vals_top = np.zeros((model_tot, vpd_tot))
    var_vals_bot = np.zeros((model_tot, vpd_tot))

    print('check columns name', var_plot.columns)
    # Binned by VPD
    for j, vpd_val in enumerate(vpd_series):

        mask_vpd       = (var_plot['VPD'] > vpd_val-vpd_interval/2) & (var_plot['VPD'] < vpd_val+vpd_interval/2)

        if np.any(mask_vpd):

            var_masked = var_plot[mask_vpd]

            # Draw the line for different models
            for i, model_out_name in enumerate(model_out_list):

                if 'obs' in model_out_name:
                    head = ''
                else:
                    head = 'model_'

                if 'VPD_caused' in var_name:
                    var_to_study = model_out_name+'_VPD_caused'
                    var_masked[var_to_study] = np.where(np.isnan(var_masked[head+model_out_name]), np.nan, var_masked[var_to_study])
                elif var_name == 'SWdown':
                    var_to_study = model_out_name+'_'+var_name
                    if model_out_name !='obs':
                        # create new columns
                        var_masked[var_to_study]    = np.where(np.isnan(var_masked[head+model_out_name]), np.nan, var_masked['obs_SWdown'])
                    else:
                        print('SWdown obs')
                        # for obs
                        var_masked[var_to_study][:] = np.where(np.isnan(var_masked[head+model_out_name]), np.nan, var_masked['obs_SWdown'])

                elif var_name == 'Qle' or var_name == 'Gs' or 'TVeg' in var_name:
                    var_to_study = head+model_out_name

                else:
                    var_to_study = model_out_name+'_'+var_name
                    # be careful, for using loc, here needs to specify row number to use, e.g. ':'
                    var_masked.loc[:,var_to_study] = np.where(np.isnan(var_masked[head+model_out_name]), np.nan, var_masked[var_to_study])


                # # ==================== Testing ====================
                # print(f"var_to_study: {var_to_study}")
                # print(f"head: {head}, model_out_name: {model_out_name}")
                # print("var_masked columns:", var_masked.columns)
                # if head + model_out_name not in var_masked.columns:
                #     raise KeyError(f"Column {head + model_out_name} does not exist in DataFrame")
                #
                # temp_result = np.where(np.isnan(var_masked[head + model_out_name]), np.nan, var_masked[var_to_study])
                # print("temp_result shape:", temp_result.shape)
                # print("var_masked[var_to_study] shape:", var_masked[var_to_study].shape)
                #
                # if temp_result.shape != var_masked[var_to_study].shape:
                #     raise ValueError("Shapes of the result of np.where and var_masked[var_to_study] do not match")
                # # ===================================================

                # calculate mean value
                var_vals[i,j] = var_masked[var_to_study].mean(skipna=True)
                print('mean', var_vals[i,j])
                # # calculate median value
                # var_vals[i,j] = var_masked[var_to_study].median(skipna=True)

                vpd_num[i,j]  = np.sum(~np.isnan(var_masked[var_to_study]))
                #print('model_out_name=',model_out_name,'j=',j,'vpd_num[i,j]=',vpd_num[i,j])

                if uncertain_type=='UCRTN_one_std':
                    # using 1 std as the uncertainty
                    var_std   = var_masked[var_to_study].std(skipna=True)
                    var_vals_top[i,j] = var_vals[i,j] + var_std
                    var_vals_bot[i,j] = var_vals[i,j] - var_std

                elif uncertain_type=='UCRTN_percentile':
                    # using percentile as the uncertainty
                    var_temp  = var_masked[var_to_study]
                    mask_temp = ~ np.isnan(var_temp)
                    if np.any(mask_temp):
                        var_vals_top[i,j] = np.percentile(var_temp[mask_temp], 95)
                        var_vals_bot[i,j] = np.percentile(var_temp[mask_temp], 5)
                    else:
                        var_vals_top[i,j] = np.nan
                        var_vals_bot[i,j] = np.nan

                elif uncertain_type=='UCRTN_bootstrap':
                    # using bootstrap to get the confidence interval for the unknown distribution dataset

                    var_temp  = var_masked[var_to_study]
                    mask_temp = ~ np.isnan(var_temp)

                    # Generate confidence intervals for the SAMPLE MEAN with bootstrapping:
                    var_vals_bot[i,j], var_vals_top[i,j] = bootstrap_ci(var_temp[mask_temp], np.mean, n_samples=10000)

                    # # Generate confidence intervals for the SAMPLE MEAN with bootstrapping:
                    # var_vals_bot[i,j], var_vals_top[i,j] = bootstrap_ci(var_temp[mask_temp], np.median, n_samples=1000)

        else:
            print('In bin_VPD, binned by VPD, var_masked = np.nan. Please check why the code goes here')
            print('j=',j, ' vpd_val=',vpd_val)

            var_vals[:,j]     = np.nan
            vpd_num[:,j]      = np.nan
            var_vals_top[:,j] = np.nan
            var_vals_bot[:,j] = np.nan
        print(var_vals)

    return vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot

def bin_VPD_EF(var_plot, model_out_name, uncertain_type='UCRTN_percentile'):

    # Set up the VPD bins
    vpd_top      = 10.001 #7.04
    vpd_bot      = 0.001 #0.02
    vpd_interval = 0.1 #0.04

    # vpd_top      = 7.1 #7.04
    # vpd_bot      = 0.1 #0.02
    # vpd_interval = 0.2 #0.04
    vpd_series   = np.arange(vpd_bot,vpd_top,vpd_interval)

    # Set up EF bins
    EF_top      = 1.025 #7.04
    EF_bot      = 0.025 #0.02
    EF_interval = 0.05  #0.04
    EF_series   = np.arange(EF_bot,EF_top,EF_interval)

    # Set up the values need to draw
    vpd_tot      = len(vpd_series)
    EF_tot       = len(EF_series)

    # Initilize variables
    vpd_num      = np.zeros((vpd_tot, EF_tot))
    var_vals     = np.zeros((vpd_tot, EF_tot))
    var_vals_top = np.zeros((vpd_tot, EF_tot))
    var_vals_bot = np.zeros((vpd_tot, EF_tot))

    # Binned by VPD
    for i, vpd_val in enumerate(vpd_series):
        for j, EF_val in enumerate(EF_series):

            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            mask_vpd    = (var_plot['VPD'] > vpd_val-vpd_interval/2) & (var_plot['VPD'] < vpd_val+vpd_interval/2)
            mask_EF     = (var_plot[model_out_name+'_EF'] > EF_val-EF_interval/2) & (var_plot[model_out_name+'_EF'] < EF_val+EF_interval/2)
            mask_vpd_EF = (mask_vpd) & (mask_EF)
            var_tmp     = var_plot[head+model_out_name]

            if np.any(mask_vpd) and np.any(mask_EF):

                var_masked = var_tmp[mask_vpd_EF]

                # calculate mean value
                var_vals[i,j] = np.nanmean(var_masked)
                vpd_num[i,j]  = np.sum(~np.isnan(var_masked))

                if uncertain_type=='UCRTN_one_std':
                    # using 1 std as the uncertainty
                    var_std           = np.nanstd(var_masked)
                    var_vals_top[i,j] = var_vals[i,j] + var_std
                    var_vals_bot[i,j] = var_vals[i,j] - var_std

                elif uncertain_type=='UCRTN_percentile':
                    # using percentile as the uncertainty
                    mask_nan = ~ np.isnan(var_masked)
                    if np.any(mask_temp):
                        var_vals_top[i,j] = np.percentile(var_masked[mask_nan], 75)
                        var_vals_bot[i,j] = np.percentile(var_masked[mask_nan], 25)
                    else:
                        var_vals_top[i,j] = np.nan
                        var_vals_bot[i,j] = np.nan
                    # print(model_out_name, 'var_vals[i,:]', var_vals[i,:])
            else:
                print('In bin_VPD_EF, binned by VPD & EF, var_masked = np.nan. Please check why the code goes here')
                print('j=',j, ' vpd_val=',vpd_val)

                var_vals[i,j]     = np.nan
                vpd_num[i,j]      = np.nan
                var_vals_top[i,j] = np.nan
                var_vals_bot[i,j] = np.nan

    return vpd_series, EF_series, vpd_num, var_vals, var_vals_top, var_vals_bot

def write_var_VPD(var_name, site_names, file_input, PLUMBER2_path, selected_by=None, bounds=None,
                  day_time=False, summer_time=False, IGBP_type=None, time_scale=None,
                  clim_type=None, energy_cor=False,VPD_num_threshold=None,LAI_range=None,
                  uncertain_type='UCRTN_percentile', models_calc_LAI=None, veg_fraction=None,
                  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                  remove_strange_values=True, country_code=None,
                  hours_precip_free=None, method='CRV_bins',dist_type='Linear'):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''
    # save data
    if var_name == 'NEE':
        var_name = 'NEP'

    # Get model names
    if var_name == 'Gs':
        site_names, IGBP_types, clim_types, model_names = load_default_list()
        model_out_list = model_names['model_select_new']
    else:
        model_out_list = get_model_out_list(var_name)

    # Read in the selected raw data
    var_input = pd.read_csv(f'./txt/process3_output/curves/{file_input}',na_values=[''],low_memory=False)
    # site_num  = len(np.unique(var_input["site_name"]))

    print('var_input is', var_input)
    print("var_input['CABLE_SMtop1m']",var_input['CABLE_SMtop1m'])

    # ============ Choosing fitting or binning ============
    if method == 'CRV_bins':

        # ============ Bin by VPD ============
        vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot = bin_VPD(\
                          var_input, model_out_list, uncertain_type, var_name)

        # ============ Creat the output dataframe ============
        var = pd.DataFrame(vpd_series, columns=['vpd_series'])

        for i, model_out_name in enumerate(model_out_list):

            var[model_out_name+'_vpd_num'] = vpd_num[i,:]

            if VPD_num_threshold == None:
                var[model_out_name+'_vals'] = var_vals[i,:]
                var[model_out_name+'_top']  = var_vals_top[i,:]
                var[model_out_name+'_bot']  = var_vals_bot[i,:]
            else:
                var[model_out_name+'_vals'] = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
                                                  var_vals[i,:], np.nan)
                var[model_out_name+'_top']  = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
                                                  var_vals_top[i,:], np.nan)
                var[model_out_name+'_bot']  = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
                                                  var_vals_bot[i,:], np.nan)

        # var['site_num']    = site_num

    elif method == 'CRV_fit_GAM_simple' or method == 'CRV_fit_GAM_complex':

        '''
        fitting GAM curve
        '''

        # ============ Creat the output dataframe ============
        x_top      = 10.001
        x_bot      = 0.001
        x_interval = 0.1

        #reshape for gam
        # MMY modified
        for i, model_out_name in enumerate(model_out_list):
            print('In GAM fitting for model:', model_out_name)
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            x_values = var_input['VPD']
            y_values = var_input[head+model_out_name]

            print(x_values.type())
            print(y_values.type())

            if method == 'CRV_fit_GAM_simple':
                vpd_pred, y_pred, y_int = fit_GAM_simple(x_top,x_bot,x_interval,x_values,y_values,n_splines=7,spline_order=3)
            elif method == 'CRV_fit_GAM_complex':
                vpd_pred, y_pred, y_int = fit_GAM_complex(model_out_name, var_name, folder_name, file_message, x_top,x_bot,
                                                          x_interval,x_values,y_values,dist_type=dist_type)
            gc.collect()

            if i == 0:
                var      = pd.DataFrame(vpd_pred, columns=['vpd_series'])

            var[model_out_name+'_vals'] = y_pred
            var[model_out_name+'_top']  = y_int[:,0]
            var[model_out_name+'_bot']  = y_int[:,1]
        # var['site_num']    = site_num

    # ============ Set the output file name ============
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                veg_fraction=veg_fraction, LAI_range=LAI_range,
                                                uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

    # Checks if a folder exists and creates it if it doesn't
    if not os.path.exists(f'./txt/process4_output/{folder_name}'):
        os.makedirs(f'./txt/process4_output/{folder_name}')
    if method == 'CRV_fit_GAM_complex':
        var.to_csv(f'./txt/process4_output/{folder_name}/{var_name}{file_message}_{dist_type}.csv')
    else:
        var.to_csv(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')

    return

def write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=None, bounds=None,
                  day_time=False, summer_time=False, IGBP_type=None, time_scale=None,
                  clim_type=None, energy_cor=False,VPD_num_threshold=None, LAI_range=None,
                  uncertain_type='UCRTN_percentile', models_calc_LAI=None, veg_fraction=None,
                  clarify_site={'opt':False,'remove_site':None}, standardize=None, select_site=None,
                  remove_strange_values=True, country_code=None, vpd_top_type='to_10',middle_day=None,
                  hours_precip_free=None, method='CRV_bins',dist_type='Linear', add_Xday_mean_EF=None,
                  data_selection=True):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    print('file_input',file_input)

    # save data
    if var_name == 'NEE':
        var_name = 'NEP'

    # Get model lists
    if var_name == 'Gs':
        site_names, IGBP_types, clim_types, model_names = load_default_list()
        model_out_list = model_names['model_select_new']
    else:
        model_out_list = get_model_out_list(var_name)
        print('model_out_list',model_out_list)

    # Read in the selected raw data
    var_input = pd.read_csv(f'./txt/process3_output/curves/{file_input}',na_values=[''])
    # site_num  = len(np.unique(var_input["site_name"]))

    # ============ Set the output file name ============
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                uncertain_type=uncertain_type, clarify_site=clarify_site,
                                                add_Xday_mean_EF=add_Xday_mean_EF,data_selection=data_selection)

    # Checks if a folder exists and creates it if it doesn't
    if not os.path.exists(f'./txt/process4_output/{folder_name}'):
        os.makedirs(f'./txt/process4_output/{folder_name}')

    # ============ Choosing fitting or binning ============
    if method == 'CRV_bins':

        print("if method == 'CRV_bins'",var_name)
        '''
        make curves by binning data
        output: one file
        '''

        # ============ Bin by VPD ============
        vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot = bin_VPD(var_input, model_out_list, uncertain_type, var_name)

        # ============ Creat the output dataframe ============
        var = pd.DataFrame(vpd_series, columns=['vpd_series'])

        for i, model_out_name in enumerate(model_out_list):

            var[model_out_name+'_vpd_num'] = vpd_num[i,:]

            if VPD_num_threshold == None:
                var[model_out_name+'_vals'] = var_vals[i,:]
                var[model_out_name+'_top']  = var_vals_top[i,:]
                var[model_out_name+'_bot']  = var_vals_bot[i,:]
            else:
                var[model_out_name+'_vals'] = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
                                                  var_vals[i,:], np.nan)
                var[model_out_name+'_top']  = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
                                                  var_vals_top[i,:], np.nan)
                var[model_out_name+'_bot']  = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
                                                  var_vals_bot[i,:], np.nan)

        if middle_day:
            message_midday = '_midday'
        else:
            message_midday = ''

        # var['site_num']    = site_num
        if select_site != None:
            var.to_csv(f'./txt/process4_output/{folder_name}/{var_name}{file_message}_{select_site}{message_midday}.csv')
        else:
            var.to_csv(f'./txt/process4_output/{folder_name}/{var_name}{file_message}{message_midday}.csv')

    elif method == 'CRV_fit_GAM_simple' or method == 'CRV_fit_GAM_complex':

        '''
        fitting GAM curve
        '''

        # if select_site != None:
        #     print("method='CRV_fit_GAM_simple' or 'CRV_fit_GAM_complex' don't support write optimized GAM model out for one selected site yet")
        #     return
        # else:
        # ============ Check whether the folder save GAM_fit data exist ============
        if not os.path.exists(f'./txt/process4_output/{folder_name}/GAM_fit'):
            os.makedirs(f'./txt/process4_output/{folder_name}/GAM_fit')

        # ============ Creat the output dataframe ============
        x_bot      = 0.001
        x_interval = 0.1

        # Use multiprocessing to fit GAM models in parallel
        if vpd_top_type == 'sample_larger_200':

            # Find x_top
            vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot = bin_VPD(var_input, model_out_list, uncertain_type)
            x_top = {}

            for i, model_out_name in enumerate(model_out_list):
                try:
                    tmp                   = np.where(vpd_num[i,:]>=VPD_num_threshold, 1, 0)
                    x_top[model_out_name] = vpd_series[np.argwhere(tmp==1)[-1]]
                except:
                    print('Totally ',np.sum(tmp),'VPD bins have data points >=',VPD_num_threshold)
                    x_top[model_out_name] = np.nan
            print(x_top)

            with mp.Pool() as pool:
                pool.starmap(fit_GAM_for_model, [(folder_name, file_message, var_name, model_in, x_top[model_in], x_bot, x_interval,
                            var_input['VPD'],  var_input[get_header(model_in) + model_in], method, dist_type, vpd_top_type)
                            for model_in in model_out_list])

        elif vpd_top_type == 'to_10':
            x_top      = 10.001
            with mp.Pool() as pool:
                pool.starmap(fit_GAM_for_model, [(folder_name, file_message, var_name, model_in, x_top, x_bot, x_interval,
                            var_input['VPD'],  var_input[get_header(model_in) + model_in], method, dist_type, vpd_top_type)
                            for model_in in model_out_list])

    return

def fit_GAM_for_model(folder_name, file_message, var_name, model_in, x_top, x_bot, x_interval,
                      x_values, y_values, method='CRV_fit_GAM_simple', dist_type='Linear',
                      vpd_top_type='to_10'):

    # Exclude VPD ==0
    x_values = np.where(x_values >0.05, x_values, np.nan)
    y_values = np.where(x_values >0.05, y_values, np.nan)

    # If there are more than 10 data points to make the curve
    if np.sum(~np.isnan(y_values)) > 10:

        if method == 'CRV_fit_GAM_simple':
            vpd_pred, y_pred, y_int = fit_GAM_simple(x_top,x_bot,x_interval,x_values,y_values,n_splines=7,spline_order=3)
        elif method == 'CRV_fit_GAM_complex':
            vpd_pred, y_pred, y_int = fit_GAM_complex(model_in, var_name, folder_name, file_message, \
                                                    x_top,x_bot,x_interval,x_values,y_values,dist_type,vpd_top_type)
        if ~np.all(np.isnan(vpd_pred)):
            var_fitted              = pd.DataFrame(vpd_pred, columns=['vpd_pred'])
            var_fitted['y_pred']    = y_pred
            var_fitted['y_int_top'] = y_int[:,0]
            var_fitted['y_int_bot'] = y_int[:,1]

            if vpd_top_type == 'sample_larger_200':
                subfolder_name = f'{dist_type}_greater_200_samples'

            elif vpd_top_type == 'to_10':
                subfolder_name = f'{dist_type}_to_10'

            var_fitted.to_csv(f'./txt/process4_output/{folder_name}/{subfolder_name}/GAM_fit/{var_name}{file_message}_{model_in}_{dist_type}.csv')

    return

def write_var_VPD_EF(var_name, site_names, file_input, PLUMBER2_path, selected_by=None, bounds=None,
                  day_time=False, summer_time=False, IGBP_type=None,
                  clim_type=None, energy_cor=False,VPD_EF_num_threshold=None,LAI_range=None,
                  uncertain_type='UCRTN_percentile', models_calc_LAI=None, veg_fraction=None,
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
    var_input   = pd.read_csv(f'./txt/process3_output/2d_grid/{file_input}',na_values=[''],low_memory=False)
    # site_num    = len(np.unique(var_input["site_name"]))

    # ========= Set output file namte =========
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                    IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                    country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                    uncertain_type=uncertain_type, method=method, LAI_range=LAI_range,
                    clarify_site=clarify_site)

    # Checks if a folder exists and creates it if it doesn't
    if not os.path.exists(f'./txt/process4_output/{folder_name}'):
        os.makedirs(f'./txt/process4_output/{folder_name}')

    # ============ Bin by VPD & EF ============
    for i, model_out_name in enumerate(model_out_list):
        vpd_series, EF_series, vpd_num, var_vals, var_vals_top, var_vals_bot =\
                            bin_VPD_EF(var_input, model_out_name, uncertain_type)

        if VPD_EF_num_threshold != None:
            var_vals     = np.where(vpd_num < VPD_EF_num_threshold, np.nan, var_vals)
            var_vals_top = np.where(vpd_num < VPD_EF_num_threshold, np.nan, var_vals_top)
            var_vals_bot = np.where(vpd_num < VPD_EF_num_threshold, np.nan, var_vals_bot)
            vpd_num      = np.where(vpd_num < VPD_EF_num_threshold, np.nan, vpd_num)

        np.savetxt(f'./txt/process4_output/{folder_name}/{var_name}_Numbers'+file_message+'_'+model_out_name+'.csv',vpd_num)
        np.savetxt(f'./txt/process4_output/{folder_name}/{var_name}_Values'+file_message+'_'+model_out_name+'.csv',var_vals)
        np.savetxt(f'./txt/process4_output/{folder_name}/{var_name}_Top_bounds'+file_message+'_'+model_out_name+'.csv',var_vals_top)
        np.savetxt(f'./txt/process4_output/{folder_name}/{var_name}_Bot_bounds'+file_message+'_'+model_out_name+'.csv',var_vals_bot)

    return

# if __name__ == "__main__":
    #
    # # Path of PLUMBER 2 dataset
    # PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    #
    # site_names, IGBP_types, clim_types, model_names = load_default_list()
    #
    # # ======================= Default setting (dont change) =======================
    # var_name       = 'Qle'       #'TVeg'
    # time_scale     = 'hourly'   #'daily'
    # selected_by    = 'EF_model' # 'EF_model'
    #                             # 'EF_obs'
    # method         = 'CRV_bins' # 'CRV_bins'
    #                             # 'CRV_fit_GAM_simple'
    #                             # 'CRV_fit_GAM_complex'
    # standardize    = None       # 'None'
    #                             # 'STD_LAI'
    #                             # 'STD_annual_obs'
    #                             # 'STD_monthly_obs'
    #                             # 'STD_monthly_model'
    #                             # 'STD_daily_obs'
    # LAI_range      = None
    # veg_fraction   = None   #[0.7,1]
    #
    # clarify_site      = {'opt': True,
    #                      'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
    #                      'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    # models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
    #
    # day_time       = False  # False for daily
    #                         # True for half-hour or hourly
    #
    # if time_scale == 'hourly':
    #     day_time   = True
    #
    # energy_cor     = False
    # if var_name == 'NEE':
    #     energy_cor = False
    #
    # # Set regions/country
    # country_code   = None#'AU'
    # if country_code != None:
    #     site_names = load_sites_in_country_list(country_code)
    #
    # # ========================================= 1D curve ========================================
    #
    # # ======================= Curves for each site =======================
    # if 0:
    #     var_name       = 'TVeg'
    #     uncertain_type = 'UCRTN_bootstrap'
    #     selected_by    = 'EF_model'
    #     # method         = 'CRV_fit_GAM_complex'
    #     method         = 'CRV_bins'
    #     dist_type      = None #'Linear' #'Poisson' # 'Gamma'
    #     VPD_num_threshold = 10 # for one site: 10 #for all sites: 200
    #     vpd_top_type   = 'sample_larger_200' # 'to_10' #
    #     for select_site in site_names:
    #
    #         # 0 < EF < 0.2
    #         bounds         = [0,0.2] #30
    #         folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                     standardize=standardize, country_code=country_code,
    #                                                     selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                     LAI_range=LAI_range, clarify_site=clarify_site) #
    #
    #         file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'_'+select_site+'.csv'
    #
    #         write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                     bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                     standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                     models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,select_site=select_site,
    #                     country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         gc.collect()
    #
    #         # 0.8 < EF < 1.0
    #         bounds         = [0.8,1.] #30
    #         folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                     standardize=standardize, country_code=country_code,
    #                                                     selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                     LAI_range=LAI_range, clarify_site=clarify_site) #
    #
    #         file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'_'+select_site+'.csv'
    #
    #         write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                     bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                     standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                     models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, select_site=select_site,
    #                     country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #         gc.collect()
    #
    # # ======================= Curves for all sites =======================
    #
    # # ======================= all sites ======================
    # if 1:
    #     # standardize    = 'STD_SWdown_LAI_SMtop1m'
    #     uncertain_type = 'UCRTN_bootstrap' #'UCRTN_percentile' #
    #     selected_by    = 'EF_obs'
    #     # method         = 'CRV_fit_GAM_complex'
    #     method         = 'CRV_bins'
    #     dist_type      = None #'Linear' #'Poisson' # 'Gamma'
    #     VPD_num_threshold = 200 # for one site: 10 #for all sites: 200
    #     vpd_top_type   = 'sample_larger_200' #'sample_larger_200'#
    #     LAI_range      = None
    #     middle_day     = False
    #
    #     if middle_day:
    #         message_midday = '_midday'
    #     else:
    #         message_midday = ''
    #
    #     bounds_all     = [[0.8,1.0]]#,[0.8,1.0]]#[[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]] # [0.0,0.2],
    #     #[[0,15],[15,30],[30,50], [50,70],[70,90],[90,100]]
    #     #[[0.8,1.0]]#[[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]]
    #
    #     # LAI_ranges     = [[0, 1.], [1.,2.], [2.,4.], [4.,10.]]
    #     IGBP_types     = ['GRA', 'OSH', 'SAV', 'WSA', 'CSH', 'DBF', 'ENF', 'EBF', 'MF', 'WET', 'CRO']
    #
    #     for bounds in bounds_all:
    #         print(f'calculate {bounds}')
    #
    #         selected_by    = 'EF_model'
    #         standardize    = 'STD_normalized_SMtopXm'
    #
    #         for IGBP_type in IGBP_types:
    #         # for LAI_range in LAI_ranges:
    #             var_name       = 'Qle'
    #             folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                         standardize=standardize, country_code=country_code,
    #                                         selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                         LAI_range=LAI_range, clarify_site=clarify_site) #
    #             file_input     = 'raw_data_'+var_name+'_VPD'+file_message+message_midday+'.csv'
    #             var_name       = 'Qle_VPD_caused'
    #             write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                                         bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                                         standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                                         models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
    #                                         country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #             gc.collect()
    #
    #
    #             # var_name       = 'TVeg'
    #             # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #             #                             standardize=standardize, country_code=country_code,
    #             #                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #             #                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #             # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+message_midday+'.csv'
    #             # var_name       = 'TVeg_VPD_caused'
    #             # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #             #                             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #             #                             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #             #                             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
    #             #                             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #             # gc.collect()
    #
    #         # var_name       = 'LAI'
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #                             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #                             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #                             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
    #         #                             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #
    #
    #         # var_name       = 'nonTVeg'
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                             standardize=standardize, country_code=country_code,
    #         #                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #         #                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #         # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+message_midday+'.csv'
    #
    #         # var_name       = 'nonTVeg' #'nonTVeg_VPD_caused'
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #                             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #                             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #                             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
    #         #                             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #
    #
    #         #
    #         # var_name       = 'TVeg'
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                             standardize=standardize, country_code=country_code,
    #         #                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #         #                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #         # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+message_midday+'.csv'
    #         # # var_name       = 'nonTVeg_VPD_caused'
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #                             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #                             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #                             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
    #         #                             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #         #
    #         # var_name       = 'nonTVeg'
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                             standardize=standardize, country_code=country_code,
    #         #                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #         #                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #         # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+message_midday+'.csv'
    #         # # var_name       = 'nonTVeg_VPD_caused'
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #                             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #                             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #                             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
    #         #                             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #         # # selected_by    = 'EF_model'
    #         # var_name       = 'Qle'
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                             standardize=standardize, country_code=country_code,
    #         #                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #         #                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #         # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+message_midday+'.csv'
    #         # var_name       = 'Qle_VPD_caused'
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #                             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #                             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #                             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
    #         #                             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #
    # # ======================= all sites-standardized by STD_LAI_SMtop1m or STD_SMtop1m ======================
    # if 0:
    #     var_name       = 'TVeg'
    #     standardize    = 'STD_SWdown_SMtop1m'
    #     uncertain_type = 'UCRTN_bootstrap'#'UCRTN_percentile' #
    #     method         = 'CRV_bins'
    #     dist_type      = None #'Linear' #'Poisson' # 'Gamma'
    #     VPD_num_threshold = 200 # for one site: 10 #for all sites: 200
    #     vpd_top_type   = 'sample_larger_200'#'to_10' #'sample_larger_200' #
    #     LAI_range      = None
    #
    #     # 0 < EF < 0.2
    #     bounds         = [0,0.2] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                 standardize=standardize, country_code=country_code,
    #                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                                 country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #     gc.collect()
    #
    #     # # 0.2 < EF < 0.4
    #     # bounds         = [0.2,0.4] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # # 0.4 < EF < 0.6
    #     # bounds         = [0.4,0.6] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site)
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # # 0.6 < EF < 0.8
    #     # bounds         = [0.6,0.8] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site)
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # 0.8 < EF < 1.0
    #     bounds         = [0.8,1.] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                 standardize=standardize, country_code=country_code,
    #                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    # # ======================= climate types ======================
    # if 0:
    #     LAI_range     = None
    #     veg_fraction  = None
    #     IGBP_types    = ['GRA', 'DBF', 'ENF', 'EBF']
    #
    #     for clim_type in clim_types:
    #
    #         # 0 < EF < 0.2
    #         bounds         = [0,0.2]
    #         folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                     standardize=standardize, country_code=country_code, selected_by=selected_by,
    #                                                     bounds=bounds, veg_fraction=veg_fraction, clim_type=clim_type, clarify_site=clarify_site)
    #         file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #
    #         write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                     bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                     standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                     models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, clim_type=clim_type,
    #                     country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         gc.collect()
    #
    #         # # 0.2 < EF < 0.4
    #         # bounds         = [0.2,0.4]
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    #         #                                             bounds=bounds, veg_fraction=veg_fraction, clim_type=clim_type,  clarify_site=clarify_site)
    #         # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #         #
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, clim_type=clim_type,
    #         #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #         #
    #         # # 0.4 < EF < 0.6
    #         # bounds         = [0.4,0.6]
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    #         #                                             bounds=bounds, veg_fraction=veg_fraction, clim_type=clim_type, clarify_site=clarify_site)
    #         # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #         #
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, clim_type=clim_type,
    #         #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #         #
    #         # # 0.6 < EF < 0.8
    #         # bounds         = [0.6,0.8]
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    #         #                                             bounds=bounds, veg_fraction=veg_fraction, clim_type=clim_type, clarify_site=clarify_site)
    #         # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #         #
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, clim_type=clim_type,
    #         #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #
    #         # 0.8 < EF < 1.0
    #         bounds         = [0.8,1.0]
    #         folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                     standardize=standardize, country_code=country_code, selected_by=selected_by,
    #                                                     bounds=bounds, veg_fraction=veg_fraction, clim_type=clim_type, clarify_site=clarify_site)
    #         file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #
    #         write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                     bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                     standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                     models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, clim_type=clim_type,
    #                     country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         gc.collect()
    #
    # # ======================== EF ==============================
    # if 0:
    #     LAI_range      = None
    #     veg_fraction   = None
    #     IGBP_type      = None
    #     clim_type      = None
    #
    #     # 0 < EF < 0.2
    #     var_name       = 'Qle'
    #     bounds         = [0,0.2]
    #
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code, selected_by=selected_by,
    #                                                 bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
    #     file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'EF'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, clim_type=clim_type,
    #                 country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #     gc.collect()
    #
    #     # 0.2 < EF < 0.4
    #     var_name       = 'Qle'
    #     bounds         = [0.2,0.4] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'EF'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    #     # 0.4 < EF < 0.6
    #     var_name       = 'Qle'
    #     bounds         = [0.4,0.6] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site)
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'EF'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    #     # 0.6 < EF < 0.8
    #     var_name       = 'Qle'
    #     bounds         = [0.6,0.8] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site)
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'EF'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    #     # 0.8 < EF < 1.0
    #     var_name       = 'Qle'
    #     bounds         = [0.8,1.] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'EF'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    # # ======================== SWdown =============================
    # if 0:
    #     uncertain_type = 'UCRTN_bootstrap'#'UCRTN_percentile' #
    #     method         = 'CRV_bins'
    #     dist_type      = None #'Linear' #'Poisson' # 'Gamma'
    #     VPD_num_threshold = 200 # for one site: 10 #for all sites: 200
    #     vpd_top_type   = 'sample_larger_200'#'to_10' #'sample_larger_200' #
    #
    #     LAI_range      = None
    #     veg_fraction   = None
    #     IGBP_type      = None
    #     clim_type      = None
    #
    #     # # 0 < EF < 0.2
    #     # var_name       = 'Qle'
    #     # bounds         = [0,0.2]
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    #     #                                             bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
    #     # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # var_name       = 'SWdown'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, clim_type=clim_type,
    #     #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # # 0.2 < EF < 0.4
    #     # var_name       = 'Qle'
    #     # bounds         = [0.2,0.4] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # var_name       = 'SWdown'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # # 0.4 < EF < 0.6
    #     # var_name       = 'Qle'
    #     # bounds         = [0.4,0.6] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site)
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # var_name       = 'SWdown'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # # 0.6 < EF < 0.8
    #     # var_name       = 'Qle'
    #     # bounds         = [0.6,0.8] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site)
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # var_name       = 'SWdown'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # 0.8 < EF < 1.0
    #     var_name       = 'Qle'
    #     bounds         = [0.8,1.] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'SWdown'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    # # ======================== LAI ==============================
    # if 0:
    #     uncertain_type = 'UCRTN_bootstrap' #'UCRTN_percentile' #
    #     method         = 'CRV_bins'
    #     dist_type      = None #'Linear' #'Poisson' # 'Gamma'
    #     VPD_num_threshold = 200 # for one site: 10 #for all sites: 200
    #     vpd_top_type   = 'sample_larger_200'#'to_10' #'sample_larger_200' #
    #
    #     LAI_range      = None
    #     veg_fraction   = None
    #     IGBP_type      = None
    #     clim_type      = None
    #
    #     # 0 < EF < 0.2
    #     var_name       = 'Qle'
    #     bounds         = [0,0.2]
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code, selected_by=selected_by,
    #                                                 bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
    #     file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'LAI'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, clim_type=clim_type,
    #                 country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #     gc.collect()
    #
    #     # 0.2 < EF < 0.4
    #     var_name       = 'Qle'
    #     bounds         = [0.2,0.4] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'LAI'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    #     # 0.4 < EF < 0.6
    #     var_name       = 'Qle'
    #     bounds         = [0.4,0.6] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site)
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'LAI'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    #     # 0.6 < EF < 0.8
    #     var_name       = 'Qle'
    #     bounds         = [0.6,0.8] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site)
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'LAI'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    #     # 0.8 < EF < 1.0
    #     var_name       = 'Qle'
    #     bounds         = [0.8,1.] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'LAI'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    # # ======================= SMtop1m ======================
    # if 0:
    #     uncertain_type = 'UCRTN_bootstrap' #'UCRTN_percentile' #
    #     method         = 'CRV_bins'
    #     dist_type      = None #'Linear' #'Poisson' # 'Gamma'
    #     VPD_num_threshold = 200 # for one site: 10 #for all sites: 200
    #     vpd_top_type   = 'sample_larger_200'#'to_10' #'sample_larger_200' #
    #
    #     LAI_range      = None
    #     veg_fraction   = None
    #     IGBP_type      = None
    #     clim_type      = None
    #
    #     # 0 < EF < 0.2
    #     var_name       = 'Qle'
    #     bounds         = [0,0.2] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'SMtop1m'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #     gc.collect()
    #
    #     # # 0.2 < EF < 0.4
    #     # var_name       = 'Qle'
    #     # bounds         = [0.2,0.4] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # var_name       = 'SMtop1m'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # # 0.4 < EF < 0.6
    #     # var_name       = 'Qle'
    #     # bounds         = [0.4,0.6] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # var_name       = 'SMtop1m'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # # 0.6 < EF < 0.8
    #     # var_name       = 'Qle'
    #     # bounds         = [0.6,0.8] #30
    #     # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #     #                                             standardize=standardize, country_code=country_code,
    #     #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #     #                                             LAI_range=LAI_range, clarify_site=clarify_site) #
    #     # file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     # var_name       = 'SMtop1m'
    #     # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #     #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #     #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #     #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #     #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #     # gc.collect()
    #
    #     # 0.8 < EF < 1.0
    #     var_name       = 'Qle'
    #     bounds         = [0.8,1.] #30
    #     folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                 standardize=standardize, country_code=country_code,
    #                                                 selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                 LAI_range=LAI_range, clarify_site=clarify_site) #
    #     file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #     var_name       = 'SMtop1m'
    #     write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                 country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    #     gc.collect()
    #
    # # ======================= LAI ======================
    # if 0:
    #     # LAI classification
    #     LAI_ranges     = [[0.,1.],
    #                       [1.,2.],
    #                       [2.,4.],
    #                       [4.,10.],] #30
    #
    #     for LAI_range in LAI_ranges:
    #         print('Calculate LAI_range',LAI_range)
    #
    #         # 0<EF<0.2
    #         bounds         = [0,0.2] #30
    #         folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                     standardize=standardize, country_code=country_code,
    #                                                     selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                     LAI_range=LAI_range, clarify_site=clarify_site)
    #         file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #         write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                     bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                     standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                     models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                     country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         gc.collect()
    #
    #         # 0.8<EF<1.
    #         bounds         = [0.8,1.] #30
    #         folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                     standardize=standardize, country_code=country_code,
    #                                                     selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                                     LAI_range=LAI_range, clarify_site=clarify_site)
    #         file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #         write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                     bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                     standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                     models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
    #                     country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         gc.collect()
    #
    # # ======================= Veg types ======================
    # if 0:
    #     # Different land cover
    #     LAI_range     = None
    #     veg_fraction  = None
    #     # IGBP_types    = ['GRA', 'DBF', 'ENF', 'EBF']
    #
    #     for IGBP_type in IGBP_types:
    #
    #         # 0 < EF < 0.2
    #         bounds         = [0,0.2]
    #         folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                     standardize=standardize, country_code=country_code, selected_by=selected_by,
    #                                                     bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
    #         file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #
    #         write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                     bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                     standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                     models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, IGBP_type=IGBP_type,
    #                     country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         gc.collect()
    #
    #         # # 0.2 < EF < 0.4
    #         # bounds         = [0.2,0.4]
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    #         #                                             bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
    #         # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, IGBP_type=IGBP_type,
    #         #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #
    #
    #         # # 0.4 < EF < 0.6
    #         # bounds         = [0.4,0.6]
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    #         #                                             bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
    #         # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, IGBP_type=IGBP_type,
    #         #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #
    #         # # 0.6 < EF < 0.8
    #         # bounds         = [0.6,0.8]
    #         # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #         #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    #         #                                             bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
    #         # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #
    #         # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #         #             bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #         #             standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #         #             models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, IGBP_type=IGBP_type,
    #         #             country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         # gc.collect()
    #
    #         # 0.8 < EF < 1.0
    #         bounds         = [0.8,1.]
    #         folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                                     standardize=standardize, country_code=country_code, selected_by=selected_by,
    #                                                     bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
    #         file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #
    #         write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
    #                 bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
    #                 standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
    #                 models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, IGBP_type=IGBP_type,
    #                 country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    #         gc.collect()
    #
    # # # # Low vegetation coverage
    # # # bounds         = [0,0.2] #30
    # # # veg_fraction   = [0,0.3]
    # # # LAI_range      = None
    # # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # # #                                             bounds=bounds, veg_fraction=veg_fraction,clarify_site=clarify_site)
    # # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # # # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by, bounds=bounds,
    # # #                 day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
    # # #                 uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # # #                 country_code=country_code,
    # # #                 energy_cor=energy_cor, method=method)
    # # # gc.collect()
    #
    # # # bounds         = [0.8,1.]
    # # # veg_fraction   = [0,0.3]
    # # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # # #                                             bounds=bounds, veg_fraction=veg_fraction, clarify_site=clarify_site)
    # # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # # # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by, bounds=bounds,
    # # #                 day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
    # # #                 uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # # #                 country_code=country_code,
    # # #                 energy_cor=energy_cor, method=method)
    # # # gc.collect()
    #
    #
    # # # # High vegetation coverage
    # # # bounds         = [0,0.2] #30
    # # # veg_fraction   = [0.7,1.]
    # # # LAI_range      =None
    # # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # # #                                             bounds=bounds, veg_fraction=veg_fraction, clarify_site=clarify_site)
    # # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # # # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by, bounds=bounds,
    # # #                 day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
    # # #                 uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # # #                 country_code=country_code,
    # # #                 energy_cor=energy_cor, method=method)
    # # # gc.collect()
    #
    #
    # # # bounds         = [0.8,1.]
    # # # veg_fraction   = [0.7,1.]
    # # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # # #                                             bounds=bounds, veg_fraction=veg_fraction, clarify_site=clarify_site)
    # # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # # # write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by, bounds=bounds,
    # # #                 day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
    # # #                 uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # # #                 country_code=country_code,
    # # #                 energy_cor=energy_cor, method=method)
    # # # gc.collect()
    #
    #
    # # ## ========================================= 2D grid ========================================
    # # # uncertain_type = 'UCRTN_one_std'# 'UCRTN_bootstrap'
    # # #                 # 'UCRTN_percentile'
    # # #                 # 'UCRTN_one_std'
    #
    # # # selected_by    = None
    # # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # # #                                             veg_fraction=veg_fraction, method=method,
    # # #                                             clarify_site=clarify_site)
    #
    # # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    #
    # # # # if the the data point is lower than 10 then the bin's value set as nan
    # # # print('file_input',file_input)
    #
    # # # VPD_EF_num_threshold = 0
    #
    # # # write_var_VPD_EF(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by, bounds=bounds,
    # # #                   time_scale=time_scale,day_time=day_time, energy_cor=energy_cor,
    # # #                   VPD_EF_num_threshold=VPD_EF_num_threshold, uncertain_type=uncertain_type,
    # # #                    models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # # #                   clarify_site=clarify_site, standardize=standardize)
