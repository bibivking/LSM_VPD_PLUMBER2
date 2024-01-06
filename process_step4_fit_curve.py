'''



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
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *

def bin_VPD(var_plot, model_out_list, uncertain_type='UCRTN_percentile'):

    # Set up the VPD bins
    vpd_top      = 7.1 #7.04
    vpd_bot      = 0.1 #0.02
    vpd_interval = 0.2 #0.04
    vpd_series   = np.arange(vpd_bot,vpd_top,vpd_interval)

    # Set up the values need to draw
    vpd_tot      = len(vpd_series)
    model_tot    = len(model_out_list)
    vpd_num      = np.zeros((model_tot, vpd_tot))
    var_vals     = np.zeros((model_tot, vpd_tot))
    var_vals_top = np.zeros((model_tot, vpd_tot))
    var_vals_bot = np.zeros((model_tot, vpd_tot))

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

                # calculate mean value
                var_vals[i,j] = var_masked[head+model_out_name].mean(skipna=True)

                vpd_num[i,j]  = np.sum(~np.isnan(var_masked[head+model_out_name]))
                #print('model_out_name=',model_out_name,'j=',j,'vpd_num[i,j]=',vpd_num[i,j])

                if uncertain_type=='UCRTN_one_std':
                    # using 1 std as the uncertainty
                    var_std   = var_masked[head+model_out_name].std(skipna=True)
                    var_vals_top[i,j] = var_vals[i,j] + var_std
                    var_vals_bot[i,j] = var_vals[i,j] - var_std

                elif uncertain_type=='UCRTN_percentile':
                    # using percentile as the uncertainty
                    var_temp  = var_masked[head+model_out_name]
                    mask_temp = ~ np.isnan(var_temp)
                    if np.any(mask_temp):
                        var_vals_top[i,j] = np.percentile(var_temp[mask_temp], 75)
                        var_vals_bot[i,j] = np.percentile(var_temp[mask_temp], 25)
                    else:
                        var_vals_top[i,j] = np.nan
                        var_vals_bot[i,j] = np.nan

                elif uncertain_type=='UCRTN_bootstrap':
                    # using bootstrap to get the confidence interval for the unknown distribution dataset

                    var_temp  = var_masked[head+model_out_name]
                    mask_temp = ~ np.isnan(var_temp)

                    # Generate confidence intervals for the SAMPLE MEAN with bootstrapping:
                    var_vals_bot[i,j], var_vals_top[i,j] = bootstrap_ci(var_temp[mask_temp], np.mean, n_samples=1000)

        else:
            print('In bin_VPD, binned by VPD, var_masked = np.nan. Please check why the code goes here')
            print('j=',j, ' vpd_val=',vpd_val)

            var_vals[:,j]     = np.nan
            vpd_num[:,j]      = np.nan
            var_vals_top[:,j] = np.nan
            var_vals_bot[:,j] = np.nan

    return vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot

def bin_VPD_EF(var_plot, model_out_name, uncertain_type='UCRTN_percentile'):

    # Set up the VPD bins
    vpd_top      = 7.1 #7.04
    vpd_bot      = 0.1 #0.02
    vpd_interval = 0.2 #0.04
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
                  clim_type=None, energy_cor=False,VPD_num_threshold=None,
                  uncertain_type='UCRTN_percentile', models_calc_LAI=None, veg_fraction=None,
                  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                  remove_strange_values=True, country_code=None,
                  hours_precip_free=None, method='CRV_bins'):

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
    var_input = pd.read_csv(f'./txt/process3_output/curves/{file_input}',na_values=[''])
    site_num  = len(np.unique(var_input["site_name"]))

    # ============ Choosing fitting or binning ============
    if method == 'CRV_bins':
        # ============ Bin by VPD ============
        vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot = bin_VPD(var_input, model_out_list, uncertain_type)

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

        var['site_num']    = site_num

    elif method == 'CRV_fit_GAM':
        '''
        fitting GAM curve
        '''

        # ============ Creat the output dataframe ============
        x_top      = 7.04
        x_bot      = 0.02
        x_interval = 0.04

        #reshape for gam
        for i, model_out_name in enumerate(model_out_list):
            print('In GAM fitting for model:', model_out_name)
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            x_values = var_output['VPD']
            y_values = var_output[head+model_out_name]
            vpd_pred, y_pred, y_int = fit_GAM(x_top,x_bot,x_interval,x_values,y_values,n_splines=7,spline_order=3)
            gc.collect()

            if i == 0:
                var      = pd.DataFrame(vpd_pred, columns=['vpd_series'])

            var[model_out_name+'_vals'] = y_pred
            var[model_out_name+'_top']  = y_int[:,0]
            var[model_out_name+'_bot']  = y_int[:,1]
        var['site_num']    = site_num

    # ============ Set the output file name ============
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

    # Checks if a folder exists and creates it if it doesn't
    if not os.path.exists(f'./txt/process4_output/{folder_name}'):
        os.makedirs(f'./txt/process4_output/{folder_name}')

    var.to_csv(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')

    return

def write_var_VPD_EF(var_name, site_names, file_input, PLUMBER2_path, selected_by=None, bounds=None,
                  day_time=False, summer_time=False, IGBP_type=None,
                  clim_type=None, energy_cor=False,VPD_EF_num_threshold=None,
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
    var_input   = pd.read_csv(f'./txt/process3_output/2d_grid/{file_input}',na_values=[''])
    site_num    = len(np.unique(var_input["site_name"]))
  
    # ========= Set output file namte =========
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                    IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                    country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                    uncertain_type=uncertain_type, method=method,
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

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    var_name       = 'Qle'      #'TVeg'
    time_scale     = 'daily'
    selected_by    = 'SLCT_EF_model' # 'EF_model' 
                                # 'EF_obs'
    method         = 'CRV_bins' # 'CRV_bins'
                                # 'CRV_fit_GAM'
    standardize    = None       # 'None'
                                # 'STD_LAI'
                                # 'STD_annual_obs'
                                # 'STD_monthly_obs'
                                # 'STD_monthly_model'
                                # 'STD_daily_obs'

    veg_fraction   = None #[0.7,1]
    day_time       = False  # False for daily
                            # True for half-hour or hourly

    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','Noah-MP']

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)


    # ================ 1D curve ================
    uncertain_type = 'UCRTN_bootstrap'  # 'UCRTN_bootstrap'
                                        # 'UCRTN_percentile'
                                        # 'UCRTN_one_std'
    selected_by    = 'SLCT_EF_model' # 'EF_model' 
    bounds         = [0,0.2] #30
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code, selected_by=selected_by,
                                                bounds=bounds, veg_fraction=veg_fraction, method=method,
                                                clarify_site=clarify_site)

    file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'

    print('file_input',file_input)

    write_var_VPD(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by, bounds=bounds,
                    day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
                    uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
                    country_code=country_code,
                    energy_cor=energy_cor, method=method)
    gc.collect()

    # ================ 2D grid ================
    uncertain_type = 'UCRTN_one_std'# 'UCRTN_bootstrap'
                    # 'UCRTN_percentile'
                    # 'UCRTN_one_std'

    selected_by    = None
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code, selected_by=selected_by,
                                                veg_fraction=veg_fraction, method=method,
                                                clarify_site=clarify_site)

    file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'

    # if the the data point is lower than 10 then the bin's value set as nan
    print('file_input',file_input)

    VPD_EF_num_threshold = 0

    write_var_VPD_EF(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by, bounds=bounds,
                      time_scale=time_scale,day_time=day_time, energy_cor=energy_cor, 
                      VPD_EF_num_threshold=VPD_EF_num_threshold, uncertain_type=uncertain_type,
                       models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
                      clarify_site=clarify_site, standardize=standardize)
