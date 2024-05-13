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

def bin_Xvar(var_plot, Xvar_name, model_out_list, uncertain_type='UCRTN_percentile'):

    # Set up the VPD bins
    if Xvar_name == 'obs_Tair':
        Xvar_top      = 50.
        Xvar_bot      = -50. 
        Xvar_interval = 1. #0.04
    elif Xvar_name == 'obs_SWdown':
        Xvar_top      = 1000.
        Xvar_bot      = -20. 
        Xvar_interval = 10. #0.04
    elif Xvar_name == 'obs_Qair':
        Xvar_top      = 0.05
        Xvar_bot      = 0
        Xvar_interval = 0.0005 #0.04
    

    Xvar_series   = np.arange(Xvar_bot,Xvar_top,Xvar_interval)

    # Give input values 
    var_input     = var_plot[Xvar_name]


    # Set up the values need to draw
    Xvar_tot     = len(Xvar_series)
    model_tot    = len(model_out_list)
    Xvar_num     = np.zeros((model_tot, Xvar_tot))
    var_vals     = np.zeros((model_tot, Xvar_tot))
    var_vals_top = np.zeros((model_tot, Xvar_tot))
    var_vals_bot = np.zeros((model_tot, Xvar_tot))

    # Binned by VPD
    for j, Xvar_val in enumerate(Xvar_series):

        mask_Xvar       = (var_input > Xvar_val-Xvar_interval/2) & (var_input < Xvar_val+Xvar_interval/2)

        if np.any(mask_Xvar):

            var_masked = var_plot[mask_Xvar]

            # Draw the line for different models
            for i, model_out_name in enumerate(model_out_list):

                if 'obs' in model_out_name:
                    head = ''
                else:
                    head = 'model_'

                # calculate mean value
                var_vals[i,j] = var_masked[head+model_out_name].mean(skipna=True)

                # # calculate median value
                # var_vals[i,j] = var_masked[head+model_out_name].median(skipna=True)

                Xvar_num[i,j]  = np.sum(~np.isnan(var_masked[head+model_out_name]))
                #print('model_out_name=',model_out_name,'j=',j,'Xvar_num[i,j]=',Xvar_num[i,j])

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

                    # # Generate confidence intervals for the SAMPLE MEAN with bootstrapping:
                    # var_vals_bot[i,j], var_vals_top[i,j] = bootstrap_ci(var_temp[mask_temp], np.median, n_samples=1000)

        else:
            print('In bin_Xvar, binned by Xvar_name, var_masked = np.nan. Please check why the code goes here')
            print('j=',j, ' Xvar_val=',Xvar_val)

            var_vals[:,j]     = np.nan
            Xvar_num[:,j]      = np.nan
            var_vals_top[:,j] = np.nan
            var_vals_bot[:,j] = np.nan

    return Xvar_series, Xvar_num, var_vals, var_vals_top, var_vals_bot

def write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=None, bounds=None,
                  day_time=False, summer_time=False, IGBP_type=None, time_scale=None,
                  clim_type=None, energy_cor=False,Xvar_num_threshold=None, LAI_range=None,
                  uncertain_type='UCRTN_percentile', models_calc_LAI=None, veg_fraction=None,
                  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                  remove_strange_values=True, country_code=None, Xvar_top_type='to_10',
                  hours_precip_free=None, method='CRV_bins',dist_type='Linear'):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # save data
    if var_name == 'NEE':
        var_name = 'NEP'

    # Get model lists
    if var_name == 'Gs':
        site_names, IGBP_types, clim_types, model_names = load_default_list()
        model_out_list = model_names['model_select_new']
    else:
        model_out_list = get_model_out_list(var_name)

    # Read in the selected raw data
    var_input = pd.read_csv(f'./txt/process3_output/curves/{file_input}',na_values=[''])
    site_num  = len(np.unique(var_input["site_name"]))

    if Xvar_name == 'obs_Tair':
        var_input['obs_Tair'] -= 273.15

    print(Xvar_name+' min is ', np.min(var_input[Xvar_name]), ' max is ', np.max(var_input[Xvar_name]))
    
    # ============ Set the output file name ============
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                uncertain_type=uncertain_type, clarify_site=clarify_site)

    # Checks if a folder exists and creates it if it doesn't
    if not os.path.exists(f'./txt/process4_output/{Xvar_name}/{folder_name}'):
        os.makedirs(f'./txt/process4_output/{Xvar_name}/{folder_name}')

    # ============ Choosing fitting or binning ============
    if method == 'CRV_bins':
        # ============ Bin by VPD ============
        Xvar_series, Xvar_num, var_vals, var_vals_top, var_vals_bot = bin_Xvar(var_input, Xvar_name, model_out_list, uncertain_type)

        # ============ Creat the output dataframe ============
        var = pd.DataFrame(Xvar_series, columns=['Xvar_series'])

        for i, model_out_name in enumerate(model_out_list):

            var[model_out_name+'_Xvar_num'] = Xvar_num[i,:]

            if Xvar_num_threshold == None:
                var[model_out_name+'_vals'] = var_vals[i,:]
                var[model_out_name+'_top']  = var_vals_top[i,:]
                var[model_out_name+'_bot']  = var_vals_bot[i,:]
            else:
                var[model_out_name+'_vals'] = np.where(var[model_out_name+'_Xvar_num'] >= Xvar_num_threshold,
                                                  var_vals[i,:], np.nan)
                var[model_out_name+'_top']  = np.where(var[model_out_name+'_Xvar_num'] >= Xvar_num_threshold,
                                                  var_vals_top[i,:], np.nan)
                var[model_out_name+'_bot']  = np.where(var[model_out_name+'_Xvar_num'] >= Xvar_num_threshold,
                                                  var_vals_bot[i,:], np.nan)

        var['site_num']    = site_num

        var.to_csv(f'./txt/process4_output/{Xvar_name}/{folder_name}/{var_name}{file_message}.csv')

    elif method == 'CRV_fit_GAM_simple' or method == 'CRV_fit_GAM_complex':

        '''
        fitting GAM curve
        '''

        # ============ Check whether the folder save GAM_fit data exist ============
        if not os.path.exists(f'./txt/process4_output/{Xvar_name}/{folder_name}/GAM_fit'):
            os.makedirs(f'./txt/process4_output/{Xvar_name}/{folder_name}/GAM_fit')

        # ============ Creat the output dataframe ============
        if Xvar_name == 'obs_Tair':
            x_bot      = -50.
            x_interval = 1.
        elif  Xvar_name == 'obs_SWdown':
            x_bot      = -20. 
            x_interval = 10. #0.04
        elif Xvar_name == 'obs_Qair':
            x_bot      = 0
            x_interval = 0.0005 #0.04

        # Use multiprocessing to fit GAM models in parallel
        if Xvar_top_type == 'sample_larger_200':

            # Find x_top
            Xvar_series, Xvar_num, var_vals, var_vals_top, var_vals_bot = bin_Xvar(var_input, Xvar_name, model_out_list, uncertain_type)
            x_top = {}

            for i, model_out_name in enumerate(model_out_list):
                try:
                    tmp                   = np.where(Xvar_num[i,:]>=Xvar_num_threshold, 1, 0)
                    x_top[model_out_name] = Xvar_series[np.argwhere(tmp==1)[-1]]
                except:
                    print('Totally ',np.sum(tmp),'VPD bins have data points >=',Xvar_num_threshold)
                    x_top[model_out_name] = np.nan
            print(x_top)

            with mp.Pool() as pool:
                pool.starmap(fit_GAM_for_model, [(folder_name, file_message, var_name, model_in, x_top[model_in], x_bot, x_interval,
                            var_input[Xvar_name],  var_input[get_header(model_in) + model_in], method, dist_type)
                            for model_in in model_out_list])

        elif Xvar_top_type == 'to_10':
            x_top      = 10.001
            with mp.Pool() as pool:
                pool.starmap(fit_GAM_for_model, [(folder_name, file_message, var_name, model_in, x_top, x_bot, x_interval,
                            var_input[Xvar_name],  var_input[get_header(model_in) + model_in], method, dist_type)
                            for model_in in model_out_list])

    return

def fit_GAM_for_model(folder_name, file_message, var_name, model_in, x_top, x_bot, x_interval,
                      x_values, y_values, method='CRV_fit_GAM_simple', dist_type='Linear'):

    # Exclude VPD ==0
    # x_values = np.where(x_values >0.05, x_values, np.nan)
    # y_values = np.where(x_values >0.05, y_values, np.nan)

    # If there are more than 10 data points to make the curve
    if np.sum(~np.isnan(y_values)) > 10:

        if method == 'CRV_fit_GAM_simple':
            Xvar_pred, y_pred, y_int = fit_GAM_simple(x_top,x_bot,x_interval,x_values,y_values,n_splines=7,spline_order=3)
        elif method == 'CRV_fit_GAM_complex':
            Xvar_pred, y_pred, y_int = fit_GAM_complex(model_in, var_name, folder_name, file_message, \
                                                    x_top,x_bot,x_interval,x_values,y_values,dist_type)
        if ~np.all(np.isnan(Xvar_pred)):
            var_fitted              = pd.DataFrame(Xvar_pred, columns=['Xvar_pred'])
            var_fitted['y_pred']    = y_pred
            var_fitted['y_int_top'] = y_int[:,0]
            var_fitted['y_int_bot'] = y_int[:,1]

            var_fitted.to_csv(f'./txt/process4_output/{Xvar_name}/{folder_name}/GAM_fit/{var_name}{file_message}_{model_in}_{dist_type}.csv')

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

    # ===========================================================================================

    # ========================================= 1D curve ========================================

    # ====================== Custom setting ========================
    var_name       = 'Qle'
    uncertain_type = 'UCRTN_bootstrap'
    selected_by    = 'EF_model'
    # method         = 'CRV_fit_GAM_complex'
    method         = 'CRV_bins'
    dist_type      = None # 'Linear' #None #'Gamma' # None #'Linear' #'Poisson' # 'Gamma'
    Xvar_num_threshold = 200
    Xvar_top_type   = 'sample_larger_200' # 'to_10' #

    Xvar_name      = 'obs_Qair' #'obs_SWdown' # 'obs_Tair'# units: K #'VPD' # 
    
    # 0 < EF < 0.2
    bounds         = [0,0.2] #30
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code,
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site) #
    file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                bounds=bounds, day_time=day_time, clarify_site=clarify_site, Xvar_num_threshold=Xvar_num_threshold,
                standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
                country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    gc.collect()

    # 0.2 < EF < 0.4
    bounds         = [0.2,0.4] #30
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code,
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site) #
    file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, Xvar_num_threshold=Xvar_num_threshold,
                standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
                country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    gc.collect()

    # 0.4 < EF < 0.6
    bounds         = [0.4,0.6] #30
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code,
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site)
    file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, Xvar_num_threshold=Xvar_num_threshold,
                standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
                country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    gc.collect()

    # 0.6 < EF < 0.8
    bounds         = [0.6,0.8] #30
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code,
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site)
    file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, Xvar_num_threshold=Xvar_num_threshold,
                standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
                country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    gc.collect()

    # 0.8 < EF < 1.0
    bounds         = [0.8,1.] #30
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code,
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site) #
    file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                bounds=bounds, day_time=day_time, clarify_site=clarify_site, method=method, Xvar_num_threshold=Xvar_num_threshold,
                standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
                country_code=country_code, energy_cor=energy_cor, dist_type=dist_type)
    gc.collect()

    # LAI classification
    LAI_ranges     = [[0.,1.],
                      [1.,2.],
                      [2.,4.],
                      [4.,10.],] #30

    for LAI_range in LAI_ranges:
        print('Calculate LAI_range',LAI_range)

        # 0<EF<0.2
        bounds         = [0,0.2] #30
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                    standardize=standardize, country_code=country_code,
                                                    selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                    LAI_range=LAI_range, clarify_site=clarify_site)
        file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
        write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                    bounds=bounds, day_time=day_time, clarify_site=clarify_site, Xvar_num_threshold=Xvar_num_threshold,
                    standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                    models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
                    country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
        gc.collect()

        # 0.8<EF<1.
        bounds         = [0.8,1.] #30
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                    standardize=standardize, country_code=country_code,
                                                    selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                    LAI_range=LAI_range, clarify_site=clarify_site)
        file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
        write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                    bounds=bounds, day_time=day_time, clarify_site=clarify_site, Xvar_num_threshold=Xvar_num_threshold,
                    standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                    models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
                    country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
        gc.collect()


    # Different land cover
    LAI_range     = None
    veg_fraction  = None
    IGBP_types    = ['GRA', 'DBF', 'ENF', 'EBF']

    for IGBP_type in IGBP_types:

        # 0 < EF < 0.2
        bounds         = [0,0.2]
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                    standardize=standardize, country_code=country_code, selected_by=selected_by,
                                                    bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
        file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'

        write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                    bounds=bounds, day_time=day_time, clarify_site=clarify_site, Xvar_num_threshold=Xvar_num_threshold,
                    standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                    models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, IGBP_type=IGBP_type,
                    country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
        gc.collect()

        # 0.8 < EF < 1.0
        bounds         = [0.8,1.]
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                    standardize=standardize, country_code=country_code, selected_by=selected_by,
                                                    bounds=bounds, veg_fraction=veg_fraction, IGBP_type=IGBP_type, clarify_site=clarify_site)
        file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'

        write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by,
                bounds=bounds, day_time=day_time, clarify_site=clarify_site, Xvar_num_threshold=Xvar_num_threshold,
                standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, Xvar_top_type=Xvar_top_type,
                models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, IGBP_type=IGBP_type,
                country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
        gc.collect()

    # # # Low vegetation coverage
    # # bounds         = [0,0.2] #30
    # # veg_fraction   = [0,0.3]
    # # LAI_range      = None
    # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # #                                             bounds=bounds, veg_fraction=veg_fraction,clarify_site=clarify_site)
    # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # # write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by, bounds=bounds,
    # #                 day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
    # #                 uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # #                 country_code=country_code,
    # #                 energy_cor=energy_cor, method=method)
    # # gc.collect()

    # # bounds         = [0.8,1.]
    # # veg_fraction   = [0,0.3]
    # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # #                                             bounds=bounds, veg_fraction=veg_fraction, clarify_site=clarify_site)
    # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # # write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by, bounds=bounds,
    # #                 day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
    # #                 uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # #                 country_code=country_code,
    # #                 energy_cor=energy_cor, method=method)
    # # gc.collect()


    # # # High vegetation coverage
    # # bounds         = [0,0.2] #30
    # # veg_fraction   = [0.7,1.]
    # # LAI_range      =None
    # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # #                                             bounds=bounds, veg_fraction=veg_fraction, clarify_site=clarify_site)
    # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # # write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by, bounds=bounds,
    # #                 day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
    # #                 uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # #                 country_code=country_code,
    # #                 energy_cor=energy_cor, method=method)
    # # gc.collect()


    # # bounds         = [0.8,1.]
    # # veg_fraction   = [0.7,1.]
    # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # #                                             bounds=bounds, veg_fraction=veg_fraction, clarify_site=clarify_site)
    # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # # write_var_Xvar_parallel(var_name, site_names, file_input, PLUMBER2_path, Xvar_name, selected_by=selected_by, bounds=bounds,
    # #                 day_time=day_time,clarify_site=clarify_site,standardize=standardize, time_scale=time_scale,
    # #                 uncertain_type=uncertain_type, models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # #                 country_code=country_code,
    # #                 energy_cor=energy_cor, method=method)
    # # gc.collect()


    # ## ========================================= 2D grid ========================================
    # # uncertain_type = 'UCRTN_one_std'# 'UCRTN_bootstrap'
    # #                 # 'UCRTN_percentile'
    # #                 # 'UCRTN_one_std'

    # # selected_by    = None
    # # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    # #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    # #                                             veg_fraction=veg_fraction, method=method,
    # #                                             clarify_site=clarify_site)

    # # file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'

    # # # if the the data point is lower than 10 then the bin's value set as nan
    # # print('file_input',file_input)

    # # Xvar_EF_num_threshold = 0

    # # write_var_Xvar_EF(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by, bounds=bounds,
    # #                   time_scale=time_scale,day_time=day_time, energy_cor=energy_cor,
    # #                   Xvar_EF_num_threshold=Xvar_EF_num_threshold, uncertain_type=uncertain_type,
    # #                    models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
    # #                   clarify_site=clarify_site, standardize=standardize)
