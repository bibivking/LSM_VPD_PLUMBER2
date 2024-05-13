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
import scipy.stats as stats
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *
# from check_vars import check_variable_exists

def calc_distribution(data_in, axis_info={'min_value':None, 'max_value':None,'num_intervals':None}):

    # Delete nan values
    notNan_mask  = ~ np.isnan(data_in)
    data_in      = data_in[notNan_mask]
    tot_num      = len(data_in)

    # Set up the VPD bins
    var_top      = axis_info['max_value']
    var_bot      = axis_info['min_value']
    var_series   = np.linspace(axis_info['min_value'], axis_info['max_value'], axis_info['num_intervals'])
    var_interval = var_series[2]-var_series[1]
    var_std      = np.nanstd(data_in)
    var_mean     = np.nanmean(data_in)

    # Set up the values need to draw
    var_tot      = len(var_series)
    var_fraction = np.zeros(var_tot)

    # Binned by VPD
    for i, var_val in enumerate(var_series):

        mask_var = (data_in > var_val-var_interval/2) & (data_in < var_val+var_interval/2)

        if np.any(mask_var):
            # calculate mean value
            var_fraction[i]  = np.sum(mask_var)/tot_num

    return var_fraction, var_std, var_mean

def write_distribution(var_name, site_names, PLUMBER2_path, bin_by=None, bounds=None,
                       day_time=False, summer_time=False, IGBP_type=None, clim_type=None, energy_cor=False, VPD_num_threshold=None,
                       models_calc_LAI=None, veg_fraction=None, clarify_site={'opt':False,'remove_site':None}, hours_precip_free=None,
                       standardize=None, remove_strange_values=True, axis_info=None):


    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # ========== read the data ==========
    var_output    = pd.read_csv(f'./txt/all_sites/{var_name}_all_sites_with_LAI.csv',na_values=[''])

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
    if var_name in ['Qle','Qh','NEE','GPP']:
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

    if remove_strange_values:
        for model_out_name in model_out_list:
            print('Checking strange values in', model_out_name)
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            var_output[head+model_out_name] = np.where(np.any([var_output[head+model_out_name]>999.,
                                                       var_output[head+model_out_name]<-999.],axis=0),
                                                       np.nan, var_output[head+model_out_name])
            print('np.any(np.isnan(var_output[head+model_out_name]))',np.any(np.isnan(var_output[head+model_out_name])))

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

    # select data with required vegetation fraction
    if veg_fraction !=None:
        veg_frac_mask= (var_output['NoahMPv401_greenness']>=veg_fraction[0]) & (var_output['NoahMPv401_greenness']<=veg_fraction[1])
        var_output   = var_output[veg_frac_mask]
        site_num     = len(np.unique(var_output["site_name"]))
        print('Point 6, site_num=',site_num)

    # whether only considers observation without precipitation in hours_precip_free hours
    if hours_precip_free!=None:
        rain_mask   = (var_output['hrs_after_precip'] > hours_precip_free)
        var_output  = var_output[rain_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 6, site_num=',site_num)

    print('Finish selecting data')

    # To exclude the sites have rainfall input problems
    if clarify_site['opt']:

        length    = len(var_output)
        site_mask = np.full(length,True)

        for site_remove in clarify_site['remove_site']:
            site_mask = np.where(var_output['site_name'] == site_remove, False, site_mask)
        print('site_mask',site_mask)

        # site_mask = ~(var_output['site_name'] in clarify_site['remove_site'])
        var_output  = var_output[site_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 7, site_num=',site_num)

    print( 'Check point 4, np.any(~np.isnan(var_output["model_CABLE"]))=',
           np.any(~np.isnan(var_output["model_CABLE"])) )

    print('Finish selecting data')

    if var_name == 'NEE':
        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            if (model_out_name == 'GFDL') | (model_out_name == 'NoahMPv401') | (model_out_name =='STEMMUS-SCOPE') | (model_out_name =='ACASA'):
                print('model_out_name=',model_out_name,'in GFDL, NoahMPv401, STEMMUS-SCOPE,ACASA')
                values = var_output[head+model_out_name]
            else:
                values = var_output[head+model_out_name]*(-1)
            var_output[head+model_out_name] = values

    if standardize == 'by_obs_mean':

        print('standardized_by_obs_mean')

        # Get all sites left
        sites_left    = np.unique(var_output["site_name"])

        # Initialize the variable of the mean of the left observation for each left site
        site_obs_mean = {}

        # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
        for site in sites_left:

            # Get the mask of this site
            site_mask_tmp       = (var_output['site_name'] == site)

            # Mask the dataframe to get slide of the dataframe for this site
            var_tmp             = var_output[site_mask_tmp]

            # Calculate site obs mean
            site_obs_mean[site] = np.nanmean(var_tmp['obs'])

            # Standardize the different model's values by the obs mean for this site
            for i, model_out_name in enumerate(model_out_list):
                if 'obs' in model_out_name:
                    head = ''
                else:
                    head = 'model_'
                var_output.loc[site_mask_tmp, head+model_out_name] = var_tmp[head+model_out_name]/site_obs_mean[site]

        print('site_obs_mean',site_obs_mean)

    elif standardize == 'by_LAI':

        print('standardized_by_LAI')

        # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
        for i, model_out_name in enumerate(model_out_list):
            use_model_LAI = False
            for model_calc_LAI in models_calc_LAI:
                if model_out_name == model_calc_LAI:
                    use_model_LAI = True
                    break
            if model_out_name in write_var_VPD:
                if 'obs' in model_out_name:
                    head = ''
                else:
                    head = 'model_'
                if use_model_LAI:
                    var_output[head+model_out_name] = np.where( np.all([ var_output[model_out_name+'_LAI'] != 0,
                                                                         np.isnan(var_output[model_out_name+'_LAI']) == False], axis=0) ,
                                                            var_output[head+model_out_name]/var_output[model_out_name+'_LAI'],
                                                            np.nan )
                else:
                    var_output[head+model_out_name] = np.where( var_output['obs_LAI'] != 0 ,
                                                            var_output[head+model_out_name]/var_output['obs_LAI'],
                                                            np.nan )


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

        var_fraction, var_std, var_mean = calc_distribution(var_output[head+model_out_name], axis_info=axis_info)

        if i == 0:
            var_pdf = pd.DataFrame({model_out_name+'_fraction': var_fraction})
            var_pdf[model_out_name+'_std']  = var_std
            var_pdf[model_out_name+'_mean'] = var_mean
        else:
            var_pdf[model_out_name+'_fraction'] = var_fraction
            var_pdf[model_out_name+'_std']      = var_std
            var_pdf[model_out_name+'_mean']     = var_mean

    print(var_pdf)

    # ============ Set the output file name ============
    message = ''

    if day_time:
        message = message + '_daytime'

    if IGBP_type !=None:
        message = message + '_IGBP='+IGBP_type

    if clim_type !=None:
        message = message + '_clim='+clim_type

    if standardize != None:
        message = message + '_standardized_'+standardize

    if clarify_site['opt']:
        message = message + '_clarify_site'

    # save data
    if var_name == 'NEE':
        var_name = 'NEP'

    folder_name='original'

    if standardize != None:
        folder_name='standardized_'+standardize

    if clarify_site['opt']:
        folder_name = folder_name+'_clarify_site'

    if veg_fraction !=None:
        message = message + '_veg_frac='+str(veg_fraction[0])+'-'+str(veg_fraction[1])

    # save data
    if bounds[1] > 1:
        var_pdf.to_csv(f'./txt/pdf/{folder_name}/pdf_{var_name}_VPD'
                            +message+'_'+bin_by+'_'+str(bounds[0])+'-'+str(bounds[1])+'th_coarse.csv')
    else:
        var_pdf.to_csv(f'./txt/pdf/{folder_name}/pdf_{var_name}_VPD'
                            +message+'_'+bin_by+'_'+str(bounds[0])+'-'+str(bounds[1])+'_coarse.csv')

    return

if __name__ == '__main__':

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    method         = 'bin_by_vpd' #'GAM'

    day_time       = True
    energy_cor     = False

    clarify_site      = {'opt': True,
                         'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                         'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    standardize    = None #'by_obs_mean' #None#'by_LAI'#'by_obs_mean'


    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','Noah-MP']


    # ================== 0-0.4 ==================
    # var_name    = 'Qle'  #'TVeg'
    # bounds      = [0.0,0.2] #30
    # axis_info   = {'min_value':0, 'max_value':100,'num_intervals':100}
    # write_distribution(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
    #               clarify_site=clarify_site, standardize=standardize, axis_info=axis_info,
    #               day_time=day_time, energy_cor=energy_cor) #IGBP_type=IGBP_type)
    # gc.collect()
    #
    # bounds      = [0.8,1.] #30
    # axis_info   = {'min_value':0, 'max_value':500,'num_intervals':100}
    # write_distribution(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
    #               clarify_site=clarify_site, standardize=standardize, axis_info=axis_info,
    #               day_time=day_time, energy_cor=energy_cor) #IGBP_type=IGBP_type)
    # gc.collect()
    #
    # var_name    = 'GPP'  #'TVeg'
    # bounds      = [0.0,0.2] #30
    # axis_info   = {'min_value':0, 'max_value':0.5,'num_intervals':100}
    # write_distribution(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
    #               clarify_site=clarify_site, standardize=standardize, axis_info=axis_info,
    #               day_time=day_time, energy_cor=energy_cor) #IGBP_type=IGBP_type)
    # gc.collect()
    #
    # bounds      = [0.8,1.] #30
    # axis_info   = {'min_value':0, 'max_value':1.5,'num_intervals':100}
    # write_distribution(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
    #               clarify_site=clarify_site, standardize=standardize, axis_info=axis_info,
    #               day_time=day_time, energy_cor=energy_cor) #IGBP_type=IGBP_type)
    # gc.collect()

    var_name    = 'NEE'  #'TVeg'
    bounds      = [0.0,0.2] #30
    axis_info   = {'min_value':-0.5, 'max_value':0.5,'num_intervals':100}
    write_distribution(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
                  clarify_site=clarify_site, standardize=standardize, axis_info=axis_info,
                  day_time=day_time, energy_cor=energy_cor) #IGBP_type=IGBP_type)
    gc.collect()

    bounds      = [0.8,1.] #30
    axis_info   = {'min_value':-0.5, 'max_value':2.2,'num_intervals':100}
    write_distribution(var_name, site_names, PLUMBER2_path, bin_by=bin_by, bounds=bounds,
                  clarify_site=clarify_site, standardize=standardize, axis_info=axis_info,
                  day_time=day_time, energy_cor=energy_cor) #IGBP_type=IGBP_type)
    gc.collect()
