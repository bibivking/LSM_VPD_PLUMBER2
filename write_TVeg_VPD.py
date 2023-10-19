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

def bin_VPD(var_plot, model_out_list):

    # Set up the VPD bins
    vpd_top      = 7.1 #7.04
    vpd_bot      = 0.1#0.02
    vpd_interval = 0.2#0.04
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

                if 0:
                    # using 1 std as the uncertainty
                    var_std   = var_masked[head+model_out_name].std(skipna=True)
                    var_vals_top[i,j] = var_vals[i,j] + var_std
                    var_vals_bot[i,j] = var_vals[i,j] - var_std

                if 1:
                    # using percentile as the uncertainty
                    var_temp  = var_masked[head+model_out_name]
                    mask_temp = ~ np.isnan(var_temp)
                    if np.any(mask_temp):
                        var_vals_top[i,j] = np.percentile(var_temp[mask_temp], 75)
                        var_vals_bot[i,j] = np.percentile(var_temp[mask_temp], 25)
                    else:
                        var_vals_top[i,j] = np.nan
                        var_vals_bot[i,j] = np.nan
                # print(model_out_name, 'var_vals[i,:]', var_vals[i,:])
        else:
            print('In bin_VPD, binned by VPD, var_masked = np.nan. Please check why the code goes here')
            print('j=',j, ' vpd_val=',vpd_val)

            var_vals[:,j]     = np.nan
            vpd_num[:,j]      = np.nan
            var_vals_top[:,j] = np.nan
            var_vals_bot[:,j] = np.nan

    return vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot

def write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=None, low_bound=30,
                  high_bound=70, day_time=False, summer_time=False, IGBP_type=None,
                  clim_type=None, energy_cor=False,VPD_num_threshold=None, 
                  hours_precip_free=None, method='GAM'):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # ========== read the data ==========
    var_output    = pd.read_csv(f'./txt/{var_name}_all_sites.csv',na_values=[''])
    print( 'Check point 1, np.any(~np.isnan(var_output["model_CABLE"]))=',
           np.any(~np.isnan(var_output["model_CABLE"])) )

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
    if var_name in ['Qle','Qh']:
        model_out_list.append('obs')
        model_out_list.append('obs_cor')

    if var_name in ['NEE']:
        model_out_list.append('obs')

    # total site number
    site_num    = len(np.unique(var_output["site_name"]))

    print('Finish reading csv file')

    # ========== select data ==========

    # whether only considers the sites with energy budget corrected fluxs
    if var_name in ['Qle','Qh'] and energy_cor:
        check_obs_cor = var_output['obs_cor']
        check_obs_cor.to_csv(f'./txt/check_obs_cor.csv')
        print( 'Check point 2, np.any(~np.isnan(var_output["model_CABLE"]))=',
                np.any(~np.isnan(var_output["model_CABLE"])) )
        cor_notNan_mask = ~ np.isnan(var_output['obs_cor'])
        var_output      = var_output[cor_notNan_mask]
        print( 'Check point 3, np.any(~np.isnan(var_output["model_CABLE"]))=',
                np.any(~np.isnan(var_output["model_CABLE"])) )
        # print('var_output["obs_EF"][:100] point 2', var_output["obs_EF"][:100])

    # whether only considers day time
    if day_time:
        day_mask    = (var_output['hour'] >= 9) & (var_output['hour'] <= 16)
        # print('np.any(day_mask)', np.any(day_mask))
        var_output  = var_output[day_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    # whether only considers summers
    if summer_time:
        summer_mask = (var_output['month'] > 11) | (var_output['month']< 3)
        # print('np.any(summer_mask)', np.any(summer_mask))
        var_output  = var_output[summer_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    # whether only considers one type of IGBP
    if IGBP_type!=None:
        IGBP_mask   = (var_output['IGBP_type'] == IGBP_type)
        # print('np.any(IGBP_mask)', np.any(IGBP_mask))
        var_output  = var_output[IGBP_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    # whether only considers one type of climate type
    if clim_type!=None:
        clim_mask   = (var_output['climate_type'] == clim_type)
        # print('np.any(clim_mask)', np.any(clim_mask))
        var_output  = var_output[clim_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    # whether only considers observation without precipitation in hours_precip_free hours
    if hours_precip_free!=None:
        rain_mask   = (var_output['hrs_after_precip'] > hours_precip_free)
        var_output  = var_output[rain_mask]
        site_num    = len(np.unique(var_output["site_name"]))


    print( 'Check point 4, np.any(~np.isnan(var_output["model_CABLE"]))=',
           np.any(~np.isnan(var_output["model_CABLE"])) )

    print('Finish selecting data')

    # ========== Divide dry and wet periods ==========

    # Calculate EF thresholds
    if bin_by == 'EF_obs':

        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        EF_notNan_mask = ~ np.isnan(var_output['obs_EF'])
        var_output     = var_output[EF_notNan_mask]

        # print('np.any(EF_notNan_mask)', np.any(EF_notNan_mask))

        # Select EF<low_bound and EF>high_bound for each site to make sure
        # that every site can contribute to the final VPD lines
        for site_name in site_names:

            # select data for this site
            site_mask       = (var_output['site_name'] == site_name)

            print('In bin by EF, site_name=', site_name, 'np.any(site_mask)',np.any(site_mask))

            # calculate EF thresholds for this site
            if len(low_bound)>1 and len(high_bound)>1:
                try:
                    bin_dry_low  = np.percentile(var_output[site_mask]['obs_EF'], low_bound[0])
                    bin_dry_high = np.percentile(var_output[site_mask]['obs_EF'], low_bound[1])
                    bin_wet_low  = np.percentile(var_output[site_mask]['obs_EF'], high_bound[0])
                    bin_wet_high = np.percentile(var_output[site_mask]['obs_EF'], high_bound[1])
                except:
                    bin_dry_low  = np.nan
                    bin_dry_high = np.nan
                    bin_wet_low  = np.nan
                    bin_wet_high = np.nan
                # make the mask based on EF thresholds and append it to a full-site long logic array
                try:
                    dry_mask = dry_mask.append((var_output[site_mask]['obs_EF'] > bin_dry_low)
                                             & (var_output[site_mask]['obs_EF'] < bin_dry_high))
                    wet_mask = wet_mask.append((var_output[site_mask]['obs_EF'] > bin_wet_low)
                                             & (var_output[site_mask]['obs_EF'] < bin_wet_high))
                except:
                    dry_mask = (var_output[site_mask]['obs_EF'] > bin_dry_low) & (var_output[site_mask]['obs_EF'] < bin_dry_high)
                    wet_mask = (var_output[site_mask]['obs_EF'] > bin_wet_low) & (var_output[site_mask]['obs_EF'] < bin_wet_high)
            elif len(low_bound)==1 and len(high_bound)==1:
                try:
                    bin_dry     = np.percentile(var_output[site_mask]['obs_EF'], low_bound)
                    bin_wet     = np.percentile(var_output[site_mask]['obs_EF'], high_bound)
                except:
                    bin_dry     = np.nan
                    bin_wet     = np.nan

                # make the mask based on EF thresholds and append it to a full-site long logic array
                try:
                    dry_mask = dry_mask.append(var_output[site_mask]['obs_EF'] < bin_dry)
                    wet_mask = wet_mask.append(var_output[site_mask]['obs_EF'] > bin_wet)
                except:
                    dry_mask = (var_output[site_mask]['obs_EF'] < bin_dry)
                    wet_mask = (var_output[site_mask]['obs_EF'] > bin_wet)
            else:
                sys.exit('len(low_bound)=',len(low_bound),'len(high_bound)=',len(high_bound))

        # Mask out the time steps beyond the EF thresholds
        var_output_dry = var_output[dry_mask]
        var_output_wet = var_output[wet_mask]

        # free memory
        EF_notNan_mask=None

    elif bin_by == 'EF_model':

        var_output_dry = copy.deepcopy(var_output)
        var_output_wet = copy.deepcopy(var_output)

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

            if len(low_bound)>1 and len(high_bound)>1:
                dry_mask  = (var_output[EF_var_name] > low_bound[0]) & (var_output[EF_var_name] < low_bound[1])
                wet_mask  = (var_output[EF_var_name] > high_bound[0]) & (var_output[EF_var_name] < high_bound[1])
            elif len(low_bound)==1 and len(high_bound)==1:
                dry_mask  = (var_output[EF_var_name] < low_bound)
                wet_mask  = (var_output[EF_var_name] > high_bound)
            else:
                sys.exit('len(low_bound)=',len(low_bound),'len(high_bound)=',len(high_bound))

            var_output_dry[head+model_out_name] = np.where(dry_mask, var_output[head+model_out_name], np.nan)
            var_output_wet[head+model_out_name] = np.where(wet_mask, var_output[head+model_out_name], np.nan)


        print( 'Check point 7, np.any(~np.isnan(var_output_dry["model_CABLE"]))=',
               np.any(~np.isnan(var_output_dry["model_CABLE"])) )

    print('Finish dividing dry and wet periods')

    # ============ Choosing fitting or binning ============

    if method == 'bin_by_vpd':
        # ============ Bin by VPD ============
        # vpd_series[vpd_tot]
        # var_vals[model_tot, vpd_tot]
        # var_vals_top[model_tot, vpd_tot]
        # var_vals_bot[model_tot, vpd_tot]

        vpd_series_dry, vpd_num_dry, var_vals_dry, var_vals_top_dry, var_vals_bot_dry = bin_VPD(var_output_dry, model_out_list)
        vpd_series_wet, vpd_num_wet, var_vals_wet, var_vals_top_wet, var_vals_bot_wet = bin_VPD(var_output_wet, model_out_list)

        # ============ Creat the output dataframe ============
        var_dry = pd.DataFrame(vpd_series_dry, columns=['vpd_series'])
        var_wet = pd.DataFrame(vpd_series_wet, columns=['vpd_series'])

        for i, model_out_name in enumerate(model_out_list):

            var_dry[model_out_name+'_vpd_num'] = vpd_num_dry[i,:]
            var_wet[model_out_name+'_vpd_num'] = vpd_num_wet[i,:]

            if VPD_num_threshold == None:
                var_dry[model_out_name+'_vals'] = var_vals_dry[i,:]
                var_dry[model_out_name+'_top']  = var_vals_top_dry[i,:]
                var_dry[model_out_name+'_bot']  = var_vals_bot_dry[i,:]
                var_wet[model_out_name+'_vals'] = var_vals_wet[i,:]
                var_wet[model_out_name+'_top']  = var_vals_top_wet[i,:]
                var_wet[model_out_name+'_bot']  = var_vals_bot_wet[i,:]
            else:
                var_dry[model_out_name+'_vals'] = np.where(var_dry[model_out_name+'vpd_num'] >= VPD_num_threshold,
                                                  var_vals_dry[i,:], np.nan)
                var_dry[model_out_name+'_top']  = np.where(var_dry[model_out_name+'vpd_num'] >= VPD_num_threshold,
                                                  var_vals_top_dry[i,:], np.nan)
                var_dry[model_out_name+'_bot']  = np.where(var_dry[model_out_name+'vpd_num'] >= VPD_num_threshold,
                                                  var_vals_bot_dry[i,:], np.nan)
                var_wet[model_out_name+'_vals'] = np.where(var_wet[model_out_name+'vpd_num'] >= VPD_num_threshold,
                                                  var_vals_wet[i,:], np.nan)
                var_wet[model_out_name+'_top']  = np.where(var_wet[model_out_name+'vpd_num'] >= VPD_num_threshold,
                                                  var_vals_top_wet[i,:], np.nan)
                var_wet[model_out_name+'_bot']  = np.where(var_wet[model_out_name+'vpd_num'] >= VPD_num_threshold,
                                                  var_vals_bot_wet[i,:], np.nan)

        var_dry['site_num']    = site_num
        var_wet['site_num']    = site_num

    elif method == 'GAM':
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

            dry_x_values = var_output_dry['VPD']
            dry_y_values = var_output_dry[head+model_out_name]
            dry_vpd_pred, dry_y_pred, dry_y_int = fit_GAM(x_top,x_bot,x_interval,dry_x_values,dry_y_values,n_splines=7,spline_order=3)
            gc.collect()

            wet_x_values = var_output_wet['VPD']
            wet_y_values = var_output_wet[head+model_out_name]
            wet_vpd_pred, wet_y_pred, wet_y_int = fit_GAM(x_top,x_bot,x_interval,wet_x_values,wet_y_values,n_splines=7,spline_order=3)
            gc.collect()
            if i == 0:
                var_dry      = pd.DataFrame(dry_vpd_pred, columns=['vpd_series'])
                var_wet      = pd.DataFrame(wet_vpd_pred, columns=['vpd_series'])

            var_dry[model_out_name+'_vals'] = dry_y_pred
            var_dry[model_out_name+'_top']  = dry_y_int[:,0]
            var_dry[model_out_name+'_bot']  = dry_y_int[:,1]
            var_wet[model_out_name+'_vals'] = wet_y_pred
            var_wet[model_out_name+'_top']  = wet_y_int[:,0]
            var_wet[model_out_name+'_bot']  = wet_y_int[:,1]
        var_dry['site_num']    = site_num
        var_wet['site_num']    = site_num

    # ============ Set the output file name ============
    message = ''

    if day_time:
        message = message + '_daytime'

    if IGBP_type !=None:
        message = message + '_IGBP='+IGBP_type

    if clim_type !=None:
        message = message + '_clim='+clim_type

    # save data
    if len(low_bound) >1 and len(high_bound) >1:
        if low_bound[1] > 1:
            var_dry.to_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'th_'+method+'_coarse.csv')
            var_wet.to_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(high_bound[0])+'-'+str(high_bound[1])+'th_'+method+'_coarse.csv')
        else:
            var_dry.to_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'_'+method+'_coarse.csv')
            var_wet.to_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(high_bound[0])+'-'+str(high_bound[1])+'_'+method+'_coarse.csv')
    elif len(low_bound) == 1 and len(high_bound) == 1:
        if low_bound > 1:
            var_dry.to_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound)+'th_'+method+'_coarse.csv')
            var_wet.to_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(high_bound)+'th_'+method+'_coarse.csv')
        else:
            var_dry.to_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound)+'_'+method+'_coarse.csv')
            var_wet.to_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(high_bound)+'_'+method+'_coarse.csv')

    return


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    var_name       = 'TVeg'  #'TVeg'
    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    site_names, IGBP_types, clim_types = load_default_list()

    day_time       = True
    energy_cor     = True
    method         = 'bin_by_vpd' #'GAM'

    if var_name == 'NEE':
        energy_cor     = False

    # # ================== 0-0.4 ==================
    low_bound      = [0,0.2] #30
    high_bound     = [0.2,0.4] #70
    
    # write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,
    #                 energy_cor=energy_cor, method=method)
    # gc.collect()
    
    for IGBP_type in IGBP_types:
    
        write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
                        high_bound=high_bound, day_time=day_time, IGBP_type=IGBP_type,
                        energy_cor=energy_cor, method=method) # clim_type=None,
        gc.collect()
    
    for clim_type in clim_types:
        write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
                        high_bound=high_bound, day_time=day_time, clim_type=clim_type,
                        energy_cor=energy_cor, method=method) # clim_type=None,
        gc.collect()

    # ================== 0.6-1.0 ==================
    low_bound      = [0.6,0.8] #30
    high_bound     = [0.8,1.] #70

    # write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,
    #                 energy_cor=energy_cor, method=method)
    # gc.collect()

    for IGBP_type in IGBP_types:

        write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
                        high_bound=high_bound, day_time=day_time, IGBP_type=IGBP_type,
                        energy_cor=energy_cor, method=method) # clim_type=None,
        gc.collect()

    for clim_type in clim_types:
        write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
                        high_bound=high_bound, day_time=day_time, clim_type=clim_type,
                        energy_cor=energy_cor, method=method) # clim_type=None,
        gc.collect()

    # ================== 0.4-0.6 ==================
    low_bound      = [0.4,0.6] #30
    high_bound     = [0.8,1.] #70

    # write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,
    #                 energy_cor=energy_cor, method=method)
    # gc.collect()

    for IGBP_type in IGBP_types:

        write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
                        high_bound=high_bound, day_time=day_time, IGBP_type=IGBP_type,
                        energy_cor=energy_cor, method=method) # clim_type=None,
        gc.collect()

    for clim_type in clim_types:
        write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
                        high_bound=high_bound, day_time=day_time, clim_type=clim_type,
                        energy_cor=energy_cor, method=method) # clim_type=None,
        gc.collect()