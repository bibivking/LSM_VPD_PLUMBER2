import os
import sys
import glob
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
from plot_script import *

def read_data(var_name, site_name, input_file, bin_by=None):

    f = nc.Dataset(input_file, mode='r')

    model_in_list = f.variables[var_name + '_models']
    time          = nc.num2date(f.variables['CABLE_time'][:],f.variables['CABLE_time'].units, 
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True)
    ntime      = len(time)
    var_output = pd.DataFrame(time, columns=['time'])

    model_out_list = []

    if bin_by=='EF_model':
        model_in_list  = f.variables['EF_models']

    for model_in in model_in_list:
        # Set the var and time names of the model 
        model_var_name  = f"{model_in}_{var_name}"
        model_time_name = f"{model_in}_time"
        model_ntime     = len(f.variables[model_time_name])

        # If the model has full time series
        if model_ntime == ntime:
            model_out_list.append(model_in)
            if var_name == 'trans':
                var_output[model_in] = f.variables[model_var_name][:]*3600
            else:
                var_output[model_in] = f.variables[model_var_name][:]
            if bin_by=='EF_model':        
                model_bin_name             = f"{model_in}_EF"
                var_output[model_in+'_EF'] = f.variables[model_bin_name][:]

    if var_name == 'latent' or var_name == 'sensible':
        var_output['obs'] = f.variables[f"obs_{var_name}"][:] 
        model_out_list.append('obs')

    if bin_by=='EF_obs' or bin_by=='EF_model' :
        var_output['obs_EF'] = f.variables["obs_EF"][:] 
        
    # Read VPD and soil moisture information
    var_output['VPD']    = f.variables['VPD'][:]

    # close the file
    f.close()

    ntime      = len(var_output)
    month      = np.zeros(ntime)
    hour       = np.zeros(ntime)
    # site       = np.full([ntime], site_name.rjust(6), dtype=str)

    for i in np.arange(ntime):
        month[i] = var_output['time'][i].month
        hour[i]  = var_output['time'][i].hour

    var_output['month']     = month
    var_output['hour']      = hour
    var_output['site_name'] = site_name
    
    # return the var values and the model list has the required output
    return var_output, model_out_list

def plot_spatial_land_days(var_name, site_names, PLUMBER2_met_path, read_write='read', bin_by=None, bin_vals=None, message=None, day_time=False, summer_time=False):

    if read_write == 'write':
        # ============= read all sites data ================
        # get veg type info
        IGBP_dict = read_IGBP_veg_type(site_names, PLUMBER2_met_path)
        
        # read data
        for i, site_name in enumerate(site_names):
            # print('site_name',site_name)
            input_file = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"+site_name+".nc"
            var_output_tmp, model_out_list = read_data(var_name, site_name, input_file, bin_by)

            # Add veg type
            var_output_tmp['IGBP_type'] = IGBP_dict[site_name]

            if i == 0:
                var_output         = var_output_tmp
                pre_model_out_list = model_out_list
            else:
                if pre_model_out_list != model_out_list: 
                    # Find the elements that are only in pre_model_out_list
                    only_in_pre_model_out_list = np.setdiff1d(pre_model_out_list, model_out_list, assume_unique=True)
                    print('Elements only in pre_model_out_list:',only_in_pre_model_out_list)
                    # Set the missing model simulation as np.nan
                    for missed_model in only_in_pre_model_out_list:
                        var_output_tmp[missed_model] = np.nan
                        if bin_by=='EF_model':        
                            var_output_tmp[missed_model+'_EF'] = np.nan

                # connect different sites data together
                var_output = var_output.append(var_output_tmp, ignore_index=True)

            print('var_output',var_output)
        model_out_list = pre_model_out_list

        # save the dataframe
        var_output.to_csv(f'./txt/{var_name}_all_sites.csv')

    elif read_write == 'read':
        # read the data
        var_output = pd.read_csv(f'./txt/{var_name}_all_sites.csv')

        # get the model namelist
        f             = nc.Dataset("/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/AR-SLu.nc", mode='r')
        model_in_list = f.variables[var_name + '_models']
        ntime         = len(f.variables['CABLE_time'])
        model_out_list= []

        for model_in in model_in_list:
            if len(f.variables[f"{model_in}_time"]) == ntime:
                model_out_list.append(model_in)
        if var_name in ['latent','sensible']:
            model_out_list.append('obs')

    # Select periods
    if day_time:
        day_mask    = (var_output['hour'] >= 9) & (var_output['hour'] <= 16)
        var_output     = var_output[day_mask]

    if summer_time:
        summer_mask = (var_output['month'] > 11) | (var_output['month']< 3)
        var_output  = var_output[summer_mask]

    # if using observed EF
    if bin_by == 'EF_obs':
        # exclude the time steps, Qh<0 or Qle+Qh<10.
        EF_notNan_mask = ~ np.isnan(var_output['obs_EF'])
        var_output     = var_output[EF_notNan_mask]

        # select the data
        if len(bin_vals) == 1:
            EF_mask  = (var_output['obs_EF'] <= bin_vals[0]) 
        elif len(bin_vals) == 2:
            EF_mask  = (var_output['obs_EF'] >= bin_vals[0]) & (var_output['obs_EF'] <= bin_vals[1]) 

        # mask out the time steps that obs_EF is np.nan
        var_output  = var_output[EF_mask]

    # if using model simulated EF
    if bin_by == 'EF_model': 
        # go throught all models which have the var_name
        for model_out_name in model_out_list:
            # bin the variable by the model's own EF
            if len(bin_vals) == 1:
                var_output[model_out_name] = np.where(
                    (var_output[model_out_name+'_EF'] <= bin_vals[0]),
                    var_output[model_out_name], np.nan )
            elif len(bin_vals) == 2:
                var_output[model_out_name] = np.where(
                    (var_output[model_out_name+'_EF'] >= bin_vals[0]) 
                  & (var_output[model_out_name+'_EF'] <= bin_vals[1]),
                    var_output[model_out_name], np.nan )

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=2, ncols=1, figsize=[10,15],sharex=True, sharey=False, squeeze=True) #
    # fig, ax = plt.subplots(figsize=[10, 7])
    # plt.subplots_adjust(wspace=0.0, hspace=0.0)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color']     = almost_black
    plt.rcParams['xtick.color']     = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']      = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black
    
    # Set the colors for different models
    model_colors = set_model_colors()

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # Set variable needed to plot
    var_plot    = var_output #var_sm_mid #var_summur

    # Set up the VPD bins
    vpd_top     = 6.54
    vpd_bot     = 0.02
    vpd_interval= 0.04
    vpd_series  = np.arange(vpd_bot,vpd_top,vpd_interval)

    # Set up the values need to draw
    vpd_sum     = len(vpd_series)
    var_vals    = np.zeros((len(model_out_list), vpd_sum))
    var_vals_top= np.zeros((len(model_out_list), vpd_sum))
    var_vals_bot= np.zeros((len(model_out_list), vpd_sum))

    # Binned by VPD
    for j, vpd_val in enumerate(vpd_series):
        mask_vpd       = (var_plot['VPD'] > vpd_val-0.02) & (var_plot['VPD'] < vpd_val+0.02)
        try:
            var_masked = var_plot[mask_vpd]
        except:
            var_masked = np.nan

        # Draw the line for different models
        for i, model_out_name in enumerate(model_out_list):

            # calculate mean value
            var_vals[i,j] = var_masked[model_out_name].mean(skipna=True)

            if 0:
                # using 1 std as the uncertainty
                var_std   = var_masked[model_out_name].std(skipna=True)
                print(i,j,var_vals[i,j],var_std)
                var_vals_top[i,j] = var_vals[i,j] + var_std
                var_vals_bot[i,j] = var_vals[i,j] - var_std

            if 1:
                # using percentile as the uncertainty
                var_temp  = var_masked[model_out_name]
                mask_temp = ~ np.isnan(var_temp)
                if np.any(mask_temp):
                    var_vals_top[i,j] = np.percentile(var_temp[mask_temp], 75)
                    var_vals_bot[i,j] = np.percentile(var_temp[mask_temp], 25)
                else:
                    var_vals_top[i,j] = np.nan
                    var_vals_bot[i,j] = np.nan

    # Plot the PDF of the normal distribution
    hist = ax[0].hist(var_plot['VPD'], bins=400, density=False, alpha=0.6, color='g', histtype='stepfilled')
    # ax[0].xlabel('VPD (kPa)', loc='center',size=14)
    # ax[0].ylabel('Probability density')
    
    for i, model_out_name in enumerate(model_out_list):
        # Calculate uncertainty
        if i == 0:
            df_var_vals     = pd.DataFrame({model_out_name: var_vals[i,:]})
            df_var_vals_top = pd.DataFrame({model_out_name: var_vals_top[i,:]})
            df_var_vals_bot = pd.DataFrame({model_out_name: var_vals_bot[i,:]})
        else:
            df_var_vals[model_out_name]    = var_vals[i,:]
            df_var_vals_top[model_out_name]= var_vals_top[i,:]
            df_var_vals_bot[model_out_name]= var_vals_bot[i,:]

        line_color = model_colors[model_out_name] #plt.cm.tab20(i / len(model_out_list))

        plot = ax[1].plot(vpd_series, df_var_vals[model_out_name], lw=2.0,  
                          color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
        fill = ax[1].fill_between(vpd_series, df_var_vals_bot[model_out_name],
                               df_var_vals_top[model_out_name],
                               color=line_color, edgecolor="none", alpha=0.05) #  .rolling(window=10).mean()
        
        if 0:
            edge_colors = cmap(np.linspace(0, 1, len(var_plot[model_out_name])))
            sct         = ax[1].scatter(var_plot['VPD'], var_plot[model_out_name],  color='none', edgecolors=edge_colors,  s=9,
                            alpha=0.05, cmap=cmap, label=model_out_name) #edgecolor='none', c='red'
        
    ax[1].legend(fontsize=8,frameon=False)
    # if var_name == 'trans':
    #     ax[1].set_ylabel('Transpiration (mm h$\mathregular{^{-1}}$)', loc='center',size=14)# rotation=270,    
    #     ax[1].set_ylim(0, 0.4)
    # if var_name == 'latent':
    #     ax[1].set_ylabel('Latent heat (W m$\mathregular{^{-2}}$)', loc='center',size=14)# rotation=270,

    # ax[1].set_xlabel('VPD (kPa)', loc='center',size=14)# rotation=270,
    fig.savefig("./plots/"+var_name+'_VPD_all_sites'+message,bbox_inches='tight',dpi=300)

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"

    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]

    var_name       = 'trans'#'latent'  #
    bin_by         = 'EF_obs'#'EF_model' #'EF_obs'#
    read_write     = 'write' #'read'

    for i in np.arange(0.0,0.2,0.1):    
        bin_vals   = [0+i,0.1+i]
        if len(bin_vals) == 1:
            message    = f'_bin_by_model_EF_{int(bin_vals[0]*10)}'
        else:
            message    = f'_bin_by_model_EF_{int(bin_vals[0]*10)}_{int(bin_vals[1]*10)}'
        
        day_time       = False
        if day_time:
            message    = message+'_daytime'
        plot_spatial_land_days(var_name, site_names, PLUMBER2_met_path, read_write, bin_by, bin_vals, message, day_time=day_time)

        day_time       = True
        if day_time:
            message    = message+'_daytime'
        plot_spatial_land_days(var_name, site_names, PLUMBER2_met_path, read_write, bin_by, bin_vals, message, day_time=day_time)
