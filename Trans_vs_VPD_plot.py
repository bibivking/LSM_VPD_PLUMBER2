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

def read_data(var_name, site_name, input_file, bin_by=None):

    f = nc.Dataset(input_file, mode='r')

    model_in_list = f.variables[var_name + '_models']
    time   = nc.num2date(f.variables['CABLE_time'][:],f.variables['CABLE_time'].units, 
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

    if bin_by=='EF_obs':
        var_output['obs_EF'] = f.variables["obs_EF"][:] 
        
    # Read VPD and soil moisture information
    var_output['VPD']    = f.variables['VPD'][:]

    print('model_out_list', model_out_list)
    print('var_output',     var_output)

    f.close()

    ntime      = len(var_output)
    month      = np.zeros(ntime)
    hour       = np.zeros(ntime)
    site       = np.full([ntime], site_name, dtype=str)

    for i in np.arange(ntime):
        month[i] = var_output['time'][i].month
        hour[i]  = var_output['time'][i].hour

    var_output['month']     = month
    var_output['hour']      = hour
    var_output['site_name'] = site
    
    # return the var values and the model list has the required output
    return var_output, model_out_list


def plot_spatial_land_days(var_name, site_name, input_file, bin_by=None, day_time=False, summer_time=False):

    # ============= read data ================
    var_output, model_out_list = read_data(var_name, site_name, input_file, bin_by)


    if 0:
        # check the diurnal cycle
        var_diurnal_cycle = var_output.groupby(['hour']).mean()

        fig1, ax1 = plt.subplots(figsize=[10, 7])
            
        # set the colors for different models
        model_colors = set_model_colors()

        for i, model_out_name in enumerate(model_out_list):
            line_color = model_colors[model_out_name]#plt.cm.tab20(i / len(model_out_list))
            sct = ax1.plot(var_diurnal_cycle[model_out_name], lw=2.0,  
                            color=line_color, alpha=0.9, label=model_out_name) 

        if var_name == 'trans':
            ax1.set_ylabel('Transpiration (mm h$\mathregular{^{-1}}$)', loc='center',size=14)# rotation=270,    
        if var_name == 'latent':
            ax1.set_ylabel('Latent heat (W m$\mathregular{^{-2}}$)', loc='center',size=14)# rotation=270,

        ax1.legend(fontsize=8,frameon=False)
        fig1.savefig("./plots/diurnal_cycle_"+var_name+"_"+site_name,bbox_inches='tight',dpi=300)

    # Select periods
    if day_time:
        day_mask    = (var_output['hour'] >= 9) & (var_output['hour'] <= 15)
        var_output     = var_output[day_mask]

    if summer_time:
        summer_mask = (var_output['month'] > 11) | (var_output['month']< 3)
        var_output  = var_output[summer_mask]

    if bin_by == 'EF_obs':
        EF_notNan_mask = ~ np.isnan(var_output['obs_EF'])
        print(EF_notNan_mask)
        var_output  = var_output[EF_notNan_mask]

        EF75        = np.percentile(var_output['obs_EF'], 75) 
        EF25        = np.percentile(var_output['obs_EF'], 25)

        EF_25_mask  = (var_output['obs_EF'] <= EF25) 
        var_EF_25   = var_output[EF_25_mask]

        EF_75_mask  = (var_output['obs_EF'] >= EF75) 
        var_EF_75   = var_output[EF_75_mask]

        EF_mid_mask = (var_output['obs_EF'] < EF75) & (var_output['obs_EF'] > EF25) 
        var_EF_mid  = var_output[EF_mid_mask]

        var_output  = var_EF_25

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=2, ncols=1, figsize=[8,10],sharex=True, sharey=False, squeeze=True) #
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
    
    # set the colors for different models
    model_colors = set_model_colors()

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # set variable needed to plot
    var_plot    = var_output #var_sm_mid #var_summur
    print(var_plot)
    vpd_series  = np.arange(0.02,5.04,0.04)
    vpd_sum     = len(vpd_series)
    var_vals    = np.zeros((len(model_out_list), vpd_sum))
    var_vals_75 = np.zeros((len(model_out_list), vpd_sum))
    var_vals_25 = np.zeros((len(model_out_list), vpd_sum))
    print('var_plot["VPD"]',var_plot['VPD'])

    # bin VPD
    for j, vpd_val in enumerate(vpd_series):
        print('vpd_val',vpd_val)
        mask_vpd       = (var_plot['VPD'] > vpd_val-0.02) & (var_plot['VPD'] < vpd_val+0.02)
        try:
            var_masked     = var_plot[mask_vpd]
        except:
            var_masked     = np.nan

        for i, model_out_name in enumerate(model_out_list):
            if j == 0:
                print('i=',i,'model_out_name=',model_out_name)

            var_vals[i,j]  = var_masked[model_out_name].mean()
            if len(var_masked[model_out_name]) > 0:
                var_vals_75[i,j] = np.percentile(var_masked[model_out_name], 75) 
                var_vals_25[i,j] = np.percentile(var_masked[model_out_name], 25) 
            else:
                var_vals_75[i,j] = np.nan
                var_vals_25[i,j] = np.nan

    # Plot the PDF of the normal distribution
    print(var_plot['VPD'])
    hist = ax[0].hist(var_plot['VPD'], bins=400, density=True, alpha=0.6, color='g', histtype='stepfilled')
    # ax[0].xlabel('VPD (kPa)', loc='center',size=14)
    # ax[0].ylabel('Probability density')
    
    for i, model_out_name in enumerate(model_out_list):
        # Calculate uncertainty
        if i == 0:
            df_var_vals     = pd.DataFrame({model_out_name: var_vals[i,:]})
            df_var_vals_75  = pd.DataFrame({model_out_name: var_vals_75[i,:]})
            df_var_vals_25  = pd.DataFrame({model_out_name: var_vals_25[i,:]})
        else:
            df_var_vals[model_out_name]    = var_vals[i,:]
            df_var_vals_75[model_out_name] = var_vals_75[i,:]
            df_var_vals_25[model_out_name] = var_vals_25[i,:]

        print('i=',i,'model_out_name=',model_out_name,np.std(df_var_vals[model_out_name]),np.mean(df_var_vals[model_out_name]))
        line_color = model_colors[model_out_name] #plt.cm.tab20(i / len(model_out_list))

        sct = ax[1].plot(vpd_series, df_var_vals[model_out_name].rolling(window=10).mean(), lw=2.0,  
                     color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
        fill = ax[1].fill_between(vpd_series, df_var_vals_25[model_out_name].rolling(window=10).mean(), 
                               df_var_vals_75[model_out_name].rolling(window=10).mean(),
                               color=line_color, edgecolor="none", alpha=0.05) #  .rolling(window=10).mean()
        
        if 0:
            edge_colors = cmap(np.linspace(0, 1, len(var_plot[model_out_name])))
            sct = ax[1].scatter(var_plot['VPD'], var_plot[model_out_name],  color='none', edgecolors=edge_colors,  s=9,
                            alpha=0.05, cmap=cmap, label=model_out_name) #edgecolor='none', c='red'
        
    ax[1].legend(fontsize=8,frameon=False)
    # if var_name == 'trans':
    #     ax[1].set_ylabel('Transpiration (mm h$\mathregular{^{-1}}$)', loc='center',size=14)# rotation=270,    
    #     ax[1].set_ylim(0, 0.4)
    # if var_name == 'latent':
    #     ax[1].set_ylabel('Latent heat (W m$\mathregular{^{-2}}$)', loc='center',size=14)# rotation=270,

    # ax[1].set_xlabel('VPD (kPa)', loc='center',size=14)# rotation=270,
    fig.savefig("./plots/"+var_name+"_VPD_"+site_name+'_EF_low',bbox_inches='tight',dpi=300)

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"

    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    # site_names     = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    site_names     = ['AU-How']
    print(site_names)

    var_name       = 'latent'#'trans'#
    bin_by         = 'EF_obs'

    for site_name in site_names:

        input_file = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"+site_name+".nc"
    
        plot_spatial_land_days(var_name, site_name, input_file, bin_by)
