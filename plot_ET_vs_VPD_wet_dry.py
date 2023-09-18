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
import gc

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
            if var_name == 'TVeg':
                var_output[model_in] = f.variables[model_var_name][:]*3600
            else:
                var_output[model_in] = f.variables[model_var_name][:]
            if bin_by=='EF_model':
                model_bin_name             = f"{model_in}_EF"
                var_output[model_in+'_EF'] = f.variables[model_bin_name][:]

    if var_name == 'Qle' or var_name == 'Qh':
        var_output['obs'] = f.variables[f"obs_{var_name}"][:]
        model_out_list.append('obs')

    if bin_by=='EF_obs' or bin_by=='EF_model' :
        var_output['obs_EF'] = f.variables["obs_EF"][:]

    # Read VPD and soil moisture information
    var_output['VPD']      = f.variables['VPD'][:]
    var_output['obs_Tair'] = f.variables['obs_Tair'][:]
    var_output['obs_Qair'] = f.variables['obs_Qair'][:]

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

def bin_VPD(var_plot,model_out_list):

    # Set up the VPD bins
    vpd_top      = 7.04
    vpd_bot      = 0.02
    vpd_interval = 0.04
    vpd_series   = np.arange(vpd_bot,vpd_top,vpd_interval)

    # Set up the values need to draw
    vpd_sum      = len(vpd_series)
    var_vals     = np.zeros((len(model_out_list), vpd_sum))
    var_vals_top = np.zeros((len(model_out_list), vpd_sum))
    var_vals_bot = np.zeros((len(model_out_list), vpd_sum))

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

    return vpd_series, var_vals, var_vals_top, var_vals_bot

def plot_spatial_land_days(var_name, site_names, PLUMBER2_path, bin_by=None, message=None, day_time=False, summer_time=False,
                           IGBP_type=None, clim_type=None):

    # r========== read the data ==========
    var_output    = pd.read_csv(f'./txt/{var_name}_all_sites.csv')

    # get the model namelist
    f             = nc.Dataset("/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/AR-SLu.nc", mode='r')
    model_in_list = f.variables[var_name + '_models']
    ntime         = len(f.variables['CABLE_time'])
    model_out_list= []

    # decide output namelist
    for model_in in model_in_list:
        if len(f.variables[f"{model_in}_time"]) == ntime:
            model_out_list.append(model_in)

    # add obs
    if var_name in ['Qle','Qh']:
        model_out_list.append('obs')

    # total site number
    site_num    = len(np.unique(var_output["site_name"]))

    print('location 1')

    # ========== select periods ==========
    if day_time:
        day_mask    = (var_output['hour'] >= 9) & (var_output['hour'] <= 16)
        var_output  = var_output[day_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    if summer_time:
        summer_mask = (var_output['month'] > 11) | (var_output['month']< 3)
        var_output  = var_output[summer_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    if IGBP_type!=None:
        IGBP_mask   = (var_output['IGBP_type'] == IGBP_type)
        var_output  = var_output[IGBP_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    if clim_type!=None:
        clim_mask   = (var_output['climate_type'] == clim_type)
        var_output  = var_output[clim_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    print('location 2')


    # ========== Calculate thresholds ==========
    if bin_by == 'EF_obs':

        # exclude the time steps, Qh<0 or Qle+Qh<10.
        EF_notNan_mask = ~ np.isnan(var_output['obs_EF'])
        var_output     = var_output[EF_notNan_mask]
        #
        # # make var_output_dry and var_output_wet as copies of var_output, otherwise
        # # var_output_dry and var_output_wet will be reviewed as var_output
        # var_output_dry = var_output.copy()
        # var_output_wet = var_output.copy()
        #
        # print(var_output_dry)
        # print(var_output_wet)

        # select the data
        for site_name in site_names:
            print(site_name)

            site_mask   = (var_output['site_name'] == site_name)
            print('np.any(site_mask)',np.any(site_mask))
            try:
                bin_dry     = np.percentile(var_output[site_mask]['obs_EF'], 30) # edited !!!
                bin_wet     = np.percentile(var_output[site_mask]['obs_EF'], 70)
            except:
                bin_dry     = np.nan
                bin_wet     = np.nan
            try:
                dry_mask = dry_mask.append(var_output[site_mask]['obs_EF'] < bin_dry)
                wet_mask = wet_mask.append(var_output[site_mask]['obs_EF'] > bin_wet)
            except:
                dry_mask = (var_output[site_mask]['obs_EF'] < bin_dry)
                wet_mask = (var_output[site_mask]['obs_EF'] > bin_wet)


            print(len(dry_mask))
            print(dry_mask)
        var_output_dry = var_output[dry_mask]
        var_output_wet = var_output[wet_mask]

        # free memory
        EF_notNan_mask=None

    print('location 3')


    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=2, ncols=2, figsize=[15,10],sharex=True, sharey=False, squeeze=True) #
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

    # Select

    vpd_series_dry, var_vals_dry, var_vals_top_dry, var_vals_bot_dry = bin_VPD(var_output_dry,model_out_list)
    vpd_series_wet, var_vals_wet, var_vals_top_wet, var_vals_bot_wet = bin_VPD(var_output_wet,model_out_list)

    # Plot the PDF of the normal distribution
    hist = ax[0,0].hist(var_output_dry['VPD'], bins=400, density=False, alpha=0.6, color='g', histtype='stepfilled')
    hist = ax[0,1].hist(var_output_wet['VPD'], bins=400, density=False, alpha=0.6, color='g', histtype='stepfilled')

    for i, model_out_name in enumerate(model_out_list):
        # Calculate uncertainty
        if i == 0:
            df_var_vals_dry     = pd.DataFrame({model_out_name: var_vals_dry[i,:]})
            df_var_vals_top_dry = pd.DataFrame({model_out_name: var_vals_top_dry[i,:]})
            df_var_vals_bot_dry = pd.DataFrame({model_out_name: var_vals_bot_dry[i,:]})
            df_var_vals_wet     = pd.DataFrame({model_out_name: var_vals_wet[i,:]})
            df_var_vals_top_wet = pd.DataFrame({model_out_name: var_vals_top_wet[i,:]})
            df_var_vals_bot_wet = pd.DataFrame({model_out_name: var_vals_bot_wet[i,:]})
        else:
            df_var_vals_dry[model_out_name]    = var_vals_dry[i,:]
            df_var_vals_top_dry[model_out_name]= var_vals_top_dry[i,:]
            df_var_vals_bot_dry[model_out_name]= var_vals_bot_dry[i,:]
            df_var_vals_wet[model_out_name]    = var_vals_wet[i,:]
            df_var_vals_top_wet[model_out_name]= var_vals_top_wet[i,:]
            df_var_vals_bot_wet[model_out_name]= var_vals_bot_wet[i,:]

        line_color = model_colors[model_out_name] #plt.cm.tab20(i / len(model_out_list))

        plot = ax[1,0].plot(vpd_series_dry, df_var_vals_dry[model_out_name].rolling(window=10).mean(), lw=2.0,
                          color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
        fill = ax[1,0].fill_between(vpd_series_dry, df_var_vals_bot_dry[model_out_name].rolling(window=10).mean(),
                               df_var_vals_top_dry[model_out_name].rolling(window=10).mean(),
                               color=line_color, edgecolor="none", alpha=0.05) #  .rolling(window=10).mean()

        plot = ax[1,1].plot(vpd_series_wet, df_var_vals_wet[model_out_name].rolling(window=10).mean(), lw=2.0,
                          color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
        fill = ax[1,1].fill_between(vpd_series_wet, df_var_vals_bot_wet[model_out_name].rolling(window=10).mean(),
                               df_var_vals_top_wet[model_out_name].rolling(window=10).mean(),
                               color=line_color, edgecolor="none", alpha=0.05) #  .rolling(window=10).mean()

    # if len(bin_vals)==1:
    # else:
    #     ax[1].text(0.1, 0.9, 'bin_by='+str(bin_vals[0])+'~'+str(bin_vals[1]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1].transAxes, fontsize=12)
    #     ax[1].text(0.1, 0.8, 'site_num='+str(site_num), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1].transAxes, fontsize=12)

    ax[1,0].legend(fontsize=8,frameon=False)
    if IGBP_type !=None:
        ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)
    if clim_type !=None:
        ax[1,0].text(0.12, 0.92, 'Clim_type='+clim_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[1,0].text(0.12, 0.87, 'site_num='+str(site_num), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[0,0].set_xlim(0, 7.)
    ax[0,1].set_xlim(0, 7.)

    ax[0,0].set_ylim(0, 500)
    ax[0,1].set_ylim(0, 500)

    #ax[1,0].set_ylim(-0.1, 0.5)
    #ax[1,1].set_ylim(-0.1, 0.5)

    ax[1,0].set_ylim(-50, 400)
    ax[1,1].set_ylim(-50, 400)

    # ax[1].set_xlabel('VPD (kPa)', loc='center',size=14)# rotation=270,
    fig.savefig("./plots/30percent/"+var_name+'_VPD_all_sites'+message,bbox_inches='tight',dpi=300) # '_30percent'

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    site_names     = ["AU-How"]
    #  
    print(site_names)

    var_name       = 'Qle'  #'TVeg' 
    bin_by         = 'EF_obs' #'EF_model' #'EF_obs'#
    bin_by_percent = True
    IGBP_types     = ['CRO']#, 'CSH', 'DBF', 'EBF','ENF', 'GRA', 'MF', 'OSH', 'WET', 'WSA' ,  'SAV']
    clim_types     = ['Af', 'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Csa', 'Csb', 'Cwa',
                      'Dfa', 'Dfb', 'Dfc', 'Dsb', 'Dsc', 'Dwa', 'Dwb', 'ET']

    day_time       = True
    #
    for IGBP_type in IGBP_types:
    # for clim_type in clim_types:
        if day_time:
            message    = '_daytime'

        if IGBP_type !=None:
            message    = message + '_' + IGBP_type

        # if clim_type !=None:
        #     message    = message + '_' + clim_type

        print('message=',message)

        plot_spatial_land_days(var_name, site_names, PLUMBER2_path,
                                bin_by, message=message, day_time=day_time,
                                IGBP_type=IGBP_type,)#clim_type=clim_type) #
        gc.collect()
