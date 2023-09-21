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

def bin_VPD(var_plot,model_out_list):

    # Set up the VPD bins
    vpd_top      = 7.1
    vpd_bot      = 0.05
    vpd_interval = 0.1
    vpd_series   = np.arange(vpd_bot,vpd_top,vpd_interval)
    
    # Set up the values need to draw
    vpd_sum      = len(vpd_series)
    var_vals     = np.zeros((len(model_out_list), vpd_sum))
    var_vals_top = np.zeros((len(model_out_list), vpd_sum))
    var_vals_bot = np.zeros((len(model_out_list), vpd_sum))

    # Binned by VPD
    for j, vpd_val in enumerate(vpd_series):

        mask_vpd       = (var_plot['VPD'] > vpd_val-vpd_interval/2) & (var_plot['VPD'] < vpd_val+vpd_interval/2)
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

def write_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=None, 
                day_time=False, summer_time=False, IGBP_type=None, 
                clim_type=None, energy_cor=False):
    
    '''
    1. bin the dataframe by percentile of obs_EF 
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # r========== read the data ==========
    var_output    = pd.read_csv(f'./txt/{var_name}_all_sites.csv',na_values=[''])
    print('var_output["obs_EF"][:100] point 1', var_output["obs_EF"][:100])
    
    # get the model namelist
    f             = nc.Dataset(PLUMBER2_path+"/AR-SLu.nc", mode='r')
    model_in_list = f.variables[var_name + '_models']
    ntime         = len(f.variables['CABLE_time'])
    model_out_list= []

    # decide output namelist
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
    print('location 1')

    # ========== select periods ==========
    if day_time:
        day_mask    = (var_output['hour'] >= 9) & (var_output['hour'] <= 16)
        print('np.any(day_mask)', np.any(day_mask))

        var_output  = var_output[day_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    if summer_time:
        summer_mask = (var_output['month'] > 11) | (var_output['month']< 3)
        print('np.any(summer_mask)', np.any(summer_mask))

        var_output  = var_output[summer_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    if IGBP_type!=None:
        IGBP_mask   = (var_output['IGBP_type'] == IGBP_type)
        print('np.any(IGBP_mask)', np.any(IGBP_mask))
        var_output  = var_output[IGBP_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    if clim_type!=None:
        clim_mask   = (var_output['climate_type'] == clim_type)
        print('np.any(clim_mask)', np.any(clim_mask))
        var_output  = var_output[clim_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    print('location 2')

    # whether plot energy budget correctted
    if energy_cor:
        cor_notNan_mask = ~ np.isnan(var_output['obs_cor'])
        print('np.any(cor_notNan_mask)', np.any(cor_notNan_mask))
        var_output      = var_output[cor_notNan_mask]
        print('var_output["obs_EF"][:100] point 2', var_output["obs_EF"][:100])

    # ========== Calculate thresholds ==========
    if bin_by == 'EF_obs':
        # set the percentile of EF 
        low_bound  = 30
        high_bound = 70
        
        print(var_output['obs_EF'])

        # exclude the time steps, Qh<0 or Qle+Qh<10.
        EF_notNan_mask = ~ np.isnan(var_output['obs_EF'])
        print('np.any(EF_notNan_mask)', np.any(EF_notNan_mask))
        var_output     = var_output[EF_notNan_mask]
        print('var_output["obs_EF"][:100] point 3', var_output["obs_EF"][:100])

        # select the data
        for site_name in site_names:
            print(site_name)

            site_mask   = (var_output['site_name'] == site_name)
            print('np.any(site_mask)',np.any(site_mask))
            try:
                bin_dry     = np.percentile(var_output[site_mask]['obs_EF'], low_bound) 
                bin_wet     = np.percentile(var_output[site_mask]['obs_EF'], high_bound)
            except:
                bin_dry     = np.nan
                bin_wet     = np.nan

            print('bin_dry', bin_dry)
            print('bin_wet', bin_wet)

            try:
                dry_mask = dry_mask.append(var_output[site_mask]['obs_EF'] < bin_dry)
                wet_mask = wet_mask.append(var_output[site_mask]['obs_EF'] > bin_wet)
            except:
                dry_mask = (var_output[site_mask]['obs_EF'] < bin_dry)
                wet_mask = (var_output[site_mask]['obs_EF'] > bin_wet)

            print('len(dry_mask)',len(dry_mask))
            print('dry_mask',dry_mask)

        var_output_dry = var_output[dry_mask]
        var_output_wet = var_output[wet_mask]

        # free memory
        EF_notNan_mask=None

    print('location 3')

    # ============ Bin by VPD ============
    vpd_series_dry, var_vals_dry, var_vals_top_dry, var_vals_bot_dry = bin_VPD(var_output_dry, model_out_list)
    vpd_series_wet, var_vals_wet, var_vals_top_wet, var_vals_bot_wet = bin_VPD(var_output_wet, model_out_list)


    # ============ Write out the variable series ============
    vars_dry                 = pd.DataFrame(vpd_series_dry, columns=['vpd_series'])
    vars_dry['var_vals']     = var_vals_dry
    vars_dry['var_vals_top'] = var_vals_top_dry
    vars_dry['var_vals_bot'] = var_vals_bot_dry

    vars_wet                 = pd.DataFrame(vpd_series_wet, columns=['vpd_series'])
    vars_wet['var_vals']     = var_vals_wet
    vars_wet['var_vals_top'] = var_vals_top_wet
    vars_wet['var_vals_bot'] = var_vals_bot_wet

    # save data
    vars_dry.to_csv(f'./txt/{var_name}_VPD_'+str(low_bound)+'th-EF.csv')
    vars_wet.to_csv(f'./txt/{var_name}_VPD_'+str(high_bound)+'th-EF.csv')

    return




def plot_spatial_land_days(var_name, site_names, PLUMBER2_path, bin_by=None, message=None, 
                           day_time=False, summer_time=False, IGBP_type=None, 
                           clim_type=None, energy_cor=False):


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

    # Plot the PDF of the normal distribution
    hist = ax[0,0].hist(var_output_dry['VPD'], bins=400, density=False, alpha=0.6, color='g', histtype='stepfilled')
    hist = ax[0,1].hist(var_output_wet['VPD'], bins=400, density=False, alpha=0.6, color='g', histtype='stepfilled')
    # Get the histogram data
    hist_counts, hist_edges = hist
    print('==== check the hist data ====')
    print('hist_counts',hist_counts)
    print('hist_edges',hist_edges)

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

        plot = ax[1,0].plot(vpd_series_dry, df_var_vals_dry[model_out_name], lw=2.0,
                          color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
        fill = ax[1,0].fill_between(vpd_series_dry, df_var_vals_bot_dry[model_out_name],
                               df_var_vals_top_dry[model_out_name],
                               color=line_color, edgecolor="none", alpha=0.05) #  .rolling(window=10).mean()

        plot = ax[1,1].plot(vpd_series_wet, df_var_vals_wet[model_out_name], lw=2.0,
                          color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
        fill = ax[1,1].fill_between(vpd_series_wet, df_var_vals_bot_wet[model_out_name],
                               df_var_vals_top_wet[model_out_name],
                               color=line_color, edgecolor="none", alpha=0.05) #  .rolling(window=10).mean()

    ax[1,0].legend(fontsize=6, frameon=False, ncol=3)

    if IGBP_type !=None:
        ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)
    if clim_type !=None:
        ax[1,0].text(0.12, 0.92, 'Clim_type='+clim_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[1,0].text(0.12, 0.87, 'site_num='+str(site_num), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[0,0].set_xlim(0, 7.)
    ax[0,1].set_xlim(0, 7.)
    
    ax[0,0].set_ylim(0, 500)
    ax[0,1].set_ylim(0, 500)

    if var_name == 'TVeg':
        ax[1,0].set_ylim(-0.1, 0.5)
        ax[1,1].set_ylim(-0.1, 0.5)
    if var_name == 'Qle':
        ax[1,0].set_ylim(-50, 400)
        ax[1,1].set_ylim(-50, 400)
    if var_name == 'NEE':
        ax[1,0].set_ylim(-1, 1)
        ax[1,1].set_ylim(-1, 1)

    # ax[1].set_xlabel('VPD (kPa)', loc='center',size=14)# rotation=270,
    fig.savefig("./plots/30percent/"+var_name+'_VPD_all_sites'+message,bbox_inches='tight',dpi=300) # '_30percent'

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    # site_names     = ["AU-Tum"]
    
    print(site_names)

    var_name       = 'Qle'  #'TVeg' 
    bin_by         = 'EF_obs' #'EF_model' #'EF_obs'#
    bin_by_percent = True
    IGBP_types     = ['CRO']#, 'CSH', 'DBF', 'EBF','EBF', 'ENF', 'GRA', 'MF', 'OSH', 'WET', 'WSA', 'SAV']
    clim_types     = ['Af', 'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Csa', 'Csb', 'Cwa',
                      'Dfa', 'Dfb', 'Dfc', 'Dsb', 'Dsc', 'Dwa', 'Dwb', 'ET']

    day_time       = True
    energy_cor     = True

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
                                IGBP_type=IGBP_type,energy_cor=energy_cor) #clim_type=clim_type) #
        gc.collect()
