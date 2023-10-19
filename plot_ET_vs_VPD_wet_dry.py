import os
import gc
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

def plot_var_VPD(var_name, bin_by=None, low_bound=None, high_bound=None,
                 day_time=False, summer_time=False, window_size=11, order=2,
                 smooth_type='S-G_filter', method='bin_by_vpd',
                 IGBP_type=None, clim_type=None, message=None):

    # ============== read data ==============
    file_name = ''

    if day_time:
        file_name = file_name + '_daytime'

    if IGBP_type !=None:
        file_name = file_name + '_IGBP='+IGBP_type

    if clim_type !=None:
        file_name = file_name + '_clim='+clim_type


    if len(low_bound) >1 and len(high_bound) >1:
        if low_bound[1] > 1:
            var_dry = pd.read_csv(f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'th_'+method+'_coarse.csv')
            var_wet = pd.read_csv(f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(high_bound[0])+'-'+str(high_bound[1])+'th_'+method+'_coarse.csv')
        else:
            var_dry = pd.read_csv(f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'_'+method+'_coarse.csv')
            var_wet = pd.read_csv(f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(high_bound[0])+'-'+str(high_bound[1])+'_'+method+'_coarse.csv')
    elif len(low_bound) == 1 and len(high_bound) == 1:
        if low_bound > 1:
            var_dry = pd.read_csv(f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(low_bound)+'th_'+method+'_coarse.csv')
            var_wet = pd.read_csv(f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(high_bound)+'th_'+method+'_coarse.csv')
        else:
            var_dry = pd.read_csv(f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(low_bound)+'_'+method+'_coarse.csv')
            var_wet = pd.read_csv(f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(high_bound)+'_'+method+'_coarse.csv')

    print('var_dry',var_dry)
    print('var_wet',var_wet)

    # how to get the model out list from the column names???
    model_out_list = []
    for column_name in var_dry.columns:
        if "_vals" in column_name:
            model_out_list.append(column_name.split("_vals")[0])
    print('Checking model_out_list',model_out_list)

    # remove two simulations

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

    for i, model_out_name in enumerate(model_out_list):
        if model_out_name in ['obs_cor','RF_eb']:
            print("Skip ",model_out_name)
        else:
            line_color = model_colors[model_out_name] #plt.cm.tab20(i / len(model_out_list))

            if smooth_type != 'no_soomth':
                dry_vals = smooth_vpd_series(var_dry[model_out_name+'_vals'], window_size, order, smooth_type)
                wet_vals = smooth_vpd_series(var_wet[model_out_name+'_vals'], window_size, order, smooth_type)
            else:
                dry_vals = var_dry[model_out_name+'_vals']
                wet_vals = var_wet[model_out_name+'_vals']

            dry_above_50 = (var_dry[model_out_name+'_vpd_num']>50)
            wet_above_50 = (var_wet[model_out_name+'_vpd_num']>50)

            var_dry_vpd_series = var_dry['vpd_series'][dry_above_50]
            dry_vals           = dry_vals[dry_above_50]
            dry_vals_bot       = var_dry[model_out_name+'_bot'][dry_above_50]
            dry_vals_top       = var_dry[model_out_name+'_top'][dry_above_50]

            var_wet_vpd_series = var_wet['vpd_series'][wet_above_50]
            wet_vals           = wet_vals[wet_above_50]
            wet_vals_bot       = var_wet[model_out_name+'_bot'][wet_above_50]
            wet_vals_top       = var_wet[model_out_name+'_top'][wet_above_50]

            # start plotting
            ax[0,0].plot(var_dry['vpd_series'], var_dry[model_out_name+'_vpd_num'], lw=2.0, color=line_color, alpha=0.9,label=model_out_name)
            ax[0,1].plot(var_wet['vpd_series'], var_wet[model_out_name+'_vpd_num'], lw=2.0, color=line_color, alpha=0.9,label=model_out_name)
            ax[0,0].axhline(y=50, color='black', linestyle='-.', linewidth=1)
            ax[0,1].axhline(y=50, color='black', linestyle='-.', linewidth=1)


            plot = ax[1,0].plot(var_dry_vpd_series, dry_vals, lw=2.0,
                                color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

            fill = ax[1,0].fill_between(var_dry_vpd_series,
                                        dry_vals_bot,
                                        dry_vals_top,
                                        color=line_color, edgecolor="none",
                                        alpha=0.05) #  .rolling(window=10).mean()

            plot = ax[1,1].plot(var_wet_vpd_series, wet_vals, lw=2.0,
                                color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

            fill = ax[1,1].fill_between(var_wet_vpd_series,
                                        wet_vals_bot,
                                        wet_vals_top,
                                        color=line_color, edgecolor="none",
                                        alpha=0.05) #  .rolling(window=10).mean()

    ax[0,0].legend(fontsize=6, frameon=False, ncol=3)

    if IGBP_type !=None:
        ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)
    if clim_type !=None:
        ax[1,0].text(0.12, 0.92, 'Clim_type='+clim_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[1,0].text(0.12, 0.87, 'site_num='+str(var_dry['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[0,0].set_xlim(0, 7.)
    ax[0,1].set_xlim(0, 7.)

    ax[0,0].set_ylim(0, 5000)
    ax[0,1].set_ylim(0, 5000)

    if var_name == 'TVeg':
        ax[1,0].set_ylim(-0.01, 0.35)
        ax[1,1].set_ylim(-0.01, 0.35)
    if var_name == 'Qle':
        ax[1,0].set_ylim(-10, 400)
        ax[1,1].set_ylim(-10, 400)
    if var_name == 'NEE':
        ax[1,0].set_ylim(-1., 1.)
        ax[1,1].set_ylim(-1., 1.)

    # ax[1].set_xlabel('VPD (kPa)', loc='center',size=14)# rotation=270,
    fig.savefig("./plots/"+var_name+'_VPD_all_sites'+file_name+'_'+message+'_'+smooth_type+'_coarse.png',bbox_inches='tight',dpi=300) # '_30percent'

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # Read site names, IGBP and clim
    site_names, IGBP_types, clim_types = load_default_list()

    var_names      = ['NEE','TVeg','Qle']
    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#

    day_time       = True
    energy_cor     = True
    method         = 'bin_by_vpd' #'GAM'
    # Smoothing setting


    window_size    = 11
    order          = 3
    smooth_type    = 'S-G_filter' #'no_soomth'


    # for IGBP_type in IGBP_types:
    # for clim_type in clim_types:
    message        = '0-0.4'
    low_bound      = [0,0.2]
    high_bound     = [0.2,0.4]
    for var_name in var_names:
        if var_name == 'NEE':
            energy_cor     = False
        plot_var_VPD(var_name, bin_by=bin_by, low_bound=low_bound, high_bound=high_bound,
             day_time=day_time, window_size=window_size, order=order,
             smooth_type=smooth_type,message=message)#, IGBP_type=IGBP_type) #, clim_type=clim_type)

        gc.collect()

    message        = '0.4-0.8'
    low_bound      = [0.4,0.6]
    high_bound     = [0.6,0.8]
    for var_name in var_names:
        if var_name == 'NEE':
            energy_cor     = False
        plot_var_VPD(var_name, bin_by=bin_by, low_bound=low_bound, high_bound=high_bound,
             day_time=day_time, window_size=window_size, order=order,
             smooth_type=smooth_type,message=message)#, IGBP_type=IGBP_type) #, clim_type=clim_type)

        gc.collect()

    message        = 'dry-wet'
    low_bound      = [0,0.2]
    high_bound     = [0.8,1.0]
    for var_name in var_names:
        if var_name == 'NEE':
            energy_cor     = False
        plot_var_VPD(var_name, bin_by=bin_by, low_bound=low_bound, high_bound=high_bound,
             day_time=day_time, window_size=window_size, order=order,
             smooth_type=smooth_type,message=message)#, IGBP_type=IGBP_type) #, clim_type=clim_type)

        gc.collect()
