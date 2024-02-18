'''
Including:
    def plot_var_VPD_uncertainty
    def plot_var_VPD_line_box
    def plot_var_VPD_line_box_three_cols
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from matplotlib.patches import Polygon
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
# from calc_turning_points import *
from PLUMBER2_VPD_common_utils import *

def plot_var_VPD_uncertainty(var_name=None, day_time=False, energy_cor=False, time_scale=None, country_code=None,
                 selected_by=None, veg_fraction=None,  standardize=None, uncertain_type='UCRTN_percentile',
                 method='CRV_bins', IGBP_type=None, clim_type=None, clarify_site={'opt':False,'remove_site':None},
                 num_threshold=200):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=1, ncols=2, figsize=[10,4],sharex=False, sharey=False, squeeze=False)
    # fig, ax = plt.subplots(figsize=[10, 7])
    plt.subplots_adjust(wspace=0.15, hspace=0.0)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    order        = ['(a)','(b)','(c)','(d)',
                    '(e)','(f)','(g)','(h)']

    # ============ Set the input file name ============
    bounds = [0.8,1.]
    folder_name, file_message_wet = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)
    bounds = [0,0.2]
    folder_name, file_message_dry = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

    file_names = [  f'./txt/process4_output/{folder_name}/{var_name}{file_message_wet}.csv',
                    f'./txt/process4_output/{folder_name}/{var_name}{file_message_dry}.csv',]

    print('Reading', file_names)

    for i, file_name in enumerate(file_names):

        # set plot row and col
        row = int(i/2)
        col = i%2

        # Read lines data
        var = pd.read_csv(file_name)

        if i == 0:
            # how to get the model out list from the column names
            model_out_list = []
            for column_name in var.columns:
                if "_vals" in column_name:
                    model_out_list.append(column_name.split("_vals")[0])

            # models to plot
            model_order     = []
            model_names_all = model_names['model_select']
            for model_name in model_names_all:
                if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
                    model_order.append(model_name)

            # Add obs
            # model_order.append('obs')
            print('model_order',model_order)

        # Calculate turning points
        if turning_point['calc']:

            nmodel   = len(model_out_list)
            nvpd     = len(var['vpd_series'])
            val_tmp  = np.zeros((nmodel,nvpd))

            for j, model_out_name in enumerate(model_order):
                vals_vpd_num = var[model_out_name+'_vpd_num']
                # find_turning_points_by_piecewise_regression will transfer NEE to NEP so don't need to do it here.
                val_tmp[j,:] = np.where(vals_vpd_num>num_threshold, var[model_out_name+'_vals'], np.nan)

            # Find the turning points
            if turning_point['method']=='kneed' :
                turning_points = find_turning_points_by_kneed(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='cdf' :
                turning_points = find_turning_points_by_cdf(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='piecewise':
                turning_points, slope = find_turning_points_by_piecewise_regression(model_order, var['vpd_series'], val_tmp, var_name)

        for j, model_out_name in enumerate(model_order):

            # set line color
            line_color = model_colors[model_out_name]

            # ===== Drawing the lines =====
            # Unify NEE units : upwards CO2 movement is positive values

            if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
                value = var[model_out_name+'_vals']*(-1)
            else:
                value = var[model_out_name+'_vals']

            # smooth or not
            if smooth_type != 'no_soomth':
                value = smooth_vpd_series(value, window_size, order, smooth_type)

            # only use vpd data points > num_threshold
            above_thres      = (var[model_out_name+'_vpd_num']>num_threshold)
            var_vpd_series = var['vpd_series'][above_thres]
            value          = value[above_thres]

            # Plot if the data point > num_threshold
            if np.sum(var[model_out_name+'_vpd_num']-num_threshold) > 0:
                if model_out_name == 'obs':
                    lw=2.5
                else:
                    lw=1.5
                plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_color,
                                        alpha=1., label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

                # Plot uncertainty
                # add error range of obs (I think top and bot boundary should be set to 1 sigema)
                vals_bot   = var[model_out_name+'_bot'][above_thres]
                vals_top   = var[model_out_name+'_top'][above_thres]

                fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                                        color=line_color, edgecolor="none", alpha=0.3) #  .rolling(window=10).mean()

            # ===== Drawing the turning points =====
            # Calculate turning points
            # if turning_point['calc']:
            #     ax[row,col].scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)

        ax[row,col].text(0.05, 0.92, order[i], va='bottom', ha='center', rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=14)

        if col == 1:
            ax[row,col].legend(fontsize=8, frameon=False, ncol=2)

        if IGBP_type !=None:
            ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type+'site_num='+str(var['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

        # ax[row,col].set_xlim(0, 11)

    ax[0,0].set_title("Wet (EF>0.8)", fontsize=16)
    ax[0,1].set_title("Dry (EF<0.2)", fontsize=16)
    if var_name == 'Qle':
        ax[0,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    elif var_name == 'Gs':
        ax[0,0].set_ylabel("Canopy Stomatal Conductance (mol m$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$)", fontsize=12)

    if var_name == 'NEE':
        ax[1,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'GPP':
        ax[1,0].set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    ax[0,0].set_xlabel("VPD (kPa)", fontsize=12)
    ax[0,1].set_xlabel("VPD (kPa)", fontsize=12)

    # ax[0,0].set_xticks([0,1,2,3,4,5])
    # ax[0,0].set_xticklabels(['0','1','2', '3','4','5'],fontsize=12)
    # ax[0,0].set_xlim(-0.2,5.3)

    # ax[0,1].set_xticks([0,1,2,3,4,5,6,7])
    # ax[0,1].set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
    # ax[0,1].set_xlim(-0.2,7.3)

    # ax[0,0].set_xticks([0,0.5,1,1.5,2,2.5])
    # ax[0,0].set_xticklabels(['0','0.5','1','1.5','2','2.5'],fontsize=12)
    if var_name == 'Qle':
        if time_scale == 'daily':
            ax[0,0].set_xlim(-0.1,2.7)
            ax[0,0].set_ylim(0,170)
            ax[0,1].set_xlim(-0.1,5.1)
            ax[0,1].set_ylim(0,40)
        elif time_scale == 'hourly':
            ax[0,0].set_xlim(-0.1,5.5)
            ax[0,0].set_xticks([0,1,2,3,4,5])
            ax[0,0].set_xticklabels(['0','1','2', '3','4','5'],fontsize=12)
            ax[0,0].set_ylim(0,350)

            ax[0,1].set_xlim(-0.1,7.5)
            ax[0,1].set_xticks([0,1,2,3,4,5,6,7])
            ax[0,1].set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
            ax[0,1].set_ylim(0,100)

    # ax[0,1].set_xticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
    # ax[0,1].set_xticklabels(['0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5'],fontsize=12)

    ax[0,0].tick_params(axis='y', labelsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)

    fig.savefig(f"./plots/Fig_{var_name}_VPD{file_message_dry}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

def plot_var_VPD_uncertainty_three_cols(var_name=None, day_time=False, energy_cor=False, time_scale=None, country_code=None,
                 selected_by=None, veg_fraction=None,  standardize=None, uncertain_type='UCRTN_percentile',
                 method='CRV_bins', IGBP_type=None, clim_type=None, clarify_site={'opt':False,'remove_site':None},
                 num_threshold=200):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=1, ncols=3, figsize=[15,4],sharex=False, sharey=False, squeeze=False)
    # fig, ax = plt.subplots(figsize=[10, 7])
    plt.subplots_adjust(wspace=0.15, hspace=0.0)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    order        = ['(a)','(b)','(c)','(d)',
                    '(e)','(f)','(g)','(h)']
    # ============ Set the input file name ============
    bounds = [0.6,0.8]
    folder_name, file_message1 = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)
    bounds = [0.4,0.6]
    folder_name, file_message2 = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)
    bounds = [0.2,0.4]
    folder_name, file_message3 = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

    file_names = [  f'./txt/process4_output/{folder_name}/{var_name}{file_message1}.csv',
                    f'./txt/process4_output/{folder_name}/{var_name}{file_message2}.csv',
                    f'./txt/process4_output/{folder_name}/{var_name}{file_message3}.csv',]

    print('Reading', file_names)

    for i, file_name in enumerate(file_names):

        # set plot row and col
        row = int(i/3)
        col = i%3

        # Read lines data
        var = pd.read_csv(file_name)
        if i == 0:
            # how to get the model out list from the column names
            model_out_list = []
            for column_name in var.columns:
                if "_vals" in column_name:
                    model_out_list.append(column_name.split("_vals")[0])

            # models to plot
            model_order     = []
            model_names_all = model_names['model_select']
            for model_name in model_names_all:
                if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
                    model_order.append(model_name)

            # # Add obs
            # model_order.append('obs')
            print('model_order',model_order)

        # Calculate turning points
        if turning_point['calc']:

            nmodel   = len(model_out_list)
            nvpd     = len(var['vpd_series'])
            val_tmp  = np.zeros((nmodel,nvpd))

            for j, model_out_name in enumerate(model_order):
                vals_vpd_num = var[model_out_name+'_vpd_num']
                # find_turning_points_by_piecewise_regression will transfer NEE to NEP so don't need to do it here.
                val_tmp[j,:] = np.where(vals_vpd_num>num_threshold, var[model_out_name+'_vals'], np.nan)

            # Find the turning points
            if turning_point['method']=='kneed' :
                turning_points = find_turning_points_by_kneed(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='cdf' :
                turning_points = find_turning_points_by_cdf(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='piecewise':
                turning_points, slope = find_turning_points_by_piecewise_regression(model_order, var['vpd_series'], val_tmp, var_name)

        for j, model_out_name in enumerate(model_order):

            # set line color
            line_color = model_colors[model_out_name]

            # ===== Drawing the lines =====
            # Unify NEE units : upwards CO2 movement is positive values

            if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
                value = var[model_out_name+'_vals']*(-1)
            else:
                value = var[model_out_name+'_vals']

            # smooth or not
            if smooth_type != 'no_soomth':
                value = smooth_vpd_series(value, window_size, order, smooth_type)

            # only use vpd data points > num_threshold
            above_thres      = (var[model_out_name+'_vpd_num']>num_threshold)
            var_vpd_series = var['vpd_series'][above_thres]
            value          = value[above_thres]

            # Plot if the data point > num_threshold
            if np.sum(var[model_out_name+'_vpd_num']-num_threshold) > 0:
                if model_out_name == 'obs':
                    lw=2.5
                else:
                    lw=1.5
                plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_color,
                                        alpha=1., label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

                # Plot uncertainty
                # add error range of obs (I think top and bot boundary should be set to 1 sigema)
                vals_bot   = var[model_out_name+'_bot'][above_thres]
                vals_top   = var[model_out_name+'_top'][above_thres]

                fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                                        color=line_color, edgecolor="none", alpha=0.3) #  .rolling(window=10).mean()

            # ===== Drawing the turning points =====
            # Calculate turning points
            # if turning_point['calc']:
            #     ax[row,col].scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)

        ax[row,col].text(0.05, 0.92, order[i], va='bottom', ha='center', rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=14)

        if col == 2:
            ax[row,col].legend(fontsize=8, frameon=False, ncol=2)

        if IGBP_type !=None:
            ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type+'site_num='+str(var['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

        # ax[row,col].set_xlim(0, 11)

    ax[0,0].set_title("0.6<EF<0.8", fontsize=16)
    ax[0,1].set_title("0.4<EF<0.6", fontsize=16)
    ax[0,2].set_title("0.2<EF<0.4", fontsize=16)

    if var_name == 'Qle':
        ax[0,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    elif var_name == 'Gs':
        ax[0,0].set_ylabel("Canopy Stomatal Conductance (mol m$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$)", fontsize=12)

    if var_name == 'NEE':
        ax[1,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'GPP':
        ax[1,0].set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    ax[0,0].set_xlabel("VPD (kPa)", fontsize=12)
    ax[0,1].set_xlabel("VPD (kPa)", fontsize=12)
    ax[0,2].set_xlabel("VPD (kPa)", fontsize=12)

    # ax[0,0].set_xticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
    # ax[0,0].set_xticklabels(['0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5'],fontsize=12)
    # ax[0,0].set_xlim(-0.1,5.1)

    ax[0,0].set_xticks([0,1,2,3,4,5,6.,7.])
    ax[0,0].set_xticklabels(['0','1','2','3','4','5','6','7'],fontsize=12)
    ax[0,0].set_xlim(-0.1,7.1)

    if var_name == 'Qle':
        if time_scale == 'daily':
            ax[0,0].set_ylim(0,100)
        elif time_scale == 'hourly':
            ax[0,0].set_ylim(0,300)

    # ax[0,1].set_xticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
    # ax[0,1].set_xticklabels(['0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5'],fontsize=12)
    # ax[0,1].set_xlim(-0.1,5.1)
    ax[0,1].set_xticks([0,1,2,3,4,5,6.,7.])
    ax[0,1].set_xticklabels(['0','1','2','3','4','5','6','7'],fontsize=12)
    ax[0,1].set_xlim(-0.1,7.1)

    if var_name == 'Qle':
        if time_scale == 'daily':
            ax[0,1].set_ylim(0,100)
        elif time_scale == 'hourly':
            ax[0,1].set_ylim(0,300)

    # ax[0,2].set_xticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
    # ax[0,2].set_xticklabels(['0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5'],fontsize=12)
    # ax[0,2].set_xlim(-0.1,5.1)

    ax[0,2].set_xticks([0,1,2,3,4,5,6.,7.])
    ax[0,2].set_xticklabels(['0','1','2','3','4','5','6','7'],fontsize=12)
    ax[0,2].set_xlim(-0.1,7.1)

    if var_name == 'Qle':
        if time_scale == 'daily':
            ax[0,2].set_ylim(0,100)
        elif time_scale == 'hourly':
            ax[0,2].set_ylim(0,300)

    ax[0,0].tick_params(axis='y', labelsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)

    fig.savefig(f"./plots/Fig_{var_name}_VPD{file_message1}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

def plot_var_VPD_uncertainty_veg_LAI_eight(var_name=None, day_time=False, energy_cor=False, time_scale=None, country_code=None,
                 selected_by=None, veg_fraction=None,  standardize=None, uncertain_type='UCRTN_percentile',
                 method='CRV_bins', IGBP_types=None, LAI_ranges=None, clim_type=None,
                 clarify_site={'opt':False,'remove_site':None}, num_threshold=200):

    # ============ Setting for plotting ============
    ncol     =4
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=4, ncols=4, figsize=[24,16],sharex=False, sharey=False, squeeze=False)
    # fig, ax = plt.subplots(figsize=[10, 7])

    plt.subplots_adjust(wspace=0.18, hspace=0.25)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    order        = ['(a)','(b)','(c)','(d)',
                    '(e)','(f)','(g)','(h)',
                    '(i)','(j)','(k)','(l)',
                    '(m)','(n)','(o)','(p)']

    # ============ Set the input file name ============
    ## Wet periods
    file_names = []

    # Loop veg types
    bounds = [0.8,1.0]
    for IGBP_type in IGBP_types:
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                    IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                    country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                    uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)
        file_names.append(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')

    bounds = [0,0.2]
    for IGBP_type in IGBP_types:
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                    IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                    country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                    uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)
        file_names.append(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')

    # Loop veg types
    bounds = [0.8,1.0]
    for LAI_range in LAI_ranges:
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                    LAI_range=LAI_range, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                    country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                    uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

        file_names.append(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')

    bounds = [0,0.2]
    for LAI_range in LAI_ranges:
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                    LAI_range=LAI_range, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                    country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                    uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

        file_names.append(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')

    print('Reading', file_names)

    for i, file_name in enumerate(file_names):

        # set plot row and col
        row = int(i/ncol)
        col = i%ncol

        # Read lines data
        var = pd.read_csv(file_name)
        if i == 0:
            # how to get the model out list from the column names
            model_out_list = []
            for column_name in var.columns:
                if "_vals" in column_name:
                    model_out_list.append(column_name.split("_vals")[0])

            # models to plot
            model_order     = []
            model_names_all = model_names['model_select']
            for model_name in model_names_all:
                if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
                    model_order.append(model_name)

            # # Add obs
            # model_order.append('obs')
            print('model_order',model_order)

        # Calculate turning points
        if turning_point['calc']:

            nmodel   = len(model_out_list)
            nvpd     = len(var['vpd_series'])
            val_tmp  = np.zeros((nmodel,nvpd))

            for j, model_out_name in enumerate(model_order):
                vals_vpd_num = var[model_out_name+'_vpd_num']
                # find_turning_points_by_piecewise_regression will transfer NEE to NEP so don't need to do it here.
                val_tmp[j,:] = np.where(vals_vpd_num>num_threshold, var[model_out_name+'_vals'], np.nan)

            # Find the turning points
            if turning_point['method']=='kneed' :
                turning_points = find_turning_points_by_kneed(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='cdf' :
                turning_points = find_turning_points_by_cdf(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='piecewise':
                turning_points, slope = find_turning_points_by_piecewise_regression(model_order, var['vpd_series'], val_tmp, var_name)

        for j, model_out_name in enumerate(model_order):

            # set line color
            line_color = model_colors[model_out_name]

            # ===== Drawing the lines =====
            # Unify NEE units : upwards CO2 movement is positive values

            if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
                value = var[model_out_name+'_vals']*(-1)
            else:
                value = var[model_out_name+'_vals']

            # smooth or not
            if smooth_type != 'no_soomth':
                value = smooth_vpd_series(value, window_size, order, smooth_type)

            # only use vpd data points > num_threshold
            above_thres      = (var[model_out_name+'_vpd_num']>num_threshold)
            var_vpd_series = var['vpd_series'][above_thres]
            value          = value[above_thres]

            # Plot if the data point > num_threshold
            if np.sum(var[model_out_name+'_vpd_num']-num_threshold) > 0:
                if model_out_name == 'obs':
                    lw=2.5
                else:
                    lw=1.5
                plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_color,
                                        alpha=1., label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

                # Plot uncertainty
                # add error range of obs (I think top and bot boundary should be set to 1 sigema)
                vals_bot   = var[model_out_name+'_bot'][above_thres]
                vals_top   = var[model_out_name+'_top'][above_thres]

                fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                                        color=line_color, edgecolor="none", alpha=0.3) #  .rolling(window=10).mean()

            # ===== Drawing the turning points =====
            # Calculate turning points
            # if turning_point['calc']:
            #     ax[row,col].scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)
            ax[row,col].text(0.05, 0.90, order[i], va='bottom', ha='center', rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=14)

        if row==0 and col == 1 and var_name =='Gs':
            ax[row,col].legend(fontsize=8, frameon=False, ncol=2)
        if row==0 and col == 1 and var_name =='Qle':
            ax[row,col].legend(fontsize=7, frameon=False, ncol=2)
        if row==0:
            ax[row,col].set_title(IGBP_types[col], fontsize=14)

        if row==2:
            ax[row,col].set_title(f'{LAI_ranges[col][0]}<LAI<{LAI_ranges[col][1]}', fontsize=14)


        # if IGBP_types !=None:
        #     ax[row,col].set_title(IGBP_types[i], fontsize=20)
        #     ax[row,col].text(0.12, 0.92, str(var['site_num'][0])+' sites', va='bottom', ha='center',
        #                      rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=12)
        #
        # elif LAI_ranges != None:
        #     ax[row,col].set_title(f'LAI {LAI_ranges[i]}', fontsize=20)
        #     ax[row,col].text(0.12, 0.92, str(var['site_num'][0])+' sites', va='bottom', ha='center',
        #                      rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=12)
        if row == 3:
            ax[row,col].set_xlabel("VPD (kPa)", fontsize=12)
        ax[row,col].tick_params(axis='y', labelsize=12)

        ax[row,col].set_xticks([0,1,2,3,4,5,6,7]) #
        ax[row,col].set_xticklabels(['0','1','2','3','4','5','6','7'],fontsize=12) #

        if row == 1 or row == 3:
            ax[row,col].set_xlim(-0.1,7.5)
            if var_name == 'Qle':
                ax[row,col].set_ylim(0,120)
        else:
            ax[row,col].set_xlim(-0.1,7.5)
            if var_name == 'Qle':
                ax[row,col].set_ylim(0,400)

        # ax[row,col].set_xlim(0, 11)

        if var_name == 'Qle':
            ax[row,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
        elif var_name == 'Gs':
            ax[row,0].set_ylabel("Canopy Stomatal Conductance\n(mol m$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$)", fontsize=12)

        elif var_name == 'NEE':
            ax[row,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
        elif var_name == 'GPP':
            ax[row,0].set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    fig.savefig("./plots/Fig_"+var_name+"_VPD_wet_dry_veg_LAI.png",bbox_inches='tight',dpi=300) # '_30percent'

    return


def plot_var_VPD_uncertainty_veg_LAI(var_name=None, day_time=False, energy_cor=False, time_scale=None, country_code=None,
                 selected_by=None, veg_fraction=None,  standardize=None, uncertain_type='UCRTN_percentile',
                 method='CRV_bins', IGBP_types=None, LAI_ranges=None, clim_type=None, bounds=None,
                 clarify_site={'opt':False,'remove_site':None}, num_threshold=200):

    # ============ Setting for plotting ============
    if IGBP_types != None:
        ncol = len(IGBP_types)

    elif LAI_ranges !=None:
        ncol = len(LAI_ranges[:])

    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=1, ncols=ncol, figsize=[5*ncol,5],sharex=False, sharey=False, squeeze=False)
    # fig, ax = plt.subplots(figsize=[10, 7])

    plt.subplots_adjust(wspace=0.12, hspace=0.02)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    order        = ['(a)','(b)','(c)','(d)',
                    '(e)','(f)','(g)','(h)']
    # ============ Set the input file name ============
    ## Wet periods
    file_names = []

    # Loop veg types
    if IGBP_types!=None:
        for IGBP_type in IGBP_types:
            folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                        IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                        country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                        uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

            file_names.append(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')

    # Loop veg types
    elif LAI_ranges!=None:
        for LAI_range in LAI_ranges:
            folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                        LAI_range=LAI_range, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                        country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                        uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

            file_names.append(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')

    print('Reading', file_names)

    for i, file_name in enumerate(file_names):

        # set plot row and col
        row = int(i/ncol)
        col = i%ncol

        # Read lines data
        var = pd.read_csv(file_name)
        if i == 0:
            # how to get the model out list from the column names
            model_out_list = []
            for column_name in var.columns:
                if "_vals" in column_name:
                    model_out_list.append(column_name.split("_vals")[0])

            # models to plot
            model_order     = []
            model_names_all = model_names['model_select']
            for model_name in model_names_all:
                if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
                    model_order.append(model_name)

            # # Add obs
            # model_order.append('obs')
            print('model_order',model_order)

        # Calculate turning points
        if turning_point['calc']:

            nmodel   = len(model_out_list)
            nvpd     = len(var['vpd_series'])
            val_tmp  = np.zeros((nmodel,nvpd))

            for j, model_out_name in enumerate(model_order):
                vals_vpd_num = var[model_out_name+'_vpd_num']
                # find_turning_points_by_piecewise_regression will transfer NEE to NEP so don't need to do it here.
                val_tmp[j,:] = np.where(vals_vpd_num>num_threshold, var[model_out_name+'_vals'], np.nan)

            # Find the turning points
            if turning_point['method']=='kneed' :
                turning_points = find_turning_points_by_kneed(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='cdf' :
                turning_points = find_turning_points_by_cdf(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='piecewise':
                turning_points, slope = find_turning_points_by_piecewise_regression(model_order, var['vpd_series'], val_tmp, var_name)

        for j, model_out_name in enumerate(model_order):

            # set line color
            line_color = model_colors[model_out_name]

            # ===== Drawing the lines =====
            # Unify NEE units : upwards CO2 movement is positive values

            if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
                value = var[model_out_name+'_vals']*(-1)
            else:
                value = var[model_out_name+'_vals']

            # smooth or not
            if smooth_type != 'no_soomth':
                value = smooth_vpd_series(value, window_size, order, smooth_type)

            # only use vpd data points > num_threshold
            above_thres      = (var[model_out_name+'_vpd_num']>num_threshold)
            var_vpd_series = var['vpd_series'][above_thres]
            value          = value[above_thres]

            # Plot if the data point > num_threshold
            if np.sum(var[model_out_name+'_vpd_num']-num_threshold) > 0:
                if model_out_name == 'obs':
                    lw=2
                else:
                    lw=1
                plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_color,
                                        alpha=0.8, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

                # Plot uncertainty
                # add error range of obs (I think top and bot boundary should be set to 1 sigema)
                vals_bot   = var[model_out_name+'_bot'][above_thres]
                vals_top   = var[model_out_name+'_top'][above_thres]

                fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                                        color=line_color, edgecolor="none", alpha=0.2) #  .rolling(window=10).mean()

            # ===== Drawing the turning points =====
            # Calculate turning points
            # if turning_point['calc']:
            #     ax[row,col].scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)

        if col == 1:
            ax[row,col].legend(fontsize=7, frameon=False, ncol=2)

        if IGBP_types !=None:
            ax[row,col].set_title(IGBP_types[i], fontsize=20)
            ax[row,col].text(0.12, 0.92, str(var['site_num'][0])+' sites', va='bottom', ha='center',
                             rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=12)

        elif LAI_ranges != None:
            ax[row,col].set_title(f'LAI {LAI_ranges[i]}', fontsize=20)
            ax[row,col].text(0.12, 0.92, str(var['site_num'][0])+' sites', va='bottom', ha='center',
                             rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=12)

        ax[row,col].set_xlabel("VPD (kPa)", fontsize=12)
        ax[row,col].tick_params(axis='y', labelsize=12)

        if bounds[0] <0.5:
            ax[row,col].set_xticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5]) #
            ax[row,col].set_xticklabels(['0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5','5.5'],fontsize=12) #
            ax[row,col].set_xlim(-0.1,5.6)
            if var_name == 'Qle':
                ax[row,col].set_ylim(0,50)
        else:
            ax[row,col].set_xticks([0,0.5,1,1.5,2,2.5,3,3.5,4])
            ax[row,col].set_xticklabels(['0','0.5','1','1.5','2','2.5','3','3.5','4'],fontsize=12)
            ax[row,col].set_xlim(-0.1,4.1)
            if var_name == 'Qle':
                ax[row,col].set_ylim(0,200)

        # ax[row,col].set_xlim(0, 11)

    if var_name == 'Qle':
        ax[0,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    elif var_name == 'Gs':
        ax[0,0].set_ylabel("Canopy Stomatal Conductance (mol m$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$)", fontsize=12)

    if var_name == 'NEE':
        ax[1,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'GPP':
        ax[1,0].set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    if IGBP_types != None:
        fig.savefig("./plots/Fig_"+var_name+"_VPD_"+str(bounds[0])+"-"+str(bounds[1])+"_veg.png",bbox_inches='tight',dpi=300) # '_30percent'
    elif LAI_ranges != None:
        fig.savefig("./plots/Fig_"+var_name+"_VPD_"+str(bounds[0])+"-"+str(bounds[1])+"_LAI.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

def plot_var_VPD_line_box(bin_by=None, window_size=11, order=3,
                 smooth_type='S-G_filter', method='bin_by_vpd', message=None, model_names=None,
                 day_time=None,  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                 error_type=None, pdf_or_box='pdf',
                 IGBP_type=None, clim_type=None,turning_point={'calc':False,'method':'kneed'}):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=2, ncols=2, figsize=[14,9],sharex=False, sharey=False, squeeze=True) #
    # fig, ax = plt.subplots(figsize=[10, 7])
    plt.subplots_adjust(wspace=0.09, hspace=0.02)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    order        = ['(a)','(b)','(c)','(d)',
                    '(e)','(f)','(g)','(h)']
    # ============== read data ==============
    message   = ''
    subfolder = ''

    if day_time:
        message = message + 'daytime'

    if IGBP_type != None:
        message = message + '_IGBP='+IGBP_type
    if clim_type != None:
        message = message + '_clim='+clim_type

    if standardize != None:
        message = message + '_standardized_'+standardize

    if clarify_site['opt']:
        message = message + '_clarify_site'

    if error_type !=None:
        message = message + '_error_type='+error_type

    folder_name = 'original'

    if standardize != None:
        folder_name = 'standardized_'+standardize

    if clarify_site['opt']:
        folder_name = folder_name+'_clarify_site'

    file_names = [
                    f'./txt/VPD_curve/{folder_name}/{subfolder}Qle_VPD_'+message+'_'+bin_by+'_0.8-1.0_'+method+'_coarse.csv',
                    f'./txt/VPD_curve/{folder_name}/{subfolder}Qle_VPD_'+message+'_'+bin_by+'_0-0.2_'+method+'_coarse.csv',
                    f'./txt/VPD_curve/{folder_name}/{subfolder}NEP_VPD_'+message+'_'+bin_by+'_0.8-1.0_'+method+'_coarse.csv',
                    f'./txt/VPD_curve/{folder_name}/{subfolder}NEP_VPD_'+message+'_'+bin_by+'_0-0.2_'+method+'_coarse.csv',]

    boxplot_file_names = [ '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/pdf/original_clarify_site/pdf_Qle_VPD_daytime_clarify_site_EF_model_0.8-1.0_coarse.csv',
                           '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/pdf/original_clarify_site/pdf_Qle_VPD_daytime_clarify_site_EF_model_0.0-0.2_coarse.csv',
                           '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/pdf/original_clarify_site/pdf_NEP_VPD_daytime_clarify_site_EF_model_0.8-1.0_coarse.csv',
                           '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/pdf/original_clarify_site/pdf_NEP_VPD_daytime_clarify_site_EF_model_0.0-0.2_coarse.csv',]

    print('Reading', file_names)
    print('Reading', boxplot_file_names)

    var_names  = ['Qle','Qle','NEE','NEE']
    # var_names  = ['Qle','Qle','GPP','GPP']

    for i, file_name in enumerate(file_names):

        # set plot row and col
        row = int(i/2)
        col = i%2

        # Read lines data
        var = pd.read_csv(file_name)

        # Read boxplot data
        dist_box_values = pd.read_csv(boxplot_file_names[i])

        # how to get the model out list from the column names???
        model_out_list = []
        for column_name in var.columns:
            if "_vals" in column_name:
                model_out_list.append(column_name.split("_vals")[0])

        # models to plot
        model_order     = []
        model_names_all = model_names['model_select']
        for model_name in model_names_all:
            if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
                model_order.append(model_name)

        # Add obs
        if var_names[i] in ['Qle','NEE','GPP']:
            model_order.append('obs')
        print('model_order',model_order)

        # Calculate turning points
        if turning_point['calc']:

            nmodel   = len(model_out_list)
            nvpd     = len(var['vpd_series'])
            val_tmp  = np.zeros((nmodel,nvpd))

            for j, model_out_name in enumerate(model_order):
                vals_vpd_num = var[model_out_name+'_vpd_num']
                # find_turning_points_by_piecewise_regression will transfer NEE to NEP so don't need to do it here.
                val_tmp[j,:] = np.where(vals_vpd_num>200, var[model_out_name+'_vals'], np.nan)

            # Find the turning points
            if turning_point['method']=='kneed' :
                turning_points = find_turning_points_by_kneed(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='cdf' :
                turning_points = find_turning_points_by_cdf(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='piecewise':
                turning_points, slope = find_turning_points_by_piecewise_regression(model_order, var['vpd_series'], val_tmp, var_names[i])

        for j, model_out_name in enumerate(model_order):

            # set line color
            line_color = model_colors[model_out_name]

            # ===== Drawing the lines =====
            # Unify NEE units : upwards CO2 movement is positive values

            if (var_names[i]=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                print("(var_names[i]=='GPP') & ('CHTESSEL' in model_out_name)")
                value = var[model_out_name+'_vals']*(-1)
            else:
                value = var[model_out_name+'_vals']

            # smooth or not
            if smooth_type != 'no_soomth':
                value = smooth_vpd_series(value, window_size, order, smooth_type)

            # only use vpd data points > 200
            above_thres      = (var[model_out_name+'_vpd_num']>200)
            var_vpd_series = var['vpd_series'][above_thres]
            value          = value[above_thres]

            # Plot if the data point > 200
            if np.sum(var[model_out_name+'_vpd_num']-200) > 0:
                if model_out_name == 'obs':
                    lw=3
                else:
                    lw=2
                plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_color,
                                        alpha=0.8, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
                # if model_out_name == 'obs':
                #     # add error range of obs (I think top and bot boundary should be set to 1 sigema)
                    # vals_bot   = var['obs_bot'][above_thres]
                    # vals_top   = var['obs_top'][above_thres]

                    # fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                    #                         color=line_color, edgecolor="none", alpha=0.1) #  .rolling(window=10).mean()

            # ===== Drawing the turning points =====
            # Calculate turning points
            # if turning_point['calc']:
            #     ax[row,col].scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)

            if pdf_or_box=='boxplot':
                # ===== Drawing the box whisker =====
                # Calculate the interquartile range (IQR)
                # median, p25, p75, minimum, maximum = dist_box_values[model_out_name]
                median, p25, p75, minimum, maximum = [0,0,0,0,0]

                if col == 1:
                    xaxis_s = 7.1 + j*0.2-0.08
                    xaxis_e = 7.1 + j*0.2+0.08
                else:
                    xaxis_s = 5.7 + j*0.2-0.08
                    xaxis_e = 5.7 + j*0.2+0.08

                # Draw the box
                ax[row,col].add_patch(Polygon([[xaxis_s, p25], [xaxis_s, p75],[xaxis_e, p75], [xaxis_e, p25]],
                                              closed=True, color=line_color, fill=True, alpha=0.8, linewidth=0.1))

                # Draw the median line
                ax[row,col].plot([xaxis_s,xaxis_e], [median,median], color = almost_black, linewidth=0.5)

                # Draw the p25 p75
                ax[row,col].plot([xaxis_s, xaxis_e], [p25, p25], color = almost_black, linewidth=0.5)
                ax[row,col].plot([xaxis_s, xaxis_e], [p75, p75], color = almost_black, linewidth=0.5)

                ax[row,col].plot([xaxis_s, xaxis_s], [p25, p75], color = almost_black, linewidth=0.5)
                ax[row,col].plot([xaxis_e, xaxis_e], [p25, p75], color = almost_black, linewidth=0.5)

                # Draw the max and min
                ax[row,col].plot([xaxis_s+0.1, xaxis_e-0.1], [minimum, minimum], color = almost_black, linewidth=0.5)
                ax[row,col].plot([xaxis_s+0.1, xaxis_e-0.1], [maximum, maximum], color = almost_black, linewidth=0.5)
                ax[row,col].plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p75, maximum], color = almost_black, linewidth=0.5)
                ax[row,col].plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p25, minimum], color = almost_black, linewidth=0.5)

            elif pdf_or_box=='pdf':

                if col == 1:
                    xaxis_s = 7.1 + j*0.2-0.08
                    xaxis_e = 7.1 + j*0.2+0.08
                else:
                    xaxis_s = 5.5 + j*0.2-0.08
                    xaxis_e = 5.5 + j*0.2+0.08

                # set y range
                if i == 0:
                    axis_info = [0,500]
                elif i == 1:
                    axis_info = [0,100]
                elif i == 2:
                    # axis_info = [0,1.5]
                    axis_info = [-0.5,2.2]
                elif i == 3:
                    # axis_info = [0,0.5]
                    axis_info = [-0.5,0.5]

                # set scale
                scale = 3

                # set y axis
                y = np.linspace(axis_info[0], axis_info[1], 100)

                # pdf
                x = xaxis_s+dist_box_values[model_out_name+'_fraction']*scale

                # vertical line
                x_line = np.zeros(len(x))
                x_line = xaxis_s

                # one standard deviation
                y_mean = dist_box_values[model_out_name+'_mean']
                y_std  = dist_box_values[model_out_name+'_std']
                y_up   = y_mean + y_std
                y_bot  = y_mean - y_std
                y_up2   = y_mean + y_std*2
                y_bot2  = y_mean - y_std*2

                y_up_abs   = np.abs(y - y_up)
                y_up_index = np.argmin(y_up_abs)
                y_up2_abs   = np.abs(y - y_up2)
                y_up2_index = np.argmin(y_up2_abs)

                y_bot_abs   = np.abs(y - y_bot)
                y_bot_index = np.argmin(y_bot_abs)
                y_bot2_abs   = np.abs(y - y_bot2)
                y_bot2_index = np.argmin(y_bot2_abs)

                ax[row,col].plot(x[y_bot2_index:y_up2_index+1], y[y_bot2_index:y_up2_index+1], color = almost_black, linewidth=0.5)
                ax[row,col].plot(x_line[y_bot2_index:y_up2_index+1], y[y_bot2_index:y_up2_index+1], color = almost_black, linewidth=0.5)
                ax[row,col].fill_betweenx(y[y_bot_index:y_up_index+1], x_line[y_bot_index:y_up_index+1], x[y_bot_index:y_up_index+1], color=line_color, alpha=0.7)

        if col == 0:
            ax[row,col].legend(fontsize=7, frameon=False, ncol=3)


        if IGBP_type !=None:
            ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type+'site_num='+str(var['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

        # ax[row,col].set_xlim(0, 11)

    ax[0,0].set_title("Wet (EF>0.8)", fontsize=20)
    ax[0,1].set_title("Dry (EF<0.2)", fontsize=20)

    ax[0,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)

    if var_names[3] == 'NEE':
        ax[1,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
    elif var_names[3] == 'GPP':
        ax[1,0].set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    ax[1,0].set_xlabel("VPD (kPa)", fontsize=12)
    ax[1,1].set_xlabel("VPD (kPa)", fontsize=12)

    ax[0,0].set_xticks([0,1,2,3,4,5])
    ax[0,0].set_xticklabels(['0','1','2', '3','4','5'],fontsize=12)
    ax[0,0].set_xlim(-0.2,9.3)

    ax[0,1].set_xticks([0,1,2,3,4,5,6,7])
    ax[0,1].set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
    ax[0,1].set_xlim(-0.2,10.9)

    ax[1,0].set_xticks([0,1,2,3,4,5])
    ax[1,0].set_xticklabels(['0','1','2', '3','4','5'],fontsize=12)
    ax[1,0].set_xlim(-0.2,9.3)

    ax[1,1].set_xticks([0,1,2,3,4,5,6,7])
    ax[1,1].set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
    ax[1,1].set_xlim(-0.2,10.9)


    ax[0,0].tick_params(axis='y', labelsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)
    # ax[0,0].set_ylim(-50,80)
    # ax[0,1].set_ylim(-50,400)
    # ax[1,0].set_ylim(-0.6,0.2)
    # ax[1,1].set_ylim(-0.5,2)

    # ax[0,0].set_ylim(-50,80)
    # ax[0,1].set_ylim(-50,400)
    # ax[1,0].set_ylim(-0.6,0.2)
    # ax[1,1].set_ylim(-0.5,2)

    # fig.savefig("./plots/Fig_var_VPD_all_sites_daytime_line_box_coarse.png",bbox_inches='tight',dpi=300) # '_30percent'
    fig.savefig("./plots/Fig_var_VPD_"+message+"_line_box_coarse.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

def plot_var_VPD_line_box_three_col(bin_by=None, window_size=11, order=3,
                 smooth_type='S-G_filter', method='bin_by_vpd', message=None, model_names=None,
                 turning_point={'calc':False,'method':'kneed'}):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=2, ncols=3, figsize=[18,9],sharex=False, sharey=False, squeeze=True) #
    # fig, ax = plt.subplots(figsize=[10, 7])
    plt.subplots_adjust(wspace=0.09, hspace=0.02)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    order        = ['(a)','(b)','(c)','(d)',
                    '(e)','(f)','(g)','(h)']
    # ============== read data ==============

    file_names = [  f'./txt/Qle_VPD_daytime_'+bin_by+'_0.2-0.4_'+method+'_coarse.csv',
                    f'./txt/Qle_VPD_daytime_'+bin_by+'_0.4-0.6_'+method+'_coarse.csv',
                    f'./txt/Qle_VPD_daytime_'+bin_by+'_0.6-0.8_'+method+'_coarse.csv',
                    f'./txt/NEE_VPD_daytime_'+bin_by+'_0.2-0.4_'+method+'_coarse.csv',
                    f'./txt/NEE_VPD_daytime_'+bin_by+'_0.4-0.6_'+method+'_coarse.csv',
                    f'./txt/NEE_VPD_daytime_'+bin_by+'_0.6-0.8_'+method+'_coarse.csv',]

    boxplot_file_names = [  f'./txt/boxplot_metrics_Qle_VPD_daytime_{bin_by}_0.2-0.4_coarse.csv',
                            f'./txt/boxplot_metrics_Qle_VPD_daytime_{bin_by}_0.4-0.6_coarse.csv',
                            f'./txt/boxplot_metrics_Qle_VPD_daytime_{bin_by}_0.6-0.8_coarse.csv',
                            f'./txt/boxplot_metrics_NEE_VPD_daytime_{bin_by}_0.2-0.4_coarse.csv',
                            f'./txt/boxplot_metrics_NEE_VPD_daytime_{bin_by}_0.4-0.6_coarse.csv',
                            f'./txt/boxplot_metrics_NEE_VPD_daytime_{bin_by}_0.6-0.8_coarse.csv',]

    var_names  = ['Qle','Qle','Qle','NEE','NEE','NEE']

    for i, file_name in enumerate(file_names):

        # set plot row and col
        row = int(i/3)
        col = i%3

        # Read lines data
        var = pd.read_csv(file_name)

        # Read boxplot data
        dist_box_values = pd.read_csv(boxplot_file_names[i])

        # how to get the model out list from the column names???
        model_out_list = []
        for column_name in var.columns:
            if "_vals" in column_name:
                model_out_list.append(column_name.split("_vals")[0])

        # models to plot
        model_order     = []
        model_names_all = model_names['model_select']
        for model_name in model_names_all:
            if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
                model_order.append(model_name)

        # Add obs
        # if var_names[i] in ['Qle','NEE']:
        #     model_order.append('obs')
        print('model_order',model_order)

        # Calculate turning points
        if turning_point['calc']:

            nmodel   = len(model_out_list)
            nvpd     = len(var['vpd_series'])
            val_tmp  = np.zeros((nmodel,nvpd))

            for j, model_out_name in enumerate(model_order):
                vals_vpd_num = var[model_out_name+'_vpd_num']
                # find_turning_points_by_piecewise_regression will transfer NEE to NEP so don't need to do it here.
                val_tmp[j,:] = np.where(vals_vpd_num>200, var[model_out_name+'_vals'], np.nan)

            # Find the turning points
            if turning_point['method']=='kneed' :
                turning_points = find_turning_points_by_kneed(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='cdf' :
                turning_points = find_turning_points_by_cdf(model_order, var['vpd_series'], val_tmp)
            elif turning_point['method']=='piecewise':
                turning_points, slope = find_turning_points_by_piecewise_regression(model_order, var['vpd_series'], val_tmp, var_names[i])

        for j, model_out_name in enumerate(model_order):

            # set line color
            line_color = model_colors[model_out_name]

            # ===== Drawing the lines =====
            # Unify NEE units : upwards CO2 movement is positive values
            if (var_names[i]=='NEE') & ~(model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA']):
                print("(var_names[i]=='NEE') & ~(model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA'])")
                value = var[model_out_name+'_vals']*(-1)
            else:
                value = var[model_out_name+'_vals']

            if (var_names[i]=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                print("(var_names[i]=='GPP') & ('CHTESSEL' in model_out_name)")
                value = var[model_out_name+'_vals']*(-1)
            else:
                value = var[model_out_name+'_vals']

            # smooth or not
            if smooth_type != 'no_soomth':
                value = smooth_vpd_series(value, window_size, order, smooth_type)

            # only use vpd data points > 200
            above_thres      = (var[model_out_name+'_vpd_num']>200)
            var_vpd_series = var['vpd_series'][above_thres]
            value          = value[above_thres]

            # Plot if the data point > 200
            if np.sum(var[model_out_name+'_vpd_num']-200) > 0:

                plot = ax[row,col].plot(var_vpd_series, value, lw=2.0, color=line_color,
                                        alpha=0.8, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
                if model_out_name == 'obs':
                    # add error range of obs (I think top and bot boundary should be set to 1 sigema)
                    vals_bot   = var['obs_bot'][above_thres]
                    vals_top   = var['obs_top'][above_thres]
                    if (var_names[i]=='NEE') &  ~(model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA']):
                        vals_bot   = vals_bot*(-1)
                        vals_top   = vals_top*(-1)
                    fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                                            color=line_color, edgecolor="none", alpha=0.1) #  .rolling(window=10).mean()

            # ===== Drawing the turning points =====
            # Calculate turning points
            # if turning_point['calc']:
            #     ax[row,col].scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)

            # ===== Drawing the box whisker =====
            # Calculate the interquartile range (IQR)
            median, p25, p75, minimum, maximum = dist_box_values[model_out_name]
            # median, p25, p75, minimum, maximum = [50,25,75,10,90]
            # if col == 0:
            xaxis_s = 7.1 + j*0.2-0.08
            xaxis_e = 7.1 + j*0.2+0.08
            # else:
            #     xaxis_s = 5.7 + j*0.2-0.08
            #     xaxis_e = 5.7 + j*0.2+0.08

            # Draw the box
            ax[row,col].add_patch(Polygon([[xaxis_s, p25], [xaxis_s, p75],[xaxis_e, p75], [xaxis_e, p25]],
                                          closed=True, color=line_color, fill=True, alpha=0.8, linewidth=0.1))

            # Draw the median line
            ax[row,col].plot([xaxis_s,xaxis_e], [median,median], color = almost_black, linewidth=0.5)

            # Draw the p25 p75
            ax[row,col].plot([xaxis_s, xaxis_e], [p25, p25], color = almost_black, linewidth=0.5)
            ax[row,col].plot([xaxis_s, xaxis_e], [p75, p75], color = almost_black, linewidth=0.5)

            ax[row,col].plot([xaxis_s, xaxis_s], [p25, p75], color = almost_black, linewidth=0.5)
            ax[row,col].plot([xaxis_e, xaxis_e], [p25, p75], color = almost_black, linewidth=0.5)

            # Draw the max and min
            ax[row,col].plot([xaxis_s+0.1, xaxis_e-0.1], [minimum, minimum], color = almost_black, linewidth=0.5)
            ax[row,col].plot([xaxis_s+0.1, xaxis_e-0.1], [maximum, maximum], color = almost_black, linewidth=0.5)
            ax[row,col].plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p75, maximum], color = almost_black, linewidth=0.5)
            ax[row,col].plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p25, minimum], color = almost_black, linewidth=0.5)

        if col == 0 and row == 0:
            ax[row,col].legend(fontsize=8, frameon=False, ncol=3)

        if IGBP_type !=None:
            ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type+'site_num='+str(var['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

        ax[row,col].set_xticks([0,1,2,3,4,5,6,7])
        ax[row,col].set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
        ax[row,col].set_xlim(-0.2,10.9)
        ax[row,col].tick_params(axis='y', labelsize=12)

    ax[0,0].set_title("(0.2<EF<0.4)", fontsize=20)
    ax[0,1].set_title("(0.4<EF<0.6)", fontsize=20)
    ax[0,2].set_title("(0.6<EF<0.8)", fontsize=20)

    ax[0,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    ax[1,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    fig.savefig("./plots/Fig_var_VPD_all_sites_daytime_line_box_EF_0.2_0.6_coarse.png",bbox_inches='tight',dpi=300) # '_30percent'

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # Read site names, IGBP and clim
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    day_time       = True
    method         = 'CRV_bins' #'GAM'
    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    error_type     = 'one_std'

    # Smoothing setting
    window_size    = 11
    order          = 3
    smooth_type    = 'no_soomth' #'S-G_filter' #
    turning_point  =  {'calc':False, 'method':'piecewise'}
                      #{'calc':True, 'method':'cdf'}#{'calc':True, 'method':'kneed'}

    # for IGBP_type in IGBP_types:
    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'


    # =========== plot_var_VPD_uncertainty ===========
    day_time    = True
    energy_cor  = False
    time_scale  = 'hourly'
    country_code= None
    selected_by = 'EF_model'
    veg_fraction= None#[0.7,1.]
    standardize = None
    uncertain_type='UCRTN_bootstrap'
    method      = 'CRV_bins'
    IGBP_type   = None
    clim_type   = None
    clarify_site={'opt': True,
                  'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                  'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    var_name    = 'Qle'
    num_threshold=200

    # # ======== Plot EF wet & dry ========
    var_name='Qle'

    plot_var_VPD_uncertainty(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
                 selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize,
                 uncertain_type=uncertain_type, method=method, IGBP_type=IGBP_type, clim_type=clim_type,
                 clarify_site=clarify_site,num_threshold=num_threshold)

    plot_var_VPD_uncertainty_three_cols(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
                 selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize,
                 uncertain_type=uncertain_type, method=method, IGBP_type=IGBP_type, clim_type=clim_type,
                 clarify_site=clarify_site,num_threshold=num_threshold)


    IGBP_types = ['GRA', 'DBF', 'ENF', 'EBF']
    LAI_ranges = [[0.,1.],
                  [1.,2.],
                  [2.,4.],
                  [4.,10.],] #30
    plot_var_VPD_uncertainty_veg_LAI_eight(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
                 selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize,
                 uncertain_type=uncertain_type, method=method, IGBP_types=IGBP_types, LAI_ranges=LAI_ranges, clim_type=clim_type,
                 clarify_site=clarify_site,num_threshold=num_threshold)

    #
    # ## ======== Plot different veg types ========
    # bounds = [0,0.2]
    # IGBP_types    = ['GRA', 'DBF', 'ENF', 'EBF']
    # plot_var_VPD_uncertainty_veg_LAI(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
    #              selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize, bounds=bounds,
    #              uncertain_type=uncertain_type, method=method, IGBP_types=IGBP_types, clim_type=clim_type,
    #              clarify_site=clarify_site,num_threshold=num_threshold)
    #
    # bounds = [0.8,1.]
    # IGBP_types    = ['GRA', 'DBF', 'ENF', 'EBF']
    # plot_var_VPD_uncertainty_veg_LAI(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
    #              selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize, bounds=bounds,
    #              uncertain_type=uncertain_type, method=method, IGBP_types=IGBP_types, clim_type=clim_type,
    #              clarify_site=clarify_site,num_threshold=num_threshold)
    #
    # # ======== Plot LAI ranges ========
    # bounds = [0,0.2]
    # IGBP_types = None
    # LAI_ranges = [[0.,1.],
    #               [1.,2.],
    #               [2.,4.],
    #               [4.,10.],] #30
    #
    # plot_var_VPD_uncertainty_veg_LAI(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
    #              selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize, bounds=bounds,
    #              uncertain_type=uncertain_type, method=method, LAI_ranges=LAI_ranges, clim_type=clim_type,
    #              clarify_site=clarify_site,num_threshold=num_threshold)
    #
    # bounds = [0.8,1.]
    # IGBP_types = None
    # LAI_ranges = [[0.,1.],
    #               [1.,2.],
    #               [2.,4.],
    #               [4.,10.],] #30
    #
    # plot_var_VPD_uncertainty_veg_LAI(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
    #              selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize, bounds=bounds,
    #              uncertain_type=uncertain_type, method=method, LAI_ranges=LAI_ranges, clim_type=clim_type,
    #              clarify_site=clarify_site,num_threshold=num_threshold)
    #

    # ======== Plot high low veg ranges ========
    # LAI_ranges=None
    # veg_fraction=[0,0.3]
    # plot_var_VPD_uncertainty(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
    #              selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize,
    #              uncertain_type=uncertain_type, method=method, IGBP_type=IGBP_type, clim_type=clim_type,
    #              clarify_site=clarify_site,num_threshold=num_threshold)

    # veg_fraction=[0.7,1.]
    # plot_var_VPD_uncertainty(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
    #              selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize,
    #              uncertain_type=uncertain_type, method=method, IGBP_type=IGBP_type, clim_type=clim_type,
    #              clarify_site=clarify_site,num_threshold=num_threshold)
