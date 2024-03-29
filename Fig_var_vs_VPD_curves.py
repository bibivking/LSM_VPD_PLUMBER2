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
                 num_threshold=200, class_by=None, dist_type=None):

    # ============== convert units =================
    to_ms = False
    if to_ms and var_name == 'Gs':
        # 25 deg C, 10100 Pa
        mol2ms = 8.313*(273.15+25)/10100

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
    if class_by=='gs_eq':
        model_colors = set_model_colors_Gs_based()
    else:
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

    model_order = model_names['model_select_new']
    print('model_order',model_order)

    for j, model_out_name in enumerate(model_order):

        # set line color
        line_color = model_colors[model_out_name]
        # if model_out_name in ['ORC2_r6593','ORC3_r8120','GFDL','QUINCY','NoahMPv401']:
        #     linestyle = 'dotted'
        # else:
        #     linestyle = 'solid'

        linestyle = 'solid'

        if dist_type!=None:
            file_names = [  f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message_wet}_{model_out_name}_{dist_type}.csv',
                            f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message_dry}_{model_out_name}_{dist_type}.csv',]
        else:
            file_names = [  f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message_wet}_{model_out_name}.csv',
                            f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message_dry}_{model_out_name}.csv',]

        for i, file_name in enumerate(file_names):

            # set plot row and col
            row = int(i/2)
            col = i%2

           # Read lines data
            var = pd.read_csv(file_name)

            # ===== Drawing the lines =====
            # Unify NEE units : upwards CO2 movement is positive values

            if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
                value        = var['y_pred']*(-1)
                vals_bot_tmp = var['y_int_bot']
                vals_top_tmp = var['y_int_top']
            elif var_name=='Gs' and to_ms:
                value        = var['y_pred']*mol2ms
                vals_bot_tmp = var['y_int_bot']*mol2ms
                vals_top_tmp = var['y_int_top']*mol2ms
            else:
                value        = var['y_pred']
                vals_bot_tmp = var['y_int_bot']
                vals_top_tmp = var['y_int_top']

            # smooth or not
            if smooth_type != 'no_soomth':
                value = smooth_vpd_series(value, window_size, order, smooth_type)

            var_vpd_series = var['vpd_pred']
            vals_bot       = vals_bot_tmp
            vals_top       = vals_top_tmp

            # Plot if the data point > num_threshold
            if model_out_name == 'obs':
                lw=2.5
            else:
                lw=1.5
            plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_color,linestyle=linestyle,
                                    alpha=1., label=change_model_name(model_out_name)) #edgecolor='none', c='red' .rolling(window=10).mean()

            # Plot uncertainty
            # add error range of obs (I think top and bot boundary should be set to 1 sigema)
            fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                                    color=line_color, edgecolor="none", alpha=0.3) #  .rolling(window=10).mean()

            ax[row,col].text(0.05, 0.92, order[i], va='bottom', ha='center', rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=14)

        if col == 1:
            ax[row,col].legend(fontsize=7.2, frameon=False, ncol=2)

    ax[0,0].set_title("Wet (EF>0.8)", fontsize=16)
    ax[0,1].set_title("Dry (EF<0.2)", fontsize=16)

    if var_name == 'Qle':
        ax[0,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    elif var_name == 'Gs' and to_ms:
        ax[0,0].set_ylabel("Canopy Stomatal Conductance\n(m s$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'Gs':
        ax[0,0].set_ylabel("Canopy Stomatal Conductance\n(mol m$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'NEE':
        ax[0,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'GPP':
        ax[0,0].set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    ax[0,0].set_xlabel("VPD (kPa)", fontsize=12)
    ax[0,1].set_xlabel("VPD (kPa)", fontsize=12)

    if var_name == 'Qle':
        if time_scale == 'daily':
            ax[0,0].set_xlim(-0.1,2.7)
            ax[0,0].set_ylim(0,170)
            ax[0,1].set_xlim(-0.1,5.1)
            ax[0,1].set_ylim(0,40)
        elif time_scale == 'hourly':
            # ax[0,0].set_xlim(-0.1,10.5)
            ax[0,0].set_xlim(-0.1,5.5)
            ax[0,0].set_xticks([0,1,2,3,4,5])
            ax[0,0].set_xticklabels(['0','1','2', '3','4','5'],fontsize=12)
            ax[0,0].set_ylim(0,350)

            # ax[0,1].set_xlim(-0.1,10.5)
            ax[0,1].set_xlim(-0.1,7.5)
            ax[0,1].set_xticks([0,1,2,3,4,5,6,7])
            ax[0,1].set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
            ax[0,1].set_ylim(0,100)

    # ax[0,1].set_xticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
    # ax[0,1].set_xticklabels(['0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5'],fontsize=12)

    ax[0,0].tick_params(axis='y', labelsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)

    if dist_type !=None:
        message = '_'+dist_type
    else:
        message = ''

    if class_by !=None:
        fig.savefig(f"./plots/Fig_{var_name}_VPD{file_message_dry}_color_{class_by}{message}.png",bbox_inches='tight',dpi=300) # '_30percent'
    else:
        fig.savefig(f"./plots/Fig_{var_name}_VPD{file_message_dry}{message}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

def plot_var_VPD_uncertainty_three_cols(var_name=None, day_time=False, energy_cor=False, time_scale=None, country_code=None,
                 selected_by=None, veg_fraction=None,  standardize=None, uncertain_type='UCRTN_percentile',
                 method='CRV_bins', IGBP_type=None, clim_type=None, clarify_site={'opt':False,'remove_site':None},
                 num_threshold=200, class_by=None, dist_type=None):

    # ============== convert units =================
    to_ms = False
    if to_ms and var_name == 'Gs':
        # 25 deg C, 10100 Pa
        mol2ms = 8.313*(273.15+25)/10100

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
    if class_by=='gs_eq':
        model_colors = set_model_colors_Gs_based()
    else:
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

    # models to plot
    model_order     = model_names['model_select_new']
    print('model_order',model_order)

    for j, model_out_name in enumerate(model_order):

        # set line color
        line_color = model_colors[model_out_name]
        # if model_out_name in ['ORC2_r6593','ORC3_r8120','GFDL','QUINCY','NoahMPv401']:
        #     linestyle = 'dotted'
        # else:
        #     linestyle = 'solid'

        linestyle = 'solid'

        if dist_type != None:
            file_names = [  f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message1}_{model_out_name}_{dist_type}.csv',
                            f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message2}_{model_out_name}_{dist_type}.csv',
                            f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message3}_{model_out_name}_{dist_type}.csv',]
        else:
            file_names = [  f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message1}_{model_out_name}.csv',
                            f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message2}_{model_out_name}.csv',
                            f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message3}_{model_out_name}.csv',]

        for i, file_name in enumerate(file_names):

            # set plot row and col
            row = int(i/3)
            col = i%3

            # Read lines data
            var = pd.read_csv(file_name)

            # ===== Drawing the lines =====
            # Unify NEE units : upwards CO2 movement is positive values

            if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
                value        = var['y_pred']*(-1)
                vals_bot_tmp = var['y_int_bot']
                vals_top_tmp = var['y_int_top']
            elif var_name=='Gs' and to_ms:
                value        = var['y_pred']*mol2ms
                vals_bot_tmp = var['y_int_bot']*mol2ms
                vals_top_tmp = var['y_int_top']*mol2ms
            else:
                value        = var['y_pred']
                vals_bot_tmp = var['y_int_bot']
                vals_top_tmp = var['y_int_top']

            # smooth or not
            if smooth_type != 'no_soomth':
                value = smooth_vpd_series(value, window_size, order, smooth_type)

            var_vpd_series = var['vpd_pred']
            vals_bot       = vals_bot_tmp
            vals_top       = vals_top_tmp

            # Plot
            if model_out_name == 'obs':
                lw=2.5
            else:
                lw=1.5
            plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_color,linestyle=linestyle,
                                    alpha=1., label=change_model_name(model_out_name)) #edgecolor='none', c='red' .rolling(window=10).mean()

            # Plot uncertainty
            # add error range of obs (I think top and bot boundary should be set to 1 sigema)
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
    elif var_name == 'Gs' and to_ms:
        ax[0,0].set_ylabel("Canopy Stomatal Conductance\n(m s$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'Gs':
        ax[0,0].set_ylabel("Canopy Stomatal Conductance\n(mol m$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'NEE':
        ax[0,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
    elif var_name == 'GPP':
        ax[0,0].set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    ax[0,0].set_xlabel("VPD (kPa)", fontsize=12)
    ax[0,1].set_xlabel("VPD (kPa)", fontsize=12)
    ax[0,2].set_xlabel("VPD (kPa)", fontsize=12)

    ax[0,0].set_xticks([0,1,2,3,4,5,6.,7.])
    ax[0,0].set_xticklabels(['0','1','2','3','4','5','6','7'],fontsize=12)
    ax[0,0].set_xlim(-0.1,7.1)

    if var_name == 'Qle':
        if time_scale == 'daily':
            ax[0,0].set_ylim(0,100)
        elif time_scale == 'hourly':
            ax[0,0].set_ylim(0,300)

    ax[0,1].set_xticks([0,1,2,3,4,5,6.,7.])
    ax[0,1].set_xticklabels(['0','1','2','3','4','5','6','7'],fontsize=12)
    ax[0,1].set_xlim(-0.1,7.1)

    if var_name == 'Qle':
        if time_scale == 'daily':
            ax[0,1].set_ylim(0,100)
        elif time_scale == 'hourly':
            ax[0,1].set_ylim(0,300)

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

    if dist_type != None:
        message = "_"+dist_type
    else:
        message = ""

    if class_by != None:
        fig.savefig(f"./plots/Fig_{var_name}_VPD{file_message1}_color_{class_by}{message}.png",bbox_inches='tight',dpi=300) # '_30percent'
    else:
        fig.savefig(f"./plots/Fig_{var_name}_VPD{file_message1}{message}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

def plot_var_VPD_uncertainty_veg_LAI_four(var_name=None, day_time=False, energy_cor=False, time_scale=None, country_code=None,
                 selected_by=None, veg_fraction=None,  standardize=None, uncertain_type='UCRTN_percentile',
                 method='CRV_bins', IGBP_types=None, LAI_ranges=None, clim_type=None,
                 clarify_site={'opt':False,'remove_site':None}, num_threshold=200, class_by=None, dist_type=None):

    # ============== convert units =================
    to_ms = False
    if to_ms and var_name == 'Gs':
        # 25 deg C, 10100 Pa
        mol2ms = 8.313*(273.15+25)/10100

    # ============ Setting for plotting ============
    ncol     = 4
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=2, ncols=4, figsize=[20,8],sharex=False, sharey=False, squeeze=False)
    # fig, ax = plt.subplots(figsize=[10, 7])

    plt.subplots_adjust(wspace=0.13, hspace=0.10)

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
    if class_by=='gs_eq':
        model_colors = set_model_colors_Gs_based()
    else:
        model_colors = set_model_colors()

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    order        = ['(a)','(b)','(c)','(d)',
                    '(e)','(f)','(g)','(h)',
                    '(i)','(j)','(k)','(l)',
                    '(m)','(n)','(o)','(p)']

    # ============ Set the input file name ============
    ## Wet periods
    if dist_type!= None:
        message = "_"+dist_type
    else:
        message = ""

    # models to plot
    model_order = model_names['model_select_new']
    print('model_order', model_order)

    for j, model_out_name in enumerate(model_order):

        file_names = []

        # set line color
        line_color = model_colors[model_out_name]
        # if model_out_name in ['ORC2_r6593','ORC3_r8120','GFDL','QUINCY','NoahMPv401']:
        #     linestyle = 'dotted'
        # else:
        #     linestyle = 'solid'

        linestyle = 'solid'

        # Loop veg types
        if IGBP_types!=None:
            bounds = [0.8,1.0]
            for IGBP_type in IGBP_types:
                folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                            IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                            country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                            uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)
                file_names.append(f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message}_{model_out_name}{message}.csv')

            bounds = [0,0.2]
            for IGBP_type in IGBP_types:
                folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                            IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                            country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                            uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)
                file_names.append(f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message}_{model_out_name}{message}.csv')

        if LAI_ranges!=None:
            # Loop veg types
            bounds = [0.8,1.0]
            for LAI_range in LAI_ranges:
                folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                            LAI_range=LAI_range, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                            country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                            uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

                file_names.append(f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message}_{model_out_name}{message}.csv')

            bounds = [0,0.2]
            for LAI_range in LAI_ranges:
                folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                            LAI_range=LAI_range, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                            country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                            uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

                file_names.append(f'./txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message}_{model_out_name}{message}.csv')

        for i, file_name in enumerate(file_names):

            if os.path.exists(file_name):

                # set plot row and col
                row = int(i/ncol)
                col = i%ncol

                # Read lines data
                var = pd.read_csv(file_name)

                # ===== Drawing the lines =====
                # Unify NEE units : upwards CO2 movement is positive values
                if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                    print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
                    value        = var['y_pred']*(-1)
                    vals_bot_tmp = var['y_int_bot']
                    vals_top_tmp = var['y_int_top']
                elif var_name=='Gs' and to_ms:
                    value        = var['y_pred']*mol2ms
                    vals_bot_tmp = var['y_int_bot']*mol2ms
                    vals_top_tmp = var['y_int_top']*mol2ms
                else:
                    value        = var['y_pred']
                    vals_bot_tmp = var['y_int_bot']
                    vals_top_tmp = var['y_int_top']

                # smooth or not
                if smooth_type != 'no_soomth':
                    value = smooth_vpd_series(value, window_size, order, smooth_type)

                var_vpd_series = var['vpd_pred']
                vals_bot       = vals_bot_tmp
                vals_top       = vals_top_tmp

                # Plot
                if model_out_name == 'obs':
                    lw=2.5
                else:
                    lw=1.5

                plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_color, linestyle=linestyle,
                                        alpha=1., label=change_model_name(model_out_name)) #edgecolor='none', c='red' .rolling(window=10).mean()

                # Plot uncertainty
                # add error range of obs (I think top and bot boundary should be set to 1 sigema)
                fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                                        color=line_color, edgecolor="none", alpha=0.3) #  .rolling(window=10).mean()

                # ===== Drawing the turning points =====
                # Calculate turning points
                # if turning_point['calc']:
                #     ax[row,col].scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)
                ax[row,col].text(0.05, 0.90, order[i], va='bottom', ha='center', rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=14)

                if row==0 and col == 1 and var_name =='Gs':
                    ax[row,col].legend(fontsize=8, frameon=False, ncol=1)
                if row==0 and col == 1 and var_name =='Qle':
                    ax[row,col].legend(fontsize=8, frameon=False, ncol=1)

                if IGBP_types != None and row==0:
                    ax[row,col].set_title(IGBP_types[col], fontsize=14)

                if LAI_ranges != None and row==0:
                    ax[row,col].set_title(f'{LAI_ranges[col][0]}<LAI<{LAI_ranges[col][1]}', fontsize=14)

                if row == 1:
                    ax[row,col].set_xlabel("VPD (kPa)", fontsize=12)
                ax[row,col].tick_params(axis='y', labelsize=12)

                ax[row,col].set_xticks([0,1,2,3,4,5,6,7]) #
                ax[row,col].set_xticklabels(['0','1','2','3','4','5','6','7'],fontsize=12) #

                if row == 1:
                    ax[row,col].set_xlim(-0.1,7.1)
                    if var_name == 'Qle':
                        ax[row,col].set_ylim(0,120)
                else:
                    ax[row,col].set_xlim(-0.1,7.1)
                    if var_name == 'Qle':
                        ax[row,col].set_ylim(0,400)

                if var_name == 'Qle':
                    ax[row,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
                elif var_name == 'Gs' and to_ms:
                    ax[row,0].set_ylabel("Surface Stomatal Conductance\n(m s$\mathregular{^{-1}}$)", fontsize=12)
                elif var_name == 'Gs':
                    ax[row,0].set_ylabel("Surface Stomatal Conductance\n(mol m$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$)", fontsize=12)
                elif var_name == 'NEE':
                    ax[row,0].set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
                elif var_name == 'GPP':
                    ax[row,0].set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)

    if IGBP_types!=None:
        message1 = "veg"
    if LAI_ranges!=None:
        message1 = "LAI"

    if class_by != None:
        fig.savefig(f"./plots/Fig_{var_name}_VPD_wet_dry_{message1}_color_{class_by}{message}.png",bbox_inches='tight',dpi=300) # '_30percent'
    else:
        fig.savefig(f"./plots/Fig_{var_name}_VPD_wet_dry_{message1}{message}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

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
    # method       = 'CRV_fit_GAM_complex'
    dist_type    = None #'Gamma' # None #'Linear' #'Poisson' # 'Gamma'

    IGBP_type   = None
    clim_type   = None
    clarify_site={'opt': True,
                  'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                  'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    var_name    = 'Qle'
    num_threshold= 200

    # # ======== Plot EF wet & dry ========
    var_name = 'TVeg'
    class_by = None # 'gs_eq'

    plot_var_VPD_uncertainty(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
                 selected_by=selected_by, veg_fraction=veg_fraction, standardize=standardize,
                 uncertain_type=uncertain_type, method=method, IGBP_type=IGBP_type, clim_type=clim_type,
                 clarify_site=clarify_site,num_threshold=num_threshold, class_by=class_by, dist_type=dist_type)

    plot_var_VPD_uncertainty_three_cols(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
                 selected_by=selected_by, veg_fraction=veg_fraction,  standardize=standardize,
                 uncertain_type=uncertain_type, method=method, IGBP_type=IGBP_type, clim_type=clim_type,
                 clarify_site=clarify_site,num_threshold=num_threshold, class_by=class_by, dist_type=dist_type)

    IGBP_types = ['GRA', 'DBF', 'ENF', 'EBF']
    LAI_ranges = None
    plot_var_VPD_uncertainty_veg_LAI_four(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
                 selected_by=selected_by, veg_fraction=veg_fraction, standardize=standardize,
                 uncertain_type=uncertain_type, method=method, IGBP_types=IGBP_types, LAI_ranges=LAI_ranges, clim_type=clim_type,
                 clarify_site=clarify_site,num_threshold=num_threshold, class_by=class_by, dist_type=dist_type)

    IGBP_types = None
    LAI_ranges = [[0.,1.],
                  [1.,2.],
                  [2.,4.],
                  [4.,10.],] #30
    plot_var_VPD_uncertainty_veg_LAI_four(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
                 selected_by=selected_by, veg_fraction=veg_fraction, standardize=standardize,
                 uncertain_type=uncertain_type, method=method, IGBP_types=IGBP_types, LAI_ranges=LAI_ranges, clim_type=clim_type,
                 clarify_site=clarify_site,num_threshold=num_threshold, class_by=class_by, dist_type=dist_type)
