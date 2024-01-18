'''
Including
    def plot_var_VPD
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
from calc_turning_points import *
from PLUMBER2_VPD_common_utils import *

def plot_var_VPD(bin_by=None, window_size=11, order=3,
                 smooth_type='S-G_filter', method='bin_by_vpd', message=None, model_names=None,
                 day_time=None,  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                 error_type=None, pdf_or_box='pdf',
                 IGBP_type=None, clim_type=None,turning_point={'calc':False,'method':'kneed'}):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow 

    fig, ax  = plt.subplots(nrows=1, ncols=2, figsize=[14,5],sharex=False, sharey=False, squeeze=True) #

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

    file_names = [  f'./txt/VPD_curve/{folder_name}/{subfolder}Qle_VPD_'+message+'_'+bin_by+'_0.8-1.0_'+method+'_coarse.csv',
                    f'./txt/VPD_curve/{folder_name}/{subfolder}Qle_VPD_'+message+'_'+bin_by+'_0-0.2_'+method+'_coarse.csv',]

    print('Reading', file_names)

    var_names  = ['Qle','Qle']

    for i, file_name in enumerate(file_names):

        # set plot row and col
        col = i

        # Read lines data
        var = pd.read_csv(file_name)

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
            above_200      = (var[model_out_name+'_vpd_num']>200)
            var_vpd_series = var['vpd_series'][above_200]
            value          = value[above_200]

            # Plot if the data point > 200
            if np.sum(var[model_out_name+'_vpd_num']-200) > 0:
                if model_out_name == 'obs':
                    lw=3
                else:
                    lw=2
                plot = ax[col].plot(var_vpd_series, value, lw=lw, color=line_color,
                                        alpha=0.8, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

        if col == 0:
            ax[col].legend(fontsize=7, frameon=False, ncol=3)

        if IGBP_type !=None:
            ax[1].text(0.12, 0.92, 'IGBP='+IGBP_type+'site_num='+str(var['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[0].set_title("Wet (EF>0.8)", fontsize=20)
    ax[1].set_title("Dry (EF<0.2)", fontsize=20)

    ax[0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)

    ax[0].set_xlabel("VPD (kPa)", fontsize=12)
    ax[1].set_xlabel("VPD (kPa)", fontsize=12)

    ax[0].set_xticks([0,1,2,3,4,5])
    ax[0].set_xticklabels(['0','1','2', '3','4','5'],fontsize=12)
    ax[0].set_xlim(-0.2,5.)

    ax[1].set_xticks([0,1,2,3,4,5,6,7])
    ax[1].set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
    ax[1].set_xlim(-0.2,7.)

    ax[0].tick_params(axis='y', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    # ax[0].set_ylim(-50,80)
    # ax[1].set_ylim(-50,400)
    # ax[0].set_ylim(-50,80)
    # ax[1].set_ylim(-50,400)

    fig.savefig("./plots/Fig_var_VPD_"+message+"_coarse.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # Read site names, IGBP and clim
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    day_time       = False
    method         = 'bin_by_vpd' #'GAM'
    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    error_type     = 'one_std'
    # Smoothing setting

    window_size    = 11
    order          = 3
    smooth_type    = 'no_soomth' #'S-G_filter' #
    turning_point  =  {'calc':True, 'method':'piecewise'}
                      #{'calc':True, 'method':'cdf'}#{'calc':True, 'method':'kneed'}

    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'
    plot_var_VPD_line_box( bin_by=bin_by, window_size=window_size, order=order,
             smooth_type=smooth_type, method='bin_by_vpd', model_names=model_names,
             day_time=day_time,  clarify_site=clarify_site, standardize=standardize,
             error_type=error_type,
             turning_point=turning_point) # IGBP_type=IGBP_type,
