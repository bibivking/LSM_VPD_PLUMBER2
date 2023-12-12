import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from kneed import KneeLocator
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *
from plot_script import *
from calc_turning_points import *
from PLUMBER2_VPD_common_utils import *

def plot_turning_points(case_names, turning_points, slopes, model_order, message=None):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow

    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[10,6],sharex=True, sharey=False, squeeze=True) #

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
    # model_out_list      = model_names['all_model']
    # if var_name in ['NEE','Qle']:
    #     model_out_list.append('obs')

    all_turining_points = np.zeros((len(model_order),len(case_names)))

    for i, case_name in enumerate(case_names):

        turning_points_tmp = turning_points[case_name]
        slopes_tmp         = slopes[case_name]
        print('slopes_tmp',slopes_tmp)

        for j, model_out_name in enumerate(model_order):

            line_color               = model_colors[model_out_name]
            all_turining_points[j,i] = turning_points_tmp[model_out_name][0]
            
            if ~np.isnan(slopes_tmp[model_out_name][0]):
                print('Point 2 slopes_tmp[model_out_name]',slopes_tmp[model_out_name])
                if slopes_tmp[model_out_name][0] > slopes_tmp[model_out_name][1]:
                    if i ==0:
                        plot = ax.scatter(i, turning_points_tmp[model_out_name][0], marker='^', color=line_color, label=model_out_name, alpha=0.9, s=20) #  label=model_out_name,
                    else:
                        plot = ax.scatter(i, turning_points_tmp[model_out_name][0], marker='^', color=line_color, alpha=0.9, s=20) #  label=model_out_name,  
                else:
                    if i ==0:
                        plot = ax.scatter(i, turning_points_tmp[model_out_name][0], marker='v', color=line_color, label=model_out_name, alpha=0.9, s=20) #  label=model_out_name,
                    else:
                        plot = ax.scatter(i, turning_points_tmp[model_out_name][0], marker='v', color=line_color, alpha=0.9, s=20) #  label=model_out_name,
                    


    for j, model_out_name in enumerate(model_order):
        line_color = model_colors[model_out_name]
        # plot       = ax.plot(all_turining_points[j,:], lw=1.0, color=line_color, alpha=0.5, label=model_out_name)

    ax.legend(fontsize=6, frameon=False, ncol=3)

    ax.set_xlim(-0.5, len(case_names)-0.5)
    ax.set_ylim(-0.1, 8)
    ax.set_xticks(ticks=np.arange(len(case_names)), labels=case_names)

    fig.savefig("./plots/plot_turning_points_"+message+'.png',bbox_inches='tight',dpi=300) # '_30percent'

if __name__ == "__main__":

    # Load site_names, IGBP_types, clim_types
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    var_name       = 'Qle'#'Qle'  #
    bin_by         = 'EF_model' #'EF_model' #'EF_model'
    day_time       = True
    energy_cor     = True
    low_bound      = [0,0.2]#[0.8,1.0]
    method         = 'bin_by_vpd' #'GAM'

    window_size    = 11
    order          = 2
    smooth_type    = 'S-G_filter'


    if var_name == 'NEE':
        energy_cor     = False

    # ============== read data ==============
    # Set file names
    file_names = [f'./txt/{var_name}_VPD_daytime_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'_'+method+'_coarse.csv']
    case_names = ['all']

    # for IGBP_type in IGBP_types:
    #     file_names.append(f'./txt/{var_name}_VPD_daytime_IGBP='+IGBP_type+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'_'+method+'_coarse.csv')
    #     case_names.append(IGBP_type)
    # message        = var_name+"_all_IGBP"
    for clim_type in clim_types:
        file_names.append(f'./txt/{var_name}_VPD_daytime_clim='+clim_type+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'_'+method+'_coarse.csv')
        case_names.append(clim_type)
    message        = var_name+"_all_clim"

    turning_points = {}
    slopes         = {}

    # Calculate turning points
    for j, file_name in enumerate(file_names):

        var_bin_by_VPD = pd.read_csv(file_name)

        # Get model namelists
        model_out_list = []
        for column_name in var_bin_by_VPD.columns:
            if "_vals" in column_name:
                model_out_list.append(column_name.split("_vals")[0])

        vpd_series      = var_bin_by_VPD['vpd_series']

        nmodel          = len(model_out_list)
        nvpd            = len(vpd_series)

        # ================= Smoothing =================
        # Smoothing the curve and remove vpd_num < 100.
        vals_smooth = np.zeros((nmodel,nvpd))

        for i, model_out_name in enumerate(model_out_list):

            # vals_smooth_tmp = smooth_vpd_series(var_bin_by_VPD[model_out_name+'_vals'],
            #                     window_size=window_size, order=order,
            #                     smooth_type=smooth_type)
            vals_smooth_tmp = var_bin_by_VPD[model_out_name+'_vals']
            vals_vpd_num    = var_bin_by_VPD[model_out_name+'_vpd_num']

            vals_smooth[i,:]=vals_smooth_tmp # np.where(vals_vpd_num>200,vals_smooth_tmp,np.nan)

        # Find the turning points
        turning_points[case_names[j]], slopes[case_names[j]] = \
                             find_turning_points_by_piecewise_regression(model_out_list, vpd_series, vals_smooth, var_name)

    model_order = []
    model_names_all = model_names['all_model']
    for model_name in model_names_all:
        if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
            model_order.append(model_name)

    if var_name in ['Qle','NEE']:
        model_order.append('obs')

    plot_turning_points(case_names, turning_points, slopes, model_order, message=message)
