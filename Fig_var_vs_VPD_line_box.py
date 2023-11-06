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

def plot_var_VPD_line_box(bin_by=None, window_size=11, order=3,
                 smooth_type='S-G_filter', method='bin_by_vpd', message=None, model_names=None,
                 turning_point={'calc':False,'method':'kneed'}):

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # ============== read data ==============
  
    file_names = [  f'./txt/Qle_VPD_daytime_'+bin_by+'_0-0.2_'+method+'_coarse.csv',
                    f'./txt/Qle_VPD_daytime_'+bin_by+'_0.8-1.0_'+method+'_coarse.csv',
                    f'./txt/NEE_VPD_daytime_'+bin_by+'_0-0.2_'+method+'_coarse.csv',
                    f'./txt/NEE_VPD_daytime_'+bin_by+'_0.8-1.0_'+method+'_coarse.csv',]
        
    boxplot_file_names = [  f'./txt/boxplot_metrics_Qle_VPD_daytime_{bin_by}_0-0.2_coarse.csv',
                            f'./txt/boxplot_metrics_Qle_VPD_daytime_{bin_by}_0.8-1.0_coarse.csv',
                            f'./txt/boxplot_metrics_NEE_VPD_daytime_{bin_by}_0-0.2_coarse.csv',
                            f'./txt/boxplot_metrics_NEE_VPD_daytime_{bin_by}_0.8-1.0_coarse.csv',]

    var_names  = ['Qle','Qle','NEE','NEE']

    for i, file_name in enumerate(file_names):

        # set plot row and col
        row = int(i/2)
        col = i%2

        # Read lines data
        var = pd.read_csv(file_name)

        # Read boxplot data
        box_metrics = pd.read_csv(boxplot_file_names[i])
        
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
        if var_names[i] in ['Qle','NEE']:
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
            if (var_names[i]=='NEE') & ~(model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA']):
                print("(var_names[i]=='NEE') & ~(model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA'])")
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

                plot = ax[row,col].plot(var_vpd_series, value, lw=2.0, color=line_color, 
                                        alpha=0.8, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
                if model_out_name == 'obs':
                    # add error range of obs (I think top and bot boundary should be set to 1 sigema)
                    vals_bot   = var['obs_bot'][above_200]
                    vals_top   = var['obs_top'][above_200]
                    if (var_names[i]=='NEE') &  ~(model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA']):
                        vals_bot   = vals_bot*(-1)
                        vals_top   = vals_top*(-1)
                    fill = ax[row,col].fill_between(var_vpd_series,vals_bot,vals_top,
                                            color=line_color, edgecolor="none", alpha=0.05) #  .rolling(window=10).mean()

            # ===== Drawing the turning points =====
            # Calculate turning points
            if turning_point['calc']:
                ax[row,col].scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)

            # ===== Drawing the box whisker =====
            # Calculate the interquartile range (IQR)
            median, p25, p75, minimum, maximum = box_metrics[model_out_name]
            # median, p25, p75, minimum, maximum = [50,25,75,10,90]
            if col == 0:
                xaxis_s = 7.1 + j*0.2-0.08
                xaxis_e = 7.1 + j*0.2+0.08
            else:
                xaxis_s = 5.9 + j*0.2-0.08
                xaxis_e = 5.9 + j*0.2+0.08

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
        if col == 0:
            ax[row,col].legend(fontsize=6, frameon=False, ncol=3)

        # if IGBP_type !=None:
        #     ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)
        # if clim_type !=None:
        #     ax[1,0].text(0.12, 0.92, 'Clim_type='+clim_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

        # ax[row,col].text(0.12, 0.87, 'site_num='+str(var['site_num'][0]), va='bottom', ha='center', 
        #                  rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=12)
        ax[row,col].set_xlim(0, 11)

    # ax[0,0].set_ylim(-50,80)
    # ax[0,1].set_ylim(-50,400)
    # ax[1,0].set_ylim(-0.6,0.2)
    # ax[1,1].set_ylim(-0.5,2)

    fig.savefig("./plots/Fig_var_VPD_all_sites_daytime_line_box_coarse.png",bbox_inches='tight',dpi=300) # '_30percent'


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # Read site names, IGBP and clim
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    
    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    day_time       = True
    method         = 'bin_by_vpd' #'GAM'
    # Smoothing setting

    window_size    = 11
    order          = 3
    smooth_type    = 'no_soomth' #'S-G_filter' #
    turning_point  =  {'calc':True, 'method':'piecewise'}
                      #{'calc':True, 'method':'cdf'}#{'calc':True, 'method':'kneed'}
    
    message        = 'all sites'
    plot_var_VPD_line_box( bin_by=bin_by, window_size=window_size, order=order,
                 smooth_type=smooth_type, method='bin_by_vpd', message=message, model_names=model_names,
                 turning_point=turning_point)

