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

def plot_var_VPD_line_box(var_name, bin_by=None, window_size=11, order=3,
                 smooth_type='S-G_filter', method='bin_by_vpd', message=None, model_names=None,
                 day_time=None,  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                 error_type=None, pdf_or_box='pdf', region_name=None,
                 IGBP_type=None, clim_type=None,turning_point={'calc':False,'method':'kneed'}):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[8,5],sharex=False, sharey=False, squeeze=True) #
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

    # ============== read data ==============
    message = ''
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

    if region_name !=None:
        message = message + '_'+region_name

    folder_name = 'original'

    if standardize != None:
        folder_name = 'standardized_'+standardize

    if clarify_site['opt']:
        folder_name = folder_name+'_clarify_site'

    if var_name == 'NEE':
        var_name = 'NEP'

    file_name = f'./txt/VPD_curve/{folder_name}/{subfolder}{var_name}_VPD_{message}_{bin_by}_0-1.0_{method}_coarse.csv'

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
    if var_name in ['Qle','NEE','GPP']:
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
            plot = ax.plot(var_vpd_series, value, lw=lw, color=line_color,
                                    alpha=0.8, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
            # vals_bot   = var[model_out_name+'_bot'][above_200]
            # vals_top   = var[model_out_name+'_top'][above_200]
            # fill = ax.fill_between(var_vpd_series,vals_bot,vals_top, color=line_color, edgecolor="none", alpha=0.1) #  .rolling(window=10).mean()

        # ===== Drawing the turning points =====
        # Calculate turning points
        # if turning_point['calc']:
        #     ax.scatter(turning_points[model_out_name][0], turning_points[model_out_name][1], marker='o', color=line_color, s=20)

    # ===== Plot box whisker =====
    if region_name!=None:
        boxplot_file_names = ['/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_historical_global_land_metrics_'+region_name+'.csv',
                                '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp126_global_land_metrics_'+region_name+'.csv',
                                '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp245_global_land_metrics_'+region_name+'.csv',
                                '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp585_global_land_metrics_'+region_name+'.csv',]
    else:
        boxplot_file_names = ['/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_historical_global_land_metrics.csv',
                                '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp126_global_land_metrics.csv',
                                '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp245_global_land_metrics.csv',
                                '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp585_global_land_metrics.csv',]


    colors = ['black','green','orange','red']

    for i, boxplot in enumerate(boxplot_file_names):

        boxplot_metrics = pd.read_csv(boxplot)
        VPD_mean, VPD_std, p5, p25, p75, p95 = boxplot_metrics['all_model']
        if var_name == 'Qle':
            middle = 190
            space  = 13
        elif var_name == 'NEP':
            middle = 0.96
            space  = 0.11
        elif var_name == 'GPP':
            middle = 0.96
            space  = 0.09
        yaxis_s = middle + i*space-space*0.36
        yaxis_e = middle + i*space+space*0.36
        yaxis_interval = yaxis_e-yaxis_s

        # Draw the box
        # ax.add_patch(Polygon([[yaxis_s, p25], [yaxis_s, p75],[yaxis_e, p75], [yaxis_e, p25]],
        #                               closed=True, color=line_color, fill=True, alpha=0.8, linewidth=0.1))
        p5  = VPD_mean - 2*VPD_std
        p25 = VPD_mean - VPD_std
        p75 = VPD_mean + VPD_std
        p95 = VPD_mean + 2*VPD_std
        # Draw the median line
        ax.plot( [VPD_mean,VPD_mean], [yaxis_s,yaxis_e], color = colors[i], linewidth=1.5)

        # Draw the p25 p75
        ax.plot( [p25, p25], [yaxis_s, yaxis_e], color = colors[i], linewidth=1.5)
        ax.plot( [p75, p75], [yaxis_s, yaxis_e],  color = colors[i], linewidth=1.5)

        ax.plot([p25, p75],  [yaxis_s, yaxis_s],  color = colors[i], linewidth=1.5)
        ax.plot([p25, p75],  [yaxis_e, yaxis_e],  color = colors[i], linewidth=1.5)

        # Draw the max and min
        ax.plot([p5, p5], [yaxis_s+yaxis_interval/4, yaxis_e-yaxis_interval/4],  color = colors[i], linewidth=1.5)
        ax.plot([p95, p95], [yaxis_s+yaxis_interval/4, yaxis_e-yaxis_interval/4],  color = colors[i], linewidth=1.5)
        ax.plot([p75, p95], [(yaxis_s+yaxis_e)/2, (yaxis_s+yaxis_e)/2], color = colors[i], linewidth=1.5)
        ax.plot([p25, p5], [(yaxis_s+yaxis_e)/2, (yaxis_s+yaxis_e)/2],  color = colors[i], linewidth=1.5)

    ax.legend(fontsize=5, frameon=False, ncol=3)

    if var_name == 'Qle':
        ax.set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
        ax.set_ylim(0,240)
    elif var_name == 'NEP':
        ax.set_ylabel("Net Ecosystem Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
        ax.set_ylim(-0.5,1.4)
    elif var_name == 'GPP':
        ax.set_ylabel("Gross Primary Production (g C m$\mathregular{^{-1}}$ h$\mathregular{^{-1}}$)", fontsize=12)
        ax.set_ylim(0,0.9)

    ax.set_xlabel("VPD (kPa)", fontsize=12)

    ax.set_xticks([0,1,2,3,4,5,6,7])
    ax.set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
    ax.set_xlim(-0.2,7.2)

    ax.tick_params(axis='y', labelsize=12)

    # fig.savefig("./plots/Fig_var_VPD_all_sites_daytime_line_box_coarse.png",bbox_inches='tight',dpi=300) # '_30percent'
    if region_name!=None:
        fig.savefig("./plots/Fig_"+var_name+"_VPD_"+message+"_CMIP6_VPD_"+region_name+".png",bbox_inches='tight',dpi=300) # '_30percent'
    else:
        fig.savefig("./plots/Fig_"+var_name+"_VPD_"+message+"_CMIP6_VPD.png",bbox_inches='tight',dpi=300) # '_30percent'

    return


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # Read site names, IGBP and clim
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    day_time       = True
    method         = 'bin_by_vpd' #'GAM'
    clarify_site   = {'opt': True,
                      'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                      'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}

    error_type     = 'one_std'
    # Smoothing setting

    window_size    = 11
    order          = 3
    smooth_type    = 'no_soomth' #'S-G_filter' #
    turning_point  =  {'calc':True, 'method':'piecewise'}
                      #{'calc':True, 'method':'cdf'}#{'calc':True, 'method':'kneed'}

    region_name='AU'

    # for IGBP_type in IGBP_types:
    var_name = 'Qle'
    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'
    plot_var_VPD_line_box(var_name=var_name, bin_by=bin_by, window_size=window_size, order=order,
             smooth_type=smooth_type, method='bin_by_vpd', model_names=model_names,
             day_time=day_time,  clarify_site=clarify_site, standardize=standardize,
             error_type=error_type,region_name=region_name,
             turning_point=turning_point) # IGBP_type=IGBP_type,

    # for IGBP_type in IGBP_types:
    var_name = 'GPP'
    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'
    plot_var_VPD_line_box(var_name=var_name, bin_by=bin_by, window_size=window_size, order=order,
             smooth_type=smooth_type, method='bin_by_vpd', model_names=model_names,
             day_time=day_time,  clarify_site=clarify_site, standardize=standardize,
             error_type=error_type,region_name=region_name,
             turning_point=turning_point) # IGBP_type=IGBP_type,

    var_name = 'NEE'
    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'
    plot_var_VPD_line_box(var_name=var_name, bin_by=bin_by, window_size=window_size, order=order,
             smooth_type=smooth_type, method='bin_by_vpd', model_names=model_names,
             day_time=day_time,  clarify_site=clarify_site, standardize=standardize,
             error_type=error_type,region_name=region_name,
             turning_point=turning_point) # IGBP_type=IGBP_type,
