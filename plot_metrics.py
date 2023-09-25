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
from plot_script import *
from PLUMBER2_VPD_common_utils import *

def calc_derivative(model_out_list, vpd_series, vals_smooth):

    nmodel          = len(model_out_list)
    vpd_series_len  = len(vpd_series)
    vpd_interval    = vpd_series[1]-vpd_series[0]
    derivative_vals = np.full([nmodel,vpd_series_len], np.nan)
    derivative      = {}

    for i,model_out_name in enumerate(model_out_list):
        for j in np.arange(1,vpd_series_len-1):
            derivative_vals[i,j] = (vals_smooth[i,j+1]-vals_smooth[i,j-1])/(2*vpd_interval)
        derivative[model_out_name] = derivative_vals[i,:]

    return derivative

def find_peak(model_out_list, vpd_series, derivative):

    '''
    output: peak_values[nmodel]
    '''

    nmodel          = len(model_out_list)
    vpd_series_len  = len(vpd_series)
    # peak_values     = np.full(nmodel, np.nan)

    peak_values     = np.full([nmodel,vpd_series_len], np.nan)
    tmp             = np.full(vpd_series_len, np.nan)

    for i,model_out_name in enumerate(model_out_list):
        derivative_tmp = derivative[model_out_name]
        for j in np.arange(0,vpd_series_len-1):
            # if they are not the beginning and ending points
            if (~ np.isnan(derivative_tmp[j])) and (~ np.isnan(derivative_tmp[j+1])):
                # if it is maximum
                if derivative_tmp[j] > 0 and derivative_tmp[j+1] < 0:
                    if abs(derivative_tmp[j]) < abs(derivative_tmp[j+1]):
                        tmp[j] = vpd_series[j]
                    else:
                        tmp[j] = vpd_series[j+1]
                # if it is minimum
            elif derivative_tmp[j] < 0 and derivative_tmp[j+1] > 0:
                    if abs(derivative_tmp[j]) < abs(derivative_tmp[j+1]):
                        tmp[j] = vpd_series[j]*(-1)
                    else:
                        tmp[j] = vpd_series[j+1]*(-1)
        peak_values[i,:] = tmp

        print('np.unique(peak_values[i,:])',np.unique(peak_values[i,:]))
    return peak_values

# def calc_slope():

def single_plot_lines(model_out_list, x_values, y_values ,message=None):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[8,6],sharex=True, sharey=False, squeeze=True) #

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

        line_color = model_colors[model_out_name]
        plot       = ax.plot(x_values, y_values[model_out_name], lw=2.0, color=line_color, alpha=0.9, label=model_out_name)
        plot       = ax.axhline(y=0.0, color='black', linestyle='-.', linewidth=1)
    ax.legend(fontsize=6, frameon=False, ncol=3)

    fig.savefig("./plots/check_single_plot_lines_"+message,bbox_inches='tight',dpi=300) # '_30percent'
#
# def plot_metrics(var_plot,low_bound,):
#
#
#     # ============ Setting for plotting ============
#     cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r
#
#     fig, ax  = plt.subplots(nrows=2, ncols=2, figsize=[15,10],sharex=True, sharey=False, squeeze=True) #
#     # fig, ax = plt.subplots(figsize=[10, 7])
#     # plt.subplots_adjust(wspace=0.0, hspace=0.0)
#
#     plt.rcParams['text.usetex']     = False
#     plt.rcParams['font.family']     = "sans-serif"
#     plt.rcParams['font.serif']      = "Helvetica"
#     plt.rcParams['axes.linewidth']  = 1.5
#     plt.rcParams['axes.labelsize']  = 14
#     plt.rcParams['font.size']       = 14
#     plt.rcParams['legend.fontsize'] = 14
#     plt.rcParams['xtick.labelsize'] = 14
#     plt.rcParams['ytick.labelsize'] = 14
#
#     almost_black = '#262626'
#     # change the tick colors also to the almost black
#     plt.rcParams['ytick.color']     = almost_black
#     plt.rcParams['xtick.color']     = almost_black
#
#     # change the text colors also to the almost black
#     plt.rcParams['text.color']      = almost_black
#
#     # Change the default axis colors from black to a slightly lighter black,
#     # and a little thinner (0.5 instead of 1)
#     plt.rcParams['axes.edgecolor']  = almost_black
#     plt.rcParams['axes.labelcolor'] = almost_black
#
#     # Set the colors for different models
#     model_colors = set_model_colors()
#
#     props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
#
#     # Plot the PDF of the normal distribution
#     # hist = ax[0,0].hist(var_output_dry['VPD'], bins=400, density=False, alpha=0.6, color='g', histtype='stepfilled')
#     # hist = ax[0,1].hist(var_output_wet['VPD'], bins=400, density=False, alpha=0.6, color='g', histtype='stepfilled')
#     # Get the histogram data
#
#     ax[0,0].bar(var_dry['vpd_series'], var_dry['vpd_num'])
#     ax[0,1].bar(var_wet['vpd_series'], var_wet['vpd_num'])
#
#     for i, model_out_name in enumerate(model_out_list):
#
#         line_color = model_colors[model_out_name] #plt.cm.tab20(i / len(model_out_list))
#
#         plot = ax[1,0].plot(var_dry['vpd_series'], var_dry[model_out_name+'_vals'], lw=2.0,
#                             color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
#
#         fill = ax[1,0].fill_between(var_dry['vpd_series'],
#                                     var_dry[model_out_name+'_bot'],
#                                     var_dry[model_out_name+'_top'],
#                                     color=line_color, edgecolor="none",
#                                     alpha=0.05) #  .rolling(window=10).mean()
#
#         plot = ax[1,1].plot(var_wet['vpd_series'], var_wet[model_out_name+'_vals'], lw=2.0,
#                             color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
#
#         fill = ax[1,1].fill_between(var_wet['vpd_series'],
#                                     var_wet[model_out_name+'_bot'],
#                                     var_wet[model_out_name+'_top'],
#                                     color=line_color, edgecolor="none",
#                                     alpha=0.05) #  .rolling(window=10).mean()
#
#     ax[1,0].legend(fontsize=6, frameon=False, ncol=3)
#
#     if IGBP_type !=None:
#         ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)
#     if clim_type !=None:
#         ax[1,0].text(0.12, 0.92, 'Clim_type='+clim_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)
#
#     ax[1,0].text(0.12, 0.87, 'site_num='+str(var_dry['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)
#
#     ax[0,0].set_xlim(0, 7.)
#     ax[0,1].set_xlim(0, 7.)
#
#     ax[0,0].set_ylim(0, 500)
#     ax[0,1].set_ylim(0, 500)
#
#     if var_name == 'TVeg':
#         ax[1,0].set_ylim(-0.1, 0.5)
#         ax[1,1].set_ylim(-0.1, 0.5)
#     if var_name == 'Qle':
#         ax[1,0].set_ylim(-50, 400)
#         ax[1,1].set_ylim(-50, 400)
#     if var_name == 'NEE':
#         ax[1,0].set_ylim(-1, 1)
#         ax[1,1].set_ylim(-1, 1)
#
#     # ax[1].set_xlabel('VPD (kPa)', loc='center',size=14)# rotation=270,
#     fig.savefig("./plots/30percent/"+var_name+'_VPD_all_sites'+message,bbox_inches='tight',dpi=300) # '_30percent'

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    # site_names     = ["AU-Tum"]

    print(site_names)

    var_name       = 'TVeg'  #'TVeg'
    bin_by         = 'EF_obs' #'EF_model' #'EF_obs'#
    IGBP_types     = ['CRO']#, 'CSH', 'DBF', 'EBF','EBF', 'ENF', 'GRA', 'MF', 'OSH', 'WET', 'WSA', 'SAV']
    clim_types     = ['Af', 'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Csa', 'Csb', 'Cwa',
                      'Dfa', 'Dfb', 'Dfc', 'Dsb', 'Dsc', 'Dwa', 'Dwb', 'ET']

    day_time       = True
    energy_cor     = True
    low_bound      = 30
    high_bound     = 70

    if var_name == 'NEE':
        energy_cor     = False

    peak_values = {}

    for IGBP_type in IGBP_types:

    # ============== read data ==============
        message      = ''
        EF_threshold = 30

        if day_time:
            message  = message + '_daytime'

        try:
            message  = message + '_IGBP='+IGBP_type
        except:
            print(' ')

        try:
            message  = message + '_clim='+clim_type
        except:
            print(' ')

        # ================= Read in csv file =================
        var_bin_by_VPD = pd.read_csv(f'./txt/{var_name}_VPD'+message+'_EF_'+str(low_bound)+'th.csv')
        print('var_bin_by_VPD',var_bin_by_VPD)

        # Get model namelists
        model_out_list = []
        for column_name in var_bin_by_VPD.columns:
            if "_vals" in column_name:
                model_out_list.append(column_name.split("_vals")[0])

        vpd_series      = var_bin_by_VPD['vpd_series']

        # ================= Smoothing =================
        window_size = 11
        order       = 3
        type        = 'S-G_filter'
        
        vals_smooth = smooth_vpd_series(var_bin_by_VPD, window_size=window_size, order=order, type=type)

        # ================= Calc derivative =================
        derivative = calc_derivative(model_out_list, vpd_series, vals_smooth)

        single_plot_lines(model_out_list, x_values=vpd_series, y_values=derivative, message=var_name+'_derivative_VPD'+message+'_EF_'+str(low_bound)+'th')

        find_peak(model_out_list, vpd_series, derivative) ## peak_values[IGBP_type] =
