'''
Check data distirbution:
Including:

'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *

def plot_pdf(file_name, model_list, plot_type='fitting_line', density=False, message=None):

    '''
    Plot different pdf
    '''

    # Setting plots
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

    # create figure
    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[10,10],sharex=True, sharey=False, squeeze=True)
    var      = pd.read_csv(file_name)

    for model_name in model_list:

        if model_name != 'obs':
            varname  = 'model_'+model_name
        else:
            varname  = model_name

        var_vals = var[varname]
        # remove nan values
        var_vals = np.sort(var_vals[~ np.isnan(var_vals)])

        if plot_type == 'fitting_line':
            # Plot the PDF of the normal distribution
            # read the data for this model

            if np.any(var_vals):

                # bandwidth = 0.5
                # Estimate the probability density function using kernel density estimation.
                pdf       = gaussian_kde(var_vals)#, bw_method=bandwidth)
                # Plot the probability density function.
                ax.plot(var_vals, pdf(var_vals), color=model_colors[model_name],label=model_name)

        if plot_type == 'hist':

            hist = ax.hist(var_vals, bins=1000, density=density, alpha=0.6, color=model_colors[model_name],
                        label=model_name, histtype='step')#'stepfilled')

            ax.set_xlim(-100,500)
            ax.set_ylim(0,5000)

        ax.legend(fontsize=8,frameon=False)

        fig.savefig(f"./plots/PDF_{message}.png",bbox_inches='tight',dpi=300)

    return

# def check_gamma_dist(file_name, model_list):

#     # column_name= 'time'
#     var_output = pd.read_csv(file_name)

#     model_list = []
#     for column_name in var_output.columns:
#         if "_EF" in column_name:
#             model_list.append(column_name.split("_EF")[0])

#     print('model_list',model_list)

#     for model_in in model_list:
#         print(model_in,' ', np.nanmean(var_output[model_in+'_EF']))

#     return

if __name__ == "__main__":

    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list = model_names['model_select_new']
    plot_type  = 'hist' #'fitting_line'
    density    = False

    # file_name  = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/raw_data_TVeg_VPD_hourly_RM16_DT_EF_model_0-0.2.csv'
    # message    = 'TVeg_VPD_hourly_RM16_DT_EF_model_0-0.2'

    EF_ranges  = ['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0']
    # model_list = model_names['model_select_new']
    model_list = model_names['model_tveg']
    var_name   = 'TVeg'

    for EF_range in EF_ranges:
        file_name  = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/raw_data_'+var_name+'_VPD_hourly_RM16_DT_EF_model_'+EF_range+'.csv'
        message    = var_name+'_VPD_hourly_RM16_DT_EF_model_'+EF_range

        plot_pdf(file_name, model_list, plot_type=plot_type, message=message)
