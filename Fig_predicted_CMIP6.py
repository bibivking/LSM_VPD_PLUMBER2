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

def plot_predicted_CMIP6_boxplot(CMIP6_txt_path, var_name, model_list):

    input_files = [f'{CMIP6_txt_path}/metrics_predicted_{var_name}_historical_global.csv',
                    f'{CMIP6_txt_path}/metrics_predicted_{var_name}_ssp245_global.csv',    ]
    labels      = ['hist_glb', 'ssp245_glb']

    metrics_hist_glb   = pd.read_csv(input_files[0], na_values=[''])
    metrics_ssp245_glb = pd.read_csv(input_files[1], na_values=[''])

    # ============ Setting for plotting ============
    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[5,5],sharex=False, sharey=False, squeeze=False)

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
    model_list   = model_list.append('CMIP6')

    # ============ Set the input file name ============
    for i, model_out_name in enumerate(model_list):

        # set line color
        line_color = model_colors[model_out_name]

        # ============= Plot hist =============
        xaxis_s = 0.5+i*3
        xaxis_e = 1.5+i*3

        # Draw the box
        ax.add_patch(Polygon([[xaxis_s, metrics1.loc[i, 'p25']], [xaxis_s, metrics1.loc[i, 'p75']],
                              [xaxis_e, metrics1.loc[i, 'p75']], [xaxis_e, metrics1.loc[i, 'p25']]],
                              closed=True, color=line_color, fill=True, alpha=0.8, linewidth=0.1))

        # Draw the metrics1.loc[i, 'median'] line
        ax.plot([xaxis_s,xaxis_e], [metrics1.loc[i, 'median'],metrics1.loc[i, 'median']], color = almost_black, linewidth=0.5)

        # Draw the metrics1.loc[i, 'p25'] metrics1.loc[i, 'p75']
        ax.plot([xaxis_s, xaxis_e], [metrics1.loc[i, 'p25'], metrics1.loc[i, 'p25']], color = almost_black, linewidth=0.5)
        ax.plot([xaxis_s, xaxis_e], [metrics1.loc[i, 'p75'], metrics1.loc[i, 'p75']], color = almost_black, linewidth=0.5)

        ax.plot([xaxis_s, xaxis_s], [metrics1.loc[i, 'p25'], metrics1.loc[i, 'p75']], color = almost_black, linewidth=0.5)
        ax.plot([xaxis_e, xaxis_e], [metrics1.loc[i, 'p25'], metrics1.loc[i, 'p75']], color = almost_black, linewidth=0.5)

        # Draw the max and min
        ax.plot([xaxis_s+0.1, xaxis_e-0.1], [metrics1.loc[i, 'min'], metrics1.loc[i, 'min']], color = almost_black, linewidth=0.5)
        ax.plot([xaxis_s+0.1, xaxis_e-0.1], [metrics1.loc[i, 'max'], metrics1.loc[i, 'max']], color = almost_black, linewidth=0.5)
        ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [metrics1.loc[i, 'p75'], metrics1.loc[i, 'max']], color = almost_black, linewidth=0.5)
        ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [metrics1.loc[i, 'p25'], metrics1.loc[i, 'min']], color = almost_black, linewidth=0.5)

        # ============= Plot hist =============
        xaxis_s = 1.5+i*3
        xaxis_e = 2.5+i*3

        # Draw the box
        ax.add_patch(Polygon([[xaxis_s, metrics2.loc[i, 'p25']], [xaxis_s, metrics2.loc[i, 'p75']],
                              [xaxis_e, metrics2.loc[i, 'p75']], [xaxis_e, metrics2.loc[i, 'p25']]],
                              closed=True, color=line_color, fill=True, alpha=0.5, linewidth=0.1))

        # Draw the metrics2.loc[i, 'median'] line
        ax.plot([xaxis_s,xaxis_e], [metrics2.loc[i, 'median'],metrics2.loc[i, 'median']], color = almost_black, linewidth=0.5)

        # Draw the metrics2.loc[i, 'p25'] metrics2.loc[i, 'p75']
        ax.plot([xaxis_s, xaxis_e], [metrics2.loc[i, 'p25'], metrics2.loc[i, 'p25']], color = almost_black, linewidth=0.5)
        ax.plot([xaxis_s, xaxis_e], [metrics2.loc[i, 'p75'], metrics2.loc[i, 'p75']], color = almost_black, linewidth=0.5)

        ax.plot([xaxis_s, xaxis_s], [metrics2.loc[i, 'p25'], metrics2.loc[i, 'p75']], color = almost_black, linewidth=0.5)
        ax.plot([xaxis_e, xaxis_e], [metrics2.loc[i, 'p25'], metrics2.loc[i, 'p75']], color = almost_black, linewidth=0.5)

        # Draw the max and min
        ax.plot([xaxis_s+0.1, xaxis_e-0.1], [metrics2.loc[i, 'min'], metrics2.loc[i, 'min']], color = almost_black, linewidth=0.5)
        ax.plot([xaxis_s+0.1, xaxis_e-0.1], [metrics2.loc[i, 'max'], metrics2.loc[i, 'max']], color = almost_black, linewidth=0.5)
        ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [metrics2.loc[i, 'p75'], metrics2.loc[i, 'max']], color = almost_black, linewidth=0.5)
        ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [metrics2.loc[i, 'p25'], metrics2.loc[i, 'min']], color = almost_black, linewidth=0.5)

    ax.legend(fontsize=7, frameon=False, ncol=2)

    # ax.set_title("Wet (EF>0.8)", fontsize=20)
    ax.set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    ax.set_xlabel("models", fontsize=12)

    ax.set_xticks(np.arange(1.5,len(model_list)*3,3))
    ax.set_xticklabels(model_list,fontsize=12)

    ax.tick_params(axis='y', labelsize=12)

    fig.savefig(f"./plots/plot_predicted_CMIP6_boxplot.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

if __name__ == "__main__":

    # Get model lists
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list = model_names['model_select']
    plot_predicted_CMIP6_boxplot(CMIP6_txt_path, var_name, model_list)
