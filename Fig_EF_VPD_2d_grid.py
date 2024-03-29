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

def plot_EF_VPD_2d_grid(var_name, model_out_list, day_time=False, uncertain_type='UCRTN_percentile', veg_fraction=None,
                        IGBP_type=None, clim_type=None, country_code=None, message=None, time_scale=None, method=None,
                        clarify_site={'opt':False,'remove_site':None}, standardize=None):

    # ========= Get input file name =========
    folder_name, file_message = decide_filename(day_time=day_time, uncertain_type=uncertain_type, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code, 
                                                veg_fraction=veg_fraction, method=method,
                                                clarify_site=clarify_site)

    # set color cmap
    cmap     = plt.cm.BrBG

    # ========= Read dataset =========
    for i, model_out_name in enumerate(model_out_list):
                                  
        vpd_num      = pd.read_csv(f'./txt/process4_output/{folder_name}/{var_name}_Numbers{file_message}_{model_out_name}.csv',
                                   header=None, dtype=float, na_values=['nan'], sep=' ').to_numpy() #
        var_vals     = pd.read_csv(f'./txt/process4_output/{folder_name}/{var_name}_Values{file_message}_{model_out_name}.csv',
                                   header=None, dtype=float, na_values=['nan'], sep=' ').to_numpy() # na_values=['nan']
        var_vals_top = pd.read_csv(f'./txt/process4_output/{folder_name}/{var_name}_Top_bounds{file_message}_{model_out_name}.csv',
                                   header=None, dtype=float, na_values=['nan'], sep=' ').to_numpy()
        var_vals_bot = pd.read_csv(f'./txt/process4_output/{folder_name}/{var_name}_Bot_bounds{file_message}_{model_out_name}.csv',
                                   header=None, dtype=float, na_values=['nan'], sep=' ').to_numpy()

        # Set plots
        fig, ax  = plt.subplots(nrows=2, ncols=2, figsize=[9,9],sharex=False, sharey=False, squeeze=True) #

        # plt.subplots_adjust(wspace=0.09, hspace=0.02)

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

        # Set up vpd and EF bounds
        vpd_top     = 7.1
        vpd_bot     = 0.1

        EF_top      = 1.025
        EF_bot      = 0.025

        extent= (EF_bot, EF_top, vpd_bot, vpd_top)

        # plot values 2D plot, X axis is EF and Y axis is VPD
        plot1 = ax[0,0].imshow(var_vals[:,:], origin="lower", extent=extent, vmin=0, vmax=2, interpolation="none", cmap=cmap, aspect="auto") #  vmin=min(var_vals), vmax=max(var_vals),
        cbar  = plt.colorbar(plot1, ax=ax[0,0], ticklocation="right", pad=0.13, orientation="horizontal",
                        aspect=20, shrink=1.) # cax=cax,

        # plot numbers 2D plot
        plot2 = ax[0,1].imshow(vpd_num, origin="lower", extent=extent, vmin=0, vmax=10000, interpolation="none", cmap=cmap, aspect="auto") # vmin=0.5, vmax=29.5,  resample=False,
        cbar  = plt.colorbar(plot2, ax=ax[0,1], ticklocation="right", pad=0.13, orientation="horizontal",
                        aspect=20, shrink=1.) # cax=cax,

        # plot top bound & low bound 2D plot
        plot3 = ax[1,0].imshow(var_vals_top, origin="lower", extent=extent, vmin=0, vmax=2, interpolation="none", cmap=cmap, aspect="auto") # vmin=0.5, vmax=29.5,  resample=False,
        cbar  = plt.colorbar(plot3, ax=ax[1,0], ticklocation="right", pad=0.13, orientation="horizontal",
                        aspect=20, shrink=1.) # cax=cax,
        plot4 = ax[1,1].imshow(var_vals_bot, origin="lower", extent=extent, vmin=0, vmax=2, interpolation="none", cmap=cmap, aspect="auto") # vmin=0.5, vmax=29.5,  resample=False,
        cbar  = plt.colorbar(plot4, ax=ax[1,1], ticklocation="right", pad=0.13, orientation="horizontal",
                        aspect=20, shrink=1.) # cax=cax,

        # ax[0,0].text(0.02, 0.15,'LH',  fontsize=14, verticalalignment='top', bbox=props)
        # ax[0,1].text(0.02, 0.15,'numbers', fontsize=14, verticalalignment='top', bbox=props)
        # ax[1,0].text(0.02, 0.15,'LH_top', fontsize=14, verticalalignment='top', bbox=props)
        # ax[1,1].text(0.02, 0.15,'LH_bot',  fontsize=14, verticalalignment='top', bbox=props)

        ax[0,0].set_xlim(EF_bot, EF_top)
        ax[0,0].set_ylim(vpd_bot,  vpd_top)

        ax[0,1].set_xlim(EF_bot, EF_top)
        ax[0,1].set_ylim(vpd_bot,  vpd_top)

        ax[1,0].set_xlim(EF_bot, EF_top)
        ax[1,0].set_ylim(vpd_bot,  vpd_top)

        ax[1,1].set_xlim(EF_bot, EF_top)
        ax[1,1].set_ylim(vpd_bot,  vpd_top)
        # ax.set_xticks(ticks=np.arange(len(case_names)), labels=case_names)

        # save figures
        if message != None:
            fig.savefig(f"./plots/Fig_2d_grid_{var_name}_{message}_{model_out_name}.png",bbox_inches='tight',dpi=300)
        else:
            fig.savefig(f"./plots/Fig_2d_grid_{var_name}{file_message}_{model_out_name}.png",bbox_inches='tight',dpi=300)
            
    return


if __name__ == "__main__":

    # ================ Setting ================
    var_name       = 'Qle'  #'TVeg'

    # Read site names, IGBP and clim
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    day_time       = False
    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    uncertain_type = 'UCRTN_one_std'# 'UCRTN_bootstrap'
                    # 'UCRTN_percentile'
                    # 'UCRTN_one_std'
    standardize    = None       # 'None'
                                # 'STD_LAI'
                                # 'STD_annual_obs'
                                # 'STD_monthly_obs'
                                # 'STD_monthly_model'
                                # 'STD_daily_obs'
    time_scale     = 'daily'
    method         = 'CRV_bins' # 'CRV_bins'
                                # 'CRV_fit_GAM'

    # ================ Getting model name list ================
    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # Get model lists
    model_out_list = get_model_out_list(var_name)

    # message = "by_daily_obs_mean"

    # ================ Plotting ================
    plot_EF_VPD_2d_grid(var_name, model_out_list, day_time=day_time, uncertain_type=uncertain_type,
                        time_scale=time_scale, method=method,
                        clarify_site=clarify_site, standardize=standardize)