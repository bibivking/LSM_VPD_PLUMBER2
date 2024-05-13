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
from PLUMBER2_VPD_common_utils import *

def plot_2d_grid(var_name, Xvar_name, Yvar_name, model_out_list, day_time=False, uncertain_type='UCRTN_percentile', 
                 energy_cor=False,veg_fraction=None, IGBP_type=None, clim_type=None, selected_by=None, 
                 country_code=None, time_scale=None, bounds=None, LAI_range=None, method=None, 
                 clarify_site={'opt':False,'remove_site':None}, standardize=None, humidity_type='specific_humidity'):

    # ========= Get input file name =========

    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor,
                        IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, 
                        standardize=standardize, country_code=country_code, selected_by=selected_by, 
                        bounds=bounds, veg_fraction=veg_fraction, uncertain_type=uncertain_type, 
                        method=method, LAI_range=LAI_range, clarify_site=clarify_site)
    # set color cmap
    cmap     = plt.cm.gist_earth_r

    # ========= Read dataset =========

    # Set plots
    fig, ax  = plt.subplots(nrows=4, ncols=4, figsize=[20,12],sharex=False, sharey=False, squeeze=False) #

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
    if Xvar_name == 'VPD':
        x_top      = 10.05 #7.04
        x_bot      = 0.05 #0.02
        x_interval = 0.1 #0.04
    if Yvar_name == 'obs_Tair':
        y_top      = 51 #7.04
        y_bot      = -20 #0.02
        y_interval = 1. 

    # Make x and y series
    x_series   = np.arange(x_bot, x_top, x_interval)
    y_series   = np.arange(y_bot, y_top, y_interval)

    extent  = (x_bot, x_top, y_bot, y_top)

    if humidity_type == 'relative_humidity':
        # set relative humidity levels
        rhs     = [0.01,10,20,30,40,50,60,70,80,90,99.9]
        VPD_rh  = np.zeros((11,len(y_series)))

    elif humidity_type == 'specific_humidity':
        # set specific humidity 
        rhs     = [0.0001,0.005,0.01,0.015,0.02,0.025,0.03]
        VPD_rh  = np.zeros((len(rhs),len(y_series)))

    for i, rh in enumerate(rhs):
        for j, y_val in enumerate(y_series):
            y_val       = y_val+273.15 # C to K 

            if humidity_type == 'relative_humidity':
                VPD_rh[i,j] = calculate_VPD_by_RH(rh, y_val)

            elif humidity_type == 'specific_humidity':
                press       = 100000. # Pa
                VPD_rh[i,j] = calculate_VPD_by_Qair(rh, y_val, press)

    order = ['(a)','(b)','(c)','(d)','(e)',
             '(f)','(g)','(h)','(i)','(j)',
             '(k)','(l)','(m)','(n)','(o)',
             '(p)','(q)','(r)','(s)','(t)',]


    if bounds[0] == 0:
        val_min = 0
        val_max = 100
    elif bounds[0] == 0.2:
        val_min = 0
        val_max = 200
    elif bounds[0] == 0.4:
        val_min = 0
        val_max = 300
    elif bounds[0] == 0.6:
        val_min = 0
        val_max = 400
    elif bounds[0] == 0.8:
        val_min = 0
        val_max = 500

    for i, model_in in enumerate(model_out_list):

        # set plot row and col
        row = int(i/4)
        col = i%4

        input_file   = f'./txt/process4_output/2d_grids/{folder_name}/val_{var_name}_{Xvar_name}_{Yvar_name}{file_message}_{model_in}.csv'
        # print(input_file)

        if os.path.exists(input_file):
            # print(input_file, 'exist')
            var_vals = pd.read_csv(input_file, header=None, dtype=float, na_values=['nan'], sep=' ').to_numpy() # na_values=['nan']
            print(np.shape(var_vals))

            # plot values 2D plot, X axis is EF and Y axis is VPD
            plot1 = ax[row,col].imshow(var_vals[:,:].transpose(), origin="lower", extent=extent, vmin=val_min, vmax=val_max, 
                                       interpolation="none", cmap=cmap, aspect="auto") #  vmin=min(var_vals), vmax=max(var_vals),
            # cbar  = plt.colorbar(plot1, ax=ax[row,col], ticklocation="right", pad=0.13, orientation="horizontal",
            #                 aspect=20, shrink=1.) # cax=cax,

            for j, rh in enumerate(rhs):
                if humidity_type == 'relative_humidity':
                    lw = 0.1+(rh/100)
                elif humidity_type == 'specific_humidity':
                    lw = 0.1+(rh*20)
                plot2 = ax[row,col].plot( VPD_rh[j,:], y_series, ls='dashed', c='black',lw=lw)

            ax[row,col].set_xlim(0, 8)
            ax[row,col].set_ylim(0, 45)
            ax[row,col].text(0.95, 0.05, order[i]+" "+change_model_name(model_in), va='bottom', ha='right', rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=14)

    location = fig.add_axes([0.13, 0.07, 0.76, 0.016] ) # [left, bottom, width, height]
    fig.colorbar(plot1, ax=ax[:,:], pad=0.12, cax=location,
                 orientation="horizontal", aspect=60, shrink=1.)

    # save figures 
    fig.savefig(f"./plots/Fig_2d_grid_{var_name}_{Xvar_name}_{Yvar_name}{file_message}_{model_in}.png",bbox_inches='tight',dpi=300)

    return


if __name__ == "__main__":

    # ======================= Default setting (dont change) =======================
    var_name       = 'Qle'       #'TVeg'
    time_scale     = 'hourly'   #'daily'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'
    method         = 'CRV_bins' # 'CRV_bins'
                                # 'CRV_fit_GAM_simple'
                                # 'CRV_fit_GAM_complex'
    standardize    = None       # 'None'
                                # 'STD_LAI'
                                # 'STD_annual_obs'
                                # 'STD_monthly_obs'
                                # 'STD_monthly_model'
                                # 'STD_daily_obs'
    LAI_range      = None
    veg_fraction   = None   #[0.7,1]

    clarify_site      = {'opt': True,
                         'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                         'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

    day_time       = False  # False for daily
                            # True for half-hour or hourly

    if time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # ====================== Custom setting ========================
    var_name         = 'Qle'
    selected_by      = 'EF_model'
    method           = 'CRV_bins'

    Xvar_name        = 'VPD' #'obs_SWdown' # 'obs_Tair'# units: K #'VPD' # 
    Yvar_name        = 'obs_Tair'
    uncertain_type   = 'UCRTN_one_std'
                     # 'UCRTN_percentile'
                     # 'UCRTN_one_std'
    
    EF_bounds        = [0,0.2] 

    # ================ Getting model name list ================
    # Get model lists
    model_out_list = get_model_out_list(var_name)

    # ================ Plotting ================
    EF_bounds_all  = [[0,  0.2],
                      [0.2,0.4],
                      [0.4,0.6],
                      [0.6,0.8],
                      [0.8,1.] ]
    
    for EF_bounds in EF_bounds_all:
        plot_2d_grid(var_name, Xvar_name, Yvar_name, model_out_list, day_time=day_time, 
                    uncertain_type=uncertain_type, time_scale=time_scale, method=method, 
                    selected_by = selected_by, energy_cor=energy_cor,
                    bounds = EF_bounds, clarify_site=clarify_site, standardize=standardize)
