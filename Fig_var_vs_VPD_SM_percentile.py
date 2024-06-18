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
import scipy.stats as stats
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

def plot_var_VPD_SM_percentile(var_name=None, day_time=False, energy_cor=False, time_scale=None, country_code=None,
                 selected_by=None, veg_fraction=None, standardize=None, uncertain_type='UCRTN_percentile',
                 method='CRV_bins', IGBP_type=None, clim_type=None, clarify_site={'opt':False,'remove_site':None},
                 num_threshold=200, class_by=None, dist_type=None, calc_correl=False, select_site=None,
                 VPD_num_threshold=200, middle_day=False):

    # ============== convert units =================
    to_ms = False
    if to_ms and var_name == 'Gs':
        # 25 deg C, 10100 Pa
        mol2ms = 8.313*(273.15+25)/10100

    if middle_day:
        message_midday = '_midday'
    else:
        message_midday = ''

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=4, ncols=4, figsize=[16,16],sharex=False, sharey=False, squeeze=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.2)

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
    line_names   = ['θ<15th','15th<θ<30th','30th<θ<50th','50th<θ<70th','70th<θ<90th','θ>90th']
    # ============ Set the input file name ============

    all_bounds   = [[0,15],[15,30],[30,50],[50,70],[70,90],[90,100]]

    file_message = []
    for bounds in all_bounds:
        folder_name, file_message_tmp = decide_filename(day_time=day_time, energy_cor=energy_cor,
                                                    IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                    country_code=country_code, selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                    uncertain_type=uncertain_type, method=method, clarify_site=clarify_site)

        file_message.append(file_message_tmp)

    # Get model names
    if var_name == 'Gs':
        site_names, IGBP_types, clim_types, model_names = load_default_list()
        model_order = model_names['model_select_new']
    else:
        model_order = get_model_out_list(var_name)

    print('model_order',model_order)

    if select_site != None:
        site_info = '_'+select_site
    else:
        site_info = ''


    line_colors = ['red','orange','gold','cyan','royalblue','blue']

    for j, model_out_name in enumerate(model_order):

        # set plot row and col
        row = int(j/4)
        col = j%4

        # set line color
        linestyle = 'solid'

        if dist_type!=None:
            file_names = [  f'./txt/process4_output/{folder_name}/{dist_type}_greater_200_samples/GAM_fit/{var_name}{file_message[0]}_{model_out_name}_{dist_type}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{dist_type}_greater_200_samples/GAM_fit/{var_name}{file_message[1]}_{model_out_name}_{dist_type}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{dist_type}_greater_200_samples/GAM_fit/{var_name}{file_message[2]}_{model_out_name}_{dist_type}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{dist_type}_greater_200_samples/GAM_fit/{var_name}{file_message[3]}_{model_out_name}_{dist_type}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{dist_type}_greater_200_samples/GAM_fit/{var_name}{file_message[4]}_{model_out_name}_{dist_type}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{dist_type}_greater_200_samples/GAM_fit/{var_name}{file_message[5]}_{model_out_name}_{dist_type}{site_info}{message_midday}.csv',]
        else:
            file_names = [  f'./txt/process4_output/{folder_name}/{var_name}{file_message[0]}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{var_name}{file_message[1]}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{var_name}{file_message[2]}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{var_name}{file_message[3]}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{var_name}{file_message[4]}{site_info}{message_midday}.csv',
                            f'./txt/process4_output/{folder_name}/{var_name}{file_message[5]}{site_info}{message_midday}.csv',]

        print(file_names)

        for i, file_name in enumerate(file_names):

            if os.path.exists(file_name):
                # Read lines data
                var = pd.read_csv(file_name)

                # ===== Drawing the lines =====
                # Unify NEE units : upwards CO2 movement is positive values
                if select_site==None and dist_type != None:
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
                else:
                    if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
                        print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
                        value        = var[model_out_name+'_vals']*(-1)
                        vals_bot_tmp = var[model_out_name+'_bot']
                        vals_top_tmp = var[model_out_name+'_top']
                    elif var_name=='Gs' and to_ms:
                        value        = var[model_out_name+'_vals']*mol2ms
                        vals_bot_tmp = var[model_out_name+'_bot']*mol2ms
                        vals_top_tmp = var[model_out_name+'_top']*mol2ms
                    else:
                        if 'SMtop' in var_name:
                            try: 
                                value        = var[model_out_name+'_vals']
                                vals_bot_tmp = var[model_out_name+'_bot']
                                vals_top_tmp = var[model_out_name+'_top']
                            except:
                                value        = var['model_mean_vals']
                                vals_bot_tmp = var['model_mean_bot']
                                vals_top_tmp = var['model_mean_top']
                        else:
                            value        = var[model_out_name+'_vals']
                            vals_bot_tmp = var[model_out_name+'_bot']
                            vals_top_tmp = var[model_out_name+'_top']
                        
                # smooth or not
                if smooth_type != 'no_soomth':
                    value = smooth_vpd_series(value, window_size, order, smooth_type)
                try:
                    var_vpd_series = var['vpd_pred']
                except:
                    var_vpd_series = var['vpd_series']

                vals_bot       = vals_bot_tmp
                vals_top       = vals_top_tmp

                # Plot if the data point > num_threshold
                lw=1.5

                plot = ax[row,col].plot(var_vpd_series, value, lw=lw, color=line_colors[i], linestyle=linestyle,
                                    alpha=1., label=line_names[i])

            else:
                print(file_name,"does not exist")

        ax[row,col].text(0.05, 0.92, order[i], va='bottom', ha='center', rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=14)

        if col == 1 and row== 0:
            ax[row,col].legend(fontsize=9, frameon=False, ncol=2)

        ax[row,col].set_title(change_model_name(model_out_name), fontsize=16)

        ax[row,col].set_xlim(-0.1,7.5)
        ax[row,col].set_xticks([0,1,2,3,4,5,6,7])
        ax[row,col].set_xticklabels(['0','1','2', '3','4','5', '6','7'],fontsize=12)
        if 'SMtop' in var_name:
            ax[row,col].set_ylim(0,0.5)
        else:
            ax[row,col].set_ylim(0,450)

        ax[row,col].tick_params(axis='y', labelsize=12)

    if var_name == 'Qle':
        ax[0,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
        ax[1,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
        ax[2,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
        ax[3,0].set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    elif 'VPD_caused' in var_name:
        ax[0,0].set_ylabel("VPD-driven latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
        ax[1,0].set_ylabel("VPD-driven latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
        ax[2,0].set_ylabel("VPD-driven latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
        ax[3,0].set_ylabel("VPD-driven latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)

    ax[3,0].set_xlabel("VPD (kPa)", fontsize=12)
    ax[3,1].set_xlabel("VPD (kPa)", fontsize=12)
    ax[3,2].set_xlabel("VPD (kPa)", fontsize=12)
    ax[3,3].set_xlabel("VPD (kPa)", fontsize=12)

    message = 'SM_percentile'
    if dist_type !=None:
        message = '_'+dist_type
    else:
        message = ''

    if clim_type !=None:
        message = '_'+clim_type
    else:
        message = ''

    if calc_correl:
        message = '_correl'

    if class_by !=None:
        fig.savefig(f"./plots/Fig_{var_name}_VPD{file_message[0]}_color_{class_by}{message}_gs_eq{site_info}{message_midday}.png",bbox_inches='tight',dpi=300) # '_30percent'
    else:
        fig.savefig(f"./plots/Fig_{var_name}_VPD{file_message[0]}{message}{site_info}{message_midday}.png",bbox_inches='tight',dpi=300) # '_30percent'

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
    turning_point  = {'calc':False, 'method':'piecewise'}
                      #{'calc':True, 'method':'cdf'}#{'calc':True, 'method':'kneed'}

    # for IGBP_type in IGBP_types:
    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'

    # =========== plot_var_VPD_uncertainty ===========
    day_time     = True
    energy_cor   = False
    time_scale   = 'hourly'
    country_code = None
    selected_by  = 'EF_model'
    veg_fraction = None#[0.7,1.]
    standardize  = None
    uncertain_type='UCRTN_bootstrap'
    method       = 'CRV_bins'
    # method       = 'CRV_fit_GAM_complex'
    dist_type    = None#'Poisson' #None #'Linear' #'Poisson' # 'Gamma'

    IGBP_type    = None
    clim_type    = None
    clarify_site = {'opt': True,
                   'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                   'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    var_name     = 'Qle'
    num_threshold= 200

    # ======== Setting Plot EF wet & dry ========
    var_name    = 'Qle'
    class_by    = None  #'gs_eq' #
    calc_correl = False #True
    select_site = None

    # ======== Figure 1: EF wet & dry ========
    if 1:
        selected_by = 'SMtop1m'
        var_name    = 'Qle_VPD_caused'
        middle_day  = False
        plot_var_VPD_SM_percentile(var_name=var_name, day_time=day_time, energy_cor=energy_cor, time_scale=time_scale, country_code=country_code,
                        selected_by=selected_by, veg_fraction=veg_fraction, standardize=standardize, uncertain_type=uncertain_type, method=method,
                        IGBP_type=IGBP_type, clim_type=clim_type, clarify_site=clarify_site,num_threshold=num_threshold, class_by=class_by,
                        dist_type=dist_type, calc_correl=calc_correl, select_site=select_site, middle_day=middle_day)
