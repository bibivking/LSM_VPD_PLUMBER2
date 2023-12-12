import os
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *

def plot_diurnal_cycle(site_name,var_output,model_out_list):

    # check the diurnal cycle
    var_diurnal_cycle = var_output.groupby(['hour']).mean()

    fig, ax = plt.subplots(figsize=[10, 7])

    # set the colors for different models
    model_colors = set_model_colors()

    for i, model_out_name in enumerate(model_out_list):
        line_color = model_colors[model_out_name]#plt.cm.tab20(i / len(model_out_list))
        sct = ax.plot(var_diurnal_cycle[model_out_name], lw=2.0,
                        color=line_color, alpha=0.9, label=model_out_name)

    if var_name == 'trans':
        ax.set_ylabel('Transpiration (mm h$\mathregular{^{-1}}$)', loc='center',size=14)# rotation=270,
    if var_name == 'latent':
        ax.set_ylabel('Latent heat (W m$\mathregular{^{-2}}$)', loc='center',size=14)# rotation=270,

    ax.legend(fontsize=8,frameon=False)
    fig.savefig("./plots/diurnal_cycle_"+var_name+"_"+site_name,bbox_inches='tight',dpi=300)

def plot_pdf(var_output, model_out_list, message=None, plot_type='fitting_line',density=False):

    # Plotting
    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[10,10],sharex=True, sharey=False, squeeze=True)

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

    for model_name in model_out_list:
        # set data
        var_name_plot = model_name#+'_latent'
        var_vals      = var_output[var_name_plot]
        notNan_mask   = ~ np.isnan(var_vals)
        var_vals      = np.sort(var_vals[notNan_mask])

        # Plot the PDF of the normal distribution
        if np.any(var_vals):

            if plot_type == 'fitting_line':
                # Estimate the probability density function using kernel density estimation.
                bandwidth = 0.5
                pdf       = gaussian_kde(var_vals, bw_method=bandwidth)

                # Plot the probability density function.
                ax.plot(var_vals, pdf(var_vals),
                        color=model_colors[model_name],label=model_name)
            if plot_type == 'hist':
                hist = ax.hist(var_output[model_name+'_EF'], bins=100, density=density, alpha=0.6, color=model_colors[model_name],
                            label=model_name, histtype='stepfilled')

    ax.legend(fontsize=8,frameon=False)
    if message == None:
        fig.savefig("./plots/"+var_name+'_PDF_all_sites',bbox_inches='tight',dpi=300)
    else:
        fig.savefig("./plots/"+var_name+'_PDF_all_sites'+message,bbox_inches='tight',dpi=300)

    return

def plot_scatter(var_output, model_out_list, message=None):

    # Plotting
    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[8,6],sharex=True, sharey=False, squeeze=True)

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

    for model_name in model_out_list:
        # set data
        var_name_plot = model_name#+'_latent'
        var_vals      = var_output[var_name_plot]
        notNan_mask   = ~ np.isnan(var_vals)
        var_vals      = np.sort(var_vals[notNan_mask])

        # Plot the PDF of the normal distribution
        if np.any(var_vals):

            if plot_type == 'fitting_line':
                # Estimate the probability density function using kernel density estimation.
                bandwidth = 0.5
                pdf       = gaussian_kde(var_vals, bw_method=bandwidth)

                # Plot the probability density function.
                ax.plot(var_vals, pdf(var_vals),
                        color=model_colors[model_name],label=model_name)
            if plot_type == 'hist':
                hist = ax.hist(var_output[model_name+'_EF'], bins=100, density=density, alpha=0.6, color=model_colors[model_name],
                            label=model_name, histtype='stepfilled')

    ax.legend(fontsize=8,frameon=False)
    if message == None:
        fig.savefig("./plots/"+var_name+'_PDF_all_sites',bbox_inches='tight',dpi=300)
    else:
        fig.savefig("./plots/"+var_name+'_PDF_all_sites'+message,bbox_inches='tight',dpi=300)

    return

def plot_lines(var_output, model_out_list, message=None):

    # ============ Setting for plotting ============
    cmap     = plt.cm.BrBG #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[8,7], squeeze=True) #
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

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    for i, model_out_name in enumerate(model_out_list):
        print('model_out_name',model_out_name)
        line_color = model_colors[model_out_name] #plt.cm.tab20(i / len(model_out_list))

        vals = var_output[model_out_name+'_vals']

        above_200      = (var_output[model_out_name+'_vpd_num']>200)
        var_vpd_series = var_output['vpd_series'][above_200]
        vals           = vals[above_200]
        vals_bot       = var_output[model_out_name+'_bot'][above_200]
        vals_top       = var_output[model_out_name+'_top'][above_200]

        print('vals_bot',vals_bot)
        print('vals_top',vals_top)
        # start plotting
        if np.sum(var_output[model_out_name+'_vpd_num']-200) > 0:
            # ax.plot(var_output['vpd_series'], var_output[model_out_name+'_vpd_num'], lw=2.0, color=line_color, alpha=0.7,label=model_out_name)
            # ax.axhline(y=200, color='black', linestyle='-.', linewidth=1)
            plot = ax.plot(var_vpd_series, vals, lw=0.2,
                                color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

            fill = ax.fill_between(var_vpd_series,
                                        vals_bot,
                                        vals_top,
                                        color=line_color, edgecolor="none",
                                        alpha=0.5) #  .rolling(window=10).mean()

    ax.legend(fontsize=6, frameon=False, ncol=3)
    ax.text(0.12, 0.87, 'site_num='+str(var_output['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax.transAxes, fontsize=12)

    ax.set_xlim(0, 7.)

    fig.savefig("./plots/"+var_name+'_'+message+'_plot_lines.png',bbox_inches='tight',dpi=300) # '_30percent'

    return

if __name__ == "__main__":


    var_name  = 'Qle'
    plot_type = 'hist'#'fitting_line'
    density   = True

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # get the model namelist
    f             = nc.Dataset("/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/AR-SLu.nc", mode='r')
    model_in_list = f.variables[var_name + '_models']
    ntime         = len(f.variables['CABLE_time'])
    model_out_list= []

    for model_in in model_in_list:
        if len(f.variables[f"{model_in}_time"]) == ntime:
            model_out_list.append(model_in)

    # get the model namelist
    model_order     = []
    model_names_all = model_names['model_select']
    for model_name in model_names_all:
        if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
            model_order.append(model_name)

    # Add obs
    if var_name in ['Qle','NEE','GPP','NEP']:
        model_order.append('obs')
    print('model_order',model_order)

    # # Reading data
    # var_output    = pd.read_csv(f'./txt/{var_name}_all_sites.csv')
    #
    # # plot_pdf(var_output, model_out_list, plot_type=plot_type, density=density)
    # message = ''
    # plot_scatter(var_output, model_out_list, message=message)

    var_output = pd.read_csv(f'./txt/VPD_curve/standardized_by_monthly_obs_mean_clarify_site/Qle_VPD_daytime_standardized_by_monthly_obs_mean_clarify_site_error_type=bootstrap_EF_model_0-0.2_bin_by_vpd_coarse.csv')
    plot_lines(var_output, model_order, message='standardized_by_monthly_obs_mean_clarify_site')
