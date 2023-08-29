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

if __name__ == "__main__":


    var_name  = 'latent'   
    plot_type = 'hist'#'fitting_line'
    density   = True

    # Reading data
    var_output    = pd.read_csv(f'./txt/{var_name}_all_sites.csv')
    
    # get the model namelist
    f             = nc.Dataset("/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/AR-SLu.nc", mode='r')
    model_in_list = f.variables[var_name + '_models']
    ntime         = len(f.variables['CABLE_time'])
    model_out_list= []

    for model_in in model_in_list:
        if len(f.variables[f"{model_in}_time"]) == ntime:
            model_out_list.append(model_in)
    model_out_list.append('obs')

    plot_pdf(var_output, model_out_list, plot_type=plot_type, density=density)