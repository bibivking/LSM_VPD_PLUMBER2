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

def plot_pdf(var_input, model_out_list, message=None, plot_type='fitting_line', density=False, check_factor=None):

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

    if check_factor == None: 

        # create figure
        fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[10,10],sharex=True, sharey=False, squeeze=True)

        for model_name in model_out_list:

            if model_name != 'obs':
                varname  = 'model_'+model_name
            else:
                varname  = model_name

            if plot_type == 'fitting_line':
                # Plot the PDF of the normal distribution
                # read the data for this model
                var_vals = var_input[varname]
                    
                # remove nan values 
                var_vals = np.sort(var_vals[~ np.isnan(var_vals)])

                if np.any(var_vals):
        
                    bandwidth = 0.5
                    # Estimate the probability density function using kernel density estimation.
                    pdf       = gaussian_kde(var_vals, bw_method=bandwidth)
                    # Plot the probability density function.
                    ax.plot(var_vals, pdf(var_vals), color=model_colors[model_name],label=model_name)
                    
            if plot_type == 'hist':

                hist = ax.hist(var_input['VPD'], bins=100, density=density, alpha=0.6, color=model_colors[model_name],
                            label=model_name, histtype='stepfilled')

        ax.legend(fontsize=8,frameon=False)

        if message == None:
            fig.savefig("./plots/"+var_name+'_PDF.png',bbox_inches='tight',dpi=300)
        else:
            fig.savefig("./plots/"+var_name+'_PDF.png'+message,bbox_inches='tight',dpi=300)

    else:

        for model_name in model_out_list:
            
            # create figure
            fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[8,7],sharex=True, sharey=False, squeeze=True)

            if model_name != 'obs':
                varname  = 'model_'+model_name
            else:
                varname  = model_name
            
            # remove nan values 
            model_mask = ~ np.isnan(var_input[varname])

            plot1 = sns.histplot(data=var_input[model_mask], x='VPD', hue=check_factor, kde=False,  stat='percent',
                                 element="step", fill=False, ax=ax)

            # ax.legend(fontsize=8,frameon=False)
            
            ax.set_ylim(0, 2)
            if message == None:
                fig.savefig(f"./plots/{var_name}_PDF_{check_factor}_{model_name}.png", bbox_inches='tight',dpi=300)
            else:
                fig.savefig(f"./plots/{var_name}_PDF{message}_{check_factor}_{model_name}.png", bbox_inches='tight',dpi=300)

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

def plot_lines(var_input, model_out_list, message=None):

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

        vals = var_input[model_out_name+'_vals']

        above_200      = (var_input[model_out_name+'_vpd_num']>200)
        var_vpd_series = var_input['vpd_series']#[above_200]
        # vals           = vals[above_200]
        vals_bot       = var_input[model_out_name+'_bot']#[above_200]
        vals_top       = var_input[model_out_name+'_top']#[above_200]

        # start plotting
        if np.sum(var_input[model_out_name+'_vpd_num'])>0:#-200) > 0:
            # ax.plot(var_input['vpd_series'], var_input[model_out_name+'_vpd_num'], lw=2.0, color=line_color, alpha=0.7,label=model_out_name)
            # ax.axhline(y=200, color='black', linestyle='-.', linewidth=1)
            plot = ax.plot(var_vpd_series, vals, lw=2,
                                color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

            # fill = ax.fill_between(var_vpd_series,
            #                             vals_bot,
            #                             vals_top,
            #                             color=line_color, edgecolor="none",
            #                             alpha=0.5) #  .rolling(window=10).mean()

    ax.legend(fontsize=6, frameon=False, ncol=3)
    ax.text(0.12, 0.87, 'site_num='+str(var_input['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax.transAxes, fontsize=12)

    ax.set_xlim(0, 7.)
    # ax.set_ylim(-1, 4)

    fig.savefig('./plots/Fig_curves_'+message+'_plot_lines.png',bbox_inches='tight',dpi=300) # '_30percent'

    return

if __name__ == "__main__":


    var_name  = 'Qle'

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # Get model lists
    model_out_list = model_names['model_select']

    # Setting 
    time_scale     = 'daily'
    selected_by    = 'SLCT_EF_model' # 'EF_model' 
                                # 'EF_obs'
    method         = 'CRV_bins' # 'CRV_bins'
                                # 'CRV_fit_GAM'
    standardize    = None       # 'None'
                                # 'STD_LAI'
                                # 'STD_annual_obs'
                                # 'STD_monthly_obs'
                                # 'STD_monthly_model'
                                # 'STD_daily_obs'

    veg_fraction   = None #[0.7,1]
    day_time       = False  # False for daily
                            # True for half-hour or hourly

    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','Noah-MP']

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # # ================ 1D curve ================
    # uncertain_type = 'UCRTN_bootstrap'  # 'UCRTN_bootstrap'
    #                                     # 'UCRTN_percentile'
    #                                     # 'UCRTN_one_std'
    # selected_by    = 'SLCT_EF_model' # 'EF_model' 
    # bounds         = [0,0.2] #30
    # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                             standardize=standardize, country_code=country_code, selected_by=selected_by,
    #                                             bounds=bounds, veg_fraction=veg_fraction, method=method, uncertain_type=uncertain_type,
    #                                             clarify_site=clarify_site)

    # var_input = pd.read_csv(f'./txt/process4_output/{folder_name}/{var_name}{file_message}.csv')
    # plot_lines(var_input, model_out_list, message=f'{var_name}{file_message}.csv')


    # ================ PDF plot ================
    selected_by = 'SLCT_EF_model' # 'EF_model' 
    bounds      = [0,0.2] #30
    plot_type   = 'hist'#'fitting_line'
    density     = True
    check_factor = 'IGBP_type' #'site_name','IGBP_type','climate_type'

    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code, selected_by=selected_by,
                                                bounds=bounds, veg_fraction=veg_fraction, method=method,
                                                clarify_site=clarify_site)

    var_input    = pd.read_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD'+file_message+'.csv',na_values=[''])
                #    'VPD','obs_Tair','obs_Qair','obs_Precip',"obs_SWdown",'NoahMPv401_greenness'
    plot_pdf(var_input, model_out_list, message=file_message, check_factor=check_factor)

    # # Reading data
    # var_output    = pd.read_csv(f'./txt/{var_name}_all_sites.csv')
    #
    # # plot_pdf(var_output, model_out_list, plot_type=plot_type, density=density)
    # message = ''
    # plot_scatter(var_output, model_out_list, message=message)