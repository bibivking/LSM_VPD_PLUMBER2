import os
import gc
import sys
import glob
import copy
import numpy as np
import pandas as pd
import netCDF4 as nc
import joblib
import multiprocessing as mp
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde
import datashader as ds
from datashader.mpl_ext import dsshow
import multiprocessing as mp
from PLUMBER2_VPD_common_utils import *

def check_saved_GAM_model():
    path1 = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/original_clarify_site/GAM_fit/bestGAM_Qle_hourly_RM16_DT_EF_model_0-0.2_CRV_fit_GAM_complex_ORC3_r8120_Gamma.pkl'
    loaded_model = joblib.load(path1)
    print('lam',loaded_model.lam,'n_splines',loaded_model.n_splines)
    return

def plotting_in_one_fig(model_list, bounds, LAI_range=None, veg_fraction=None, use_model_pkl=False,
                        confidence_intervals=False):

    # Path of PLUMBER 2 dataset
    # ======================= Setting =======================
    var_name       = 'Qle'       #'TVeg'
    time_scale     = 'hourly'   #'daily'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'
    standardize    = None
    IGBP_type      = None # 'GRA'
    clim_type      = None
    day_time       = True

    LAI_range      = None #[1.0, 2.0]

    # ===================== Default pre-processing =======================
    clarify_site   = {'opt': True,
                    'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                    'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

    if time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # ================= Plotting =================
    order    = ['(a)','(b)','(c)','(d)', '(e)','(f)','(g)','(h)']
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=2, ncols=4, figsize=[18,8],sharex=False, sharey=False, squeeze=False)

    # plt.subplots_adjust(wspace=0.15, hspace=0.1)

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

    part_name = 'part2'
    model_s   = 8#0
    model_e   = 16#8

    for i, model_in in enumerate(model_list[model_s:model_e]):

        row = int(i/4)
        col = i%4

        # read mean curves
        method         = 'CRV_bins'
        uncertain_type = 'UCRTN_bootstrap'
        dist_type      = None
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                    clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                    country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                    veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                    uncertain_type=uncertain_type, clarify_site=clarify_site)

        mean_curves = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/VPD_bins/{var_name}{file_message}.csv',
                                usecols=['vpd_series',model_in+'_vals',model_in+'_bot',model_in+'_top'])

        # read the GAM model (VPD from 0.001 to last VPD with >200 samples)
        dist_type      = 'Poisson'
        method         = 'CRV_fit_GAM_complex'
        uncertain_type = None
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                    clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                    country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                    veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                    uncertain_type=uncertain_type, clarify_site=clarify_site)
        if use_model_pkl:

            # Set up the VPD bins
            vpd_top      = 10.001 #7.04
            vpd_bot      = 0.001 #0.02
            vpd_interval = 0.1 #0.04
            vpd_series   = np.arange(vpd_bot,vpd_top,vpd_interval)

            vpd_top_type = 'sample_larger_200'
            y_pred, y_int = read_best_GAM_model(var_name, model_in, folder_name, file_message, vpd_series,
            dist_type=dist_type,vpd_top_type=vpd_top_type,confidence_intervals=confidence_intervals)
            GAM_curves              = pd.DataFrame(vpd_series, columns=['vpd_pred'])
            GAM_curves['y_pred']    = y_pred
            GAM_curves['y_int_bot'] = y_int[:,1]
            GAM_curves['y_int_top'] = y_int[:,0]

            vpd_top_type = 'to_10'
            y_pred, y_int = read_best_GAM_model(var_name, model_in, folder_name, file_message, vpd_series,
            dist_type=dist_type,vpd_top_type=vpd_top_type,confidence_intervals=confidence_intervals)
            GAM_curves_extent              = pd.DataFrame(vpd_series, columns=['vpd_pred'])
            GAM_curves_extent['y_pred']    = y_pred
            GAM_curves_extent['y_int_bot'] = y_int[:,1]
            GAM_curves_extent['y_int_top'] = y_int[:,0]

        else:
            GAM_curves = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/{dist_type}_greater_200_samples/GAM_fit/{var_name}{file_message}_{model_in}_{dist_type}.csv',
                                     usecols=['vpd_pred','y_pred','y_int_bot','y_int_top'])

            # read the concave GAM model (VPD 0.001 to 10.001)
            GAM_curves_extent = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/{dist_type}_to_10/GAM_fit/{var_name}{file_message}_{model_in}_{dist_type}.csv',
                                     usecols=['vpd_pred','y_pred','y_int_bot','y_int_top'])
                                     # usecols=['vpd_series',model_in+'_vals',model_in+'_bot',model_in+'_top'])

        # process3 selected data points
        folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                    standardize=standardize, country_code=country_code,
                                                    selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                    LAI_range=LAI_range, clarify_site=clarify_site) #

        #### MMY I edit here !!! ####
        file_input = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
        # file_input= f'raw_data_nonTVeg_VPD_hourly_RM16_DT_EF_model_{bounds[0]}-{bounds[1]}.csv'

        if model_in == 'obs':
            var_input = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/{file_input}',na_values=[''],usecols=['VPD',model_in])
        else:
            var_input = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/{file_input}',na_values=[''],usecols=['VPD','model_'+model_in])

        # Calculate the point density
        vpd= var_input['VPD']
        if model_in == 'obs':
            var= var_input[model_in]
        else:
            var= var_input['model_'+model_in]

        vpd = vpd[~np.isnan(var)]
        var = var[~np.isnan(var)]

        df = pd.DataFrame(dict(x=vpd, y=var))
        dsartist = dsshow(
            df,
            ds.Point("x", "y"),
            ds.count(),
            vmin=0,
            vmax=100,
            norm="linear",
            aspect="auto",
            ax=ax[row,col],
        )

        # Plot the line for X_predict and Y_predict
        ax[row,col].plot(mean_curves['vpd_series'][:len(GAM_curves)], mean_curves[model_in+'_vals'][:len(GAM_curves)], color='lightgray', ls= 'solid', label='mean')
        ax[row,col].fill_between(mean_curves['vpd_series'][:len(GAM_curves)], mean_curves[model_in+'_bot'][:len(GAM_curves)], mean_curves[model_in+'_top'][:len(GAM_curves)], color='lightgray', edgecolor="none", alpha=0.1) #  .

        ax[row,col].plot(GAM_curves['vpd_pred'], GAM_curves['y_pred'], color='gray', label='GAM', ls= 'dashed')
        ax[row,col].fill_between(GAM_curves['vpd_pred'], GAM_curves['y_int_bot'], GAM_curves['y_int_top'], color='gray', edgecolor="none", alpha=0.1) #  .

        # ax[row,col].plot(GAM_curves_extent['vpd_series'], GAM_curves_extent[model_in+'_vals'], color='black', label='GAM_concave', ls= 'dotted')
        # ax[row,col].fill_between(GAM_curves_extent['vpd_series'], GAM_curves_extent[model_in+'_bot'], GAM_curves_extent[model_in+'_top'], color='black', edgecolor="none", alpha=0.1) #  .

        ax[row,col].plot(GAM_curves_extent['vpd_pred'], GAM_curves_extent['y_pred'], color='black', label='GAM_concave', ls= 'dotted')
        ax[row,col].fill_between(GAM_curves_extent['vpd_pred'], GAM_curves_extent['y_int_bot'], GAM_curves_extent['y_int_top'], color='black', edgecolor="none", alpha=0.1) #  .

        # Add labels and title
        ax[row,col].text(0.05, 0.9, order[i]+' '+change_model_name(model_in), va='bottom', ha='left', rotation_mode='anchor',transform=ax[row,col].transAxes, fontsize=14)

        ax[row,col].set_xlim(0, 10)  # Set x-axis limits

        # ax[row,col].set_xticks(fontsize=12)
        # ax[row,col].set_yticks(fontsize=12)

        if bounds[0] == 0:
            ax[row,col].set_ylim(0, 200)  # Set y-axis limits
        elif bounds[0] == 0.2:
            ax[row,col].set_ylim(0, 600)  # Set y-axis limits
        elif bounds[0] == 0.4:
            ax[row,col].set_ylim(0, 600)  # Set y-axis limits
        elif bounds[0] == 0.6:
            ax[row,col].set_ylim(0, 600)  # Set y-axis limits
        elif bounds[0] == 0.8:
            ax[row,col].set_ylim(0, 600)  # Set y-axis limits

    ax[1, 0].set_xlabel('VPD (kPa)')
    ax[1, 1].set_xlabel('VPD (kPa)')
    ax[1, 2].set_xlabel('VPD (kPa)')
    ax[1, 3].set_xlabel('VPD (kPa)')
    ax[0, 0].set_ylabel('Latent heat (W m$\mathregular{^{-2}}$)')
    ax[1, 0].set_ylabel('Latent heat (W m$\mathregular{^{-2}}$)')
    location = fig.add_axes([0.13, 0.02, 0.76, 0.016] ) # [left, bottom, width, height]
    fig.colorbar(dsartist, ax=ax[:,:], pad=0.5, cax=location, orientation="horizontal", aspect=60, shrink=1.)

    if LAI_range != None:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{part_name}_LAI_{LAI_range[0]}-{LAI_range[1]}_EF_{bounds[0]}-{bounds[1]}_test.png",bbox_inches='tight',dpi=300)
    elif IGBP_type != None:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{part_name}_IGBP={IGBP_type}_EF_{bounds[0]}-{bounds[1]}_test.png",bbox_inches='tight',dpi=300)
    else:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{part_name}_EF_{bounds[0]}-{bounds[1]}_test.png",bbox_inches='tight',dpi=300)

def plotting(model_in, bounds, LAI_range=None, veg_fraction=None):

    print('model is ', model_in)

    # Path of PLUMBER 2 dataset

    # ======================= Setting =======================
    var_name       = 'Qle'       #'TVeg'
    time_scale     = 'hourly'   #'daily'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'
    standardize    = None
    IGBP_type      = None # 'GRA'
    clim_type      = None
    day_time       = True
    uncertain_type = 'UCRTN_bootstrap'

    LAI_range      = None #[1.0, 2.0]

    # ===================== Default pre-processing =======================
    clarify_site   = {'opt': True,
                    'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                    'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

    if time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # read mean curves
    method         = 'CRV_bins'
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                uncertain_type=uncertain_type, clarify_site=clarify_site)
    mean_curves = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/{var_name}{file_message}.csv',
                              usecols=['vpd_series',model_in+'_vals',model_in+'_bot',model_in+'_top'])


    # read the GAM model (VPD from 0.001 to last VPD with >200 samples)
    dist_type     = 'Gamma'
    method        = 'CRV_fit_GAM_complex'
    uncertain_type= None
    #
    # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
    #                                             clim_type=clim_type, time_scale=time_scale, standardize=standardize,
    #                                             country_code=country_code, selected_by=selected_by, bounds=bounds,
    #                                             veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
    #                                             uncertain_type=uncertain_type, clarify_site=clarify_site)
    #
    # GAM_curves = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message}_{model_in}_{dist_type}.csv',
    #                          usecols=['vpd_pred','y_pred','y_int_bot','y_int_top'])
    #
    # # read the concave GAM model (VPD 0.001 to 10.001)
    # GAM_curves_extent = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/Gamma_concave/{var_name}{file_message}_{dist_type}.csv',
    #                          usecols=['vpd_series',model_in+'_vals',model_in+'_bot',model_in+'_top'])
    #
    # # process3 selected data points
    # folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
    #                                             standardize=standardize, country_code=country_code,
    #                                             selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
    #                                             LAI_range=LAI_range, clarify_site=clarify_site) #

    #### MMY I edit here !!! ####
    file_input= 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    # file_input= f'raw_data_nonTVeg_VPD_hourly_RM16_DT_EF_model_{bounds[0]}-{bounds[1]}.csv'

    if model_in == 'obs':
        var_input = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/{file_input}',na_values=[''],usecols=['VPD',model_in])
    else:
        var_input = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/{file_input}',na_values=[''],usecols=['VPD','model_'+model_in])

    # ================= Method 2 =================
    fig, ax = plt.subplots()

    # Calculate the point density
    vpd= var_input['VPD']
    if model_in == 'obs':
        var= var_input[model_in]
    else:
        var= var_input['model_'+model_in]
    vpd = vpd[~np.isnan(var)]
    var = var[~np.isnan(var)]

    df = pd.DataFrame(dict(x=vpd, y=var))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin=0,
        vmax=100,
        norm="linear",
        aspect="auto",
        ax=ax,
    )

    # Plot the line for X_predict and Y_predict
    ax.plot(mean_curves['vpd_series'], mean_curves[model_in+'_vals'], color='red', label='mean')
    ax.fill_between(mean_curves['vpd_series'], mean_curves[model_in+'_bot'], mean_curves[model_in+'_top'], color='red', edgecolor="none", alpha=0.1) #  .

    ax.plot(GAM_curves['vpd_pred'], GAM_curves['y_pred'], color='blue', label='GAM')
    ax.fill_between(GAM_curves['vpd_pred'], GAM_curves['y_int_bot'], GAM_curves['y_int_top'], color='blue', edgecolor="none", alpha=0.1) #  .

    ax.plot(GAM_curves_extent['vpd_series'], GAM_curves_extent[model_in+'_vals'], color='green', label='GAM_concave')
    ax.fill_between(GAM_curves_extent['vpd_series'], GAM_curves_extent[model_in+'_bot'], GAM_curves_extent[model_in+'_top'], color='green', edgecolor="none", alpha=0.1) #  .

    # Add labels and title
    ax.set_xlabel('VPD')
    ax.set_ylabel('Qle')
    ax.set_title(model_in)
    ax.set_xlim(0, 10)  # Set x-axis limits
    ax.set_ylim(0, 600)  # Set y-axis limits

    plt.colorbar(dsartist)

    if LAI_range != None:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_LAI_{LAI_range[0]}-{LAI_range[1]}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)
    elif IGBP_type != None:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_IGBP={IGBP_type}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)
    else:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)

def plotting_clim_colors(model_in, bounds, LAI_range=None, veg_fraction=None):

    print('model is ', model_in)

    # Path of PLUMBER 2 dataset

    # ======================= Setting =======================
    var_name       = 'Qle'       #'TVeg'
    time_scale     = 'hourly'   #'daily'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'
    standardize    = None
    IGBP_type      = None # 'GRA'
    clim_type      = None
    day_time       = True
    uncertain_type = 'UCRTN_bootstrap'

    # ===================== Default pre-processing =======================
    clarify_site      = {'opt': True,
                         'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                         'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

    if time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # read mean curves
    method         = 'CRV_bins'
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                uncertain_type=uncertain_type, clarify_site=clarify_site)
    mean_curves = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/{var_name}{file_message}.csv',
                              usecols=['vpd_series',model_in+'_vals',model_in+'_bot',model_in+'_top'])


    # read the GAM model (VPD from 0.001 to last VPD with >200 samples)
    dist_type     = 'Gamma'
    method        = 'CRV_fit_GAM_complex'
    uncertain_type= None
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                uncertain_type=uncertain_type, clarify_site=clarify_site)

    GAM_curves = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/Gamma_greater_200_samples/GAM_fit/{var_name}{file_message}_{model_in}_{dist_type}.csv',
                             usecols=['vpd_pred','y_pred','y_int_bot','y_int_top'])

    # read the concave GAM model (VPD 0.001 to 10.001)
    GAM_curves_extent = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/Gamma_concave/{var_name}{file_message}_{dist_type}.csv',
                             usecols=['vpd_series',model_in+'_vals',model_in+'_bot',model_in+'_top'])

    # process3 selected data points
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code,
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site) #

    file_input= 'raw_data_'+var_name+'_VPD'+file_message+'.csv'

    if model_in == 'obs':
        var_input = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/{file_input}',na_values=[''],usecols=['VPD','climate_type',model_in])
    else:
        var_input = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/{file_input}',na_values=[''],usecols=['VPD','climate_type','model_'+model_in])

    # all climate types
    climate_types = ['Af','Am','Aw','BWh','BWk','BSh','BSk','Csa',
                     'Csb','Cwa','Cfa','Cfb','Dsb','Dwa','Dwb',
                     'Dfa','Dfb','Dfc','ET']

    # Plot site the color based on climate
    clim_colors = set_clim_colors()

    # Dictionary to store unique labels and corresponding colors
    unique_labels = {}

    # ================= Method 2 =================
    fig, ax = plt.subplots()

    for climate_type in climate_types:

        color   = clim_colors[climate_type]
        var_tmp = var_input[var_input['climate_type']==climate_type]
        print(climate_type, var_tmp)
        # Calculate the point density
        vpd     = var_tmp['VPD']
        if model_in == 'obs':
            var= var_tmp[model_in]
        else:
            var= var_tmp['model_'+model_in]
        vpd = vpd[~np.isnan(var)]
        var = var[~np.isnan(var)]

        ax.scatter(vpd, var, color=color, alpha=0.05, marker='o', s=0.1) #markeredgewidth=0.8)

        unique_labels[climate_type] = color  # First occurrence, add label and color

    # Plot the line for X_predict and Y_predict
    ax.plot(mean_curves['vpd_series'], mean_curves[model_in+'_vals'], color='lightgray', ls= 'solid', label='mean')
    ax.fill_between(mean_curves['vpd_series'], mean_curves[model_in+'_bot'], mean_curves[model_in+'_top'], color='lightgray', edgecolor="none", alpha=0.1) #  .

    ax.plot(GAM_curves['vpd_pred'], GAM_curves['y_pred'], color='gray', label='GAM', ls= 'dashed')
    ax.fill_between(GAM_curves['vpd_pred'], GAM_curves['y_int_bot'], GAM_curves['y_int_top'], color='gray', edgecolor="none", alpha=0.1) #  .

    ax.plot(GAM_curves_extent['vpd_series'], GAM_curves_extent[model_in+'_vals'], color='black',ls= 'dotted', label='GAM_concave')
    ax.fill_between(GAM_curves_extent['vpd_series'], GAM_curves_extent[model_in+'_bot'], GAM_curves_extent[model_in+'_top'], color='black', edgecolor="none", alpha=0.1) #  .

    # Add labels and title
    ax.set_xlabel('VPD')
    ax.set_ylabel('Qle')
    ax.set_title(model_in)
    ax.set_xlim(0, 10)  # Set x-axis limits
    if bounds[0] == 0:
        ax.set_ylim(0, 200)  # Set y-axis limits
    elif bounds[0] == 0.2:
        ax.set_ylim(0, 600)  # Set y-axis limits
    elif bounds[0] == 0.4:
        ax.set_ylim(0, 600)  # Set y-axis limits
    elif bounds[0] == 0.6:
        ax.set_ylim(0, 600)  # Set y-axis limits
    elif bounds[0] == 0.8:
        ax.set_ylim(0, 600)  # Set y-axis limits

    # Create legend entries with unique labels and colors
    legend_handles = []
    for label, color in unique_labels.items():
        legend_handles.append(plt.Line2D([], [], marker='o', color=color, alpha=1, markerfacecolor=color, markeredgewidth=0.8,
                            markersize=5, label=label, linestyle='None'))

    # Add the legend with unique entries
    ax.legend(handles=legend_handles, fontsize=8, frameon=False, ncol=2)

    # # Add labels and title
    # ax.set_xlabel('VPD')
    # ax.set_ylabel('Qle')
    # ax.set_title(model_in)
    # ax.set_xlim(0, 10)  # Set x-axis limits
    # ax.set_ylim(0, 600)  # Set y-axis limits

    # plt.colorbar(dsartist)

    if LAI_range != None:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_LAI_{LAI_range[0]}-{LAI_range[1]}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)
    elif IGBP_type != None:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_IGBP={IGBP_type}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)
    else:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)
    return

def plotting_parallel(model_list, bounds, LAI_range=None, veg_fraction=None, clim_color=False):

    if clim_color:
        with mp.Pool() as pool:
            pool.starmap(plotting_clim_colors, [(model_in, bounds, LAI_range, veg_fraction) for model_in in model_list])

    else:
        with mp.Pool() as pool:
            pool.starmap(plotting, [(model_in, bounds, LAI_range, veg_fraction) for model_in in model_list])

    # with mp.Pool() as pool:
    #     pool.map(plotting, model_list)

    return


if __name__ == "__main__":

    # model_list        = ['CABLE', 'CABLE-POP-CN', 'CHTESSEL_ERA5_3',
    #                      'CHTESSEL_Ref_exp1', 'CLM5a', 'GFDL',
    #                      'JULES_GL9_withLAI', 'JULES_test',
    #                      'MATSIRO', 'MuSICA', 'NASAEnt',
    #                      'NoahMPv401', 'ORC2_r6593', 'ORC2_r6593_CO2',
    #                      'ORC3_r7245_NEE', 'ORC3_r8120', 'QUINCY',
    #                      'STEMMUS-SCOPE', 'obs'] #"BEPS"

    model_list        = ['CABLE', 'CABLE-POP-CN',
                         'CHTESSEL_Ref_exp1', 'CLM5a', 'GFDL',
                         'JULES_GL9', 'JULES_GL9_withLAI',
                         'MATSIRO', 'MuSICA', 'NASAEnt',
                         'NoahMPv401', 'ORC2_r6593',
                         'ORC3_r8120', 'QUINCY',
                         'STEMMUS-SCOPE', 'obs'] #"BEPS"

    # model_list        = ['QUINCY']#'CABLE-POP-CN']#, 'QUINCY',]
    # bounds_all    = [[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.]]

    bounds_all    = [[0,0.2],[0.8,1.]]

    for bounds in bounds_all:

        LAI_range     = None
        veg_fraction  = None
        clim_color    = True
        use_model_pkl = True
        confidence_intervals=True
        plotting_in_one_fig(model_list, bounds=bounds, LAI_range=LAI_range, veg_fraction=veg_fraction,use_model_pkl=use_model_pkl,
                            confidence_intervals=confidence_intervals)

        # plotting_parallel(model_list, bounds=bounds, LAI_range=LAI_range, veg_fraction=veg_fraction,
        #                 clim_color=clim_color)

    # for model_in in model_list:
    #     plotting(model_in)
