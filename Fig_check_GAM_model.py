import os
import gc
import sys
import glob
import copy
import numpy as np
import pandas as pd
import netCDF4 as nc
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

# def decide_filename(day_time=False, summer_time=False, energy_cor=False,
#                     IGBP_type=None, clim_type=None, time_scale=None, standardize=None,
#                     country_code=None, selected_by=None, bounds=None, veg_fraction=None,
#                     uncertain_type=None, method=None,LAI_range=None,
#                     clarify_site={'opt':False,'remove_site':None}):

#     # file name
#     file_message = ''

#     if time_scale != None:
#         file_message = file_message + '_' + time_scale

#     if IGBP_type != None:
#         file_message = file_message + '_PFT='+IGBP_type

#     if clim_type != None:
#         file_message = file_message + '_CLIM='+clim_type

#     if veg_fraction !=None:
#         # if selected based on vegetation fraction
#         file_message = file_message + '_VF='+str(veg_fraction[0])+'-'+str(veg_fraction[1])

#     if LAI_range !=None:
#         # if selected based on LAI
#         file_message = file_message + '_LAI='+str(LAI_range[0])+'-'+str(LAI_range[1])

#     if country_code !=None:
#         # if for a country/region
#         file_message = file_message +'_'+country_code

#     if clarify_site['opt']:
#         # if remove 16 sites with problems in observation
#         file_message = file_message + '_RM16'

#     if day_time:
#         # if only daytime
#         file_message = file_message + '_DT'

#     if standardize != None:
#         # if the data is standardized
#         file_message = file_message + '_'+standardize

#     if selected_by !=None:
#         # which criteria used for binning the data
#         file_message = file_message +'_'+selected_by

#         if len(bounds) >1:
#             # percentile
#             if bounds[1] > 1:
#                 file_message = file_message + '_'+str(bounds[0])+'-'+str(bounds[1])+'th'
#             else:
#                 file_message = file_message + '_'+str(bounds[0])+'-'+str(bounds[1])
#         elif len(bounds) == 1 :
#             # fraction
#             if bounds[1] > 1:
#                 file_message = file_message + '_'+str(bounds[0])+'th'
#             else:
#                 file_message = file_message + '_'+str(bounds[0])

#     if method != None:
#         file_message = file_message + '_' + method

#     if uncertain_type != None and method == 'CRV_bins':
#         file_message = file_message + '_' + uncertain_type

#     folder_name = 'original'

#     if standardize != None:
#         folder_name = 'standardized_'+standardize

#     if clarify_site['opt']:
#         folder_name = folder_name+'_clarify_site'

#     return folder_name, file_message

def plotting(model_in):

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
    LAI_range      = None
    veg_fraction   = None   #[0.7,1]
    bounds         = [0, 0.2] #30

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
    # GAM_curves = pd.read_csv(f'/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process4_output/{folder_name}/GAM_fit/{var_name}{file_message}_{model_in}_{dist_type}.csv',
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
    # file_input= 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    file_input= f'raw_data_nonTVeg_VPD_hourly_RM16_DT_EF_model_{bounds[0]}-{bounds[1]}.csv'

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
    #
    # # Plot the line for X_predict and Y_predict
    # ax.plot(mean_curves['vpd_series'], mean_curves[model_in+'_vals'], color='red', label='mean')
    # ax.fill_between(mean_curves['vpd_series'], mean_curves[model_in+'_bot'], mean_curves[model_in+'_top'], color='red', edgecolor="none", alpha=0.1) #  .
    #
    # ax.plot(GAM_curves['vpd_pred'], GAM_curves['y_pred'], color='blue', label='GAM')
    # ax.fill_between(GAM_curves['vpd_pred'], GAM_curves['y_int_bot'], GAM_curves['y_int_top'], color='blue', edgecolor="none", alpha=0.1) #  .
    #
    # ax.plot(GAM_curves_extent['vpd_series'], GAM_curves_extent[model_in+'_vals'], color='green', label='GAM_concave')
    # ax.fill_between(GAM_curves_extent['vpd_series'], GAM_curves_extent[model_in+'_bot'], GAM_curves_extent[model_in+'_top'], color='green', edgecolor="none", alpha=0.1) #  .
    #
    # # Add labels and title
    # ax.set_xlabel('VPD')
    # ax.set_ylabel('Qle')
    # ax.set_title(model_in)
    # ax.set_xlim(0, 10)  # Set x-axis limits
    # ax.set_ylim(0, 600)  # Set y-axis limits

    plt.colorbar(dsartist)
    if LAI_range != None:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_LAI_{LAI_range[0]}-{LAI_range[1]}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)
    elif IGBP_type != None:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_IGBP={IGBP_type}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)
    else:
        fig.savefig(f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/plots/Check_GAM_model_{model_in}_EF_{bounds[0]}-{bounds[1]}.png",bbox_inches='tight',dpi=300)

def plotting_parallel(model_list):

    with mp.Pool() as pool:
        pool.map(plotting, model_list)

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

    plotting_parallel(model_list)

    # for model_in in model_list:
    #     plotting(model_in)
