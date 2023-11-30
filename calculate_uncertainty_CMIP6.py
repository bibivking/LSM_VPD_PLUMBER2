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

def calculate_uncertainties_at_CMIP6_VPD(var_name, bin_by=None, window_size=11, order=3,
                 smooth_type='S-G_filter', method='bin_by_vpd', message=None, model_names=None,
                 day_time=None,  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                 error_type=None, pdf_or_box='pdf',boxplot_file_names=None,region_name=None,
                 IGBP_type=None, clim_type=None,turning_point={'calc':False,'method':'kneed'}):


    # ============== read data ==============
    message = ''
    subfolder = ''

    if day_time:
        message = message + 'daytime'

    if IGBP_type != None:
        message = message + '_IGBP='+IGBP_type
    if clim_type != None:
        message = message + '_clim='+clim_type

    if standardize != None:
        message = message + '_standardized_'+standardize

    if clarify_site['opt']:
        message = message + '_clarify_site'

    if error_type !=None:
        message = message + '_error_type='+error_type

    if region_name !=None:
        message = message + '_'+region_name

    folder_name = 'original'

    if standardize != None:
        folder_name = 'standardized_'+standardize

    if clarify_site['opt']:
        folder_name = folder_name+'_clarify_site'

    if var_name == 'NEE':
        var_name = 'NEP'
    file_name = f'./txt/VPD_curve/{folder_name}/{subfolder}{var_name}_VPD_{message}_{bin_by}_0-1.0_{method}_coarse.csv'

    # Read lines data
    var = pd.read_csv(file_name)

    # how to get the model out list from the column names???
    model_out_list = []
    for column_name in var.columns:
        if "_vals" in column_name:
            model_out_list.append(column_name.split("_vals")[0])

    # models to plot
    model_order     = []
    model_names_all = model_names['model_select']
    for model_name in model_names_all:
        if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
            model_order.append(model_name)

    # Add obs
    if var_name in ['Qle','NEE','GPP','NEP']:
        model_order.append('obs')
    print('model_order',model_order)

    for j, model_out_name in enumerate(model_order):
        var_values = np.zeros(20)

        # ===== Drawing the lines =====
        # Unify NEE units : upwards CO2 movement is positive values

        if (var_name=='GPP') & ((model_out_name == 'CHTESSEL_ERA5_3') | (model_out_name == 'CHTESSEL_Ref_exp1')):
            print("(var_name=='GPP') & ('CHTESSEL' in model_out_name)")
            value = var[model_out_name+'_vals']*(-1)
        else:
            value = var[model_out_name+'_vals']

        # smooth or not
        if smooth_type != 'no_soomth':
            value = smooth_vpd_series(value, window_size, order, smooth_type)

        # only use vpd data points > 200
        above_200      = (var[model_out_name+'_vpd_num']>200)
        var_vpd_series = var['vpd_series'][above_200]
        value          = value[above_200]

        for i, boxplot in enumerate(boxplot_file_names):
            print('boxplot=',boxplot)
            boxplot_metrics = pd.read_csv(boxplot)

            VPD_mean, VPD_std, p5, p25, p75, p95 = boxplot_metrics['all_model']
            low_2std   = VPD_mean - 2*VPD_std
            low_1std   = VPD_mean - VPD_std
            high_1std  = VPD_mean + VPD_std
            high_2std  = VPD_mean + 2*VPD_std

            print('VPD_mean',VPD_mean,'low_2std',low_2std,'low_1std',low_1std)
            #var_at_mean
            var_values[0+i*5] = value[np.argmin(np.abs(var_vpd_series - VPD_mean))]
            var_values[1+i*5] = value[np.argmin(np.abs(var_vpd_series - low_2std))]
            var_values[2+i*5] = value[np.argmin(np.abs(var_vpd_series - low_1std))]
            var_values[3+i*5] = value[np.argmin(np.abs(var_vpd_series - high_1std))]
            var_values[4+i*5] = value[np.argmin(np.abs(var_vpd_series - high_2std))]

        if j == 0:
            var_at_VPD_metrics = pd.DataFrame({model_out_name: var_values})
        else:
            var_at_VPD_metrics[model_out_name] = var_values

    var_at_VPD_metrics.index = ['historical_mean', 'historical_mean-2std','historical_mean-std','historical_mean+std','historical_mean+2std',
                                'ssp126_mean', 'ssp126_mean-2std','ssp126_mean-std','ssp126_mean+std','ssp126_mean+2std',
                                'ssp245_mean', 'ssp245_mean-2std','ssp245_mean-std','ssp245_mean+std','ssp245_mean+2std',
                                'ssp585_mean','ssp585_mean-2std','ssp585_mean-std','ssp585_mean+std','ssp585_mean+2std',]

    if region_name!=None:
        var_at_VPD_metrics.to_csv(f'./txt/CMIP6/{var_name}_uncertainties_at_CMIP6_VPD_'+region_name+'.csv')
    else:
        var_at_VPD_metrics.to_csv(f'./txt/CMIP6/{var_name}_uncertainties_at_CMIP6_VPD.csv')

    return


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # Read site names, IGBP and clim
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    day_time       = True
    method         = 'bin_by_vpd' #'GAM'
    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    error_type     = 'one_std'
    # Smoothing setting

    region_name    = 'AU'
    if region_name != None:
        boxplot_file_names = ['/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_historical_global_land_metrics_'+region_name+'.csv',
                              '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp126_global_land_metrics_'+region_name+'.csv',
                              '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp245_global_land_metrics_'+region_name+'.csv',
                              '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp585_global_land_metrics_'+region_name+'.csv',]
    else:
        boxplot_file_names = ['/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_historical_global_land_metrics.csv',
                              '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp126_global_land_metrics.csv',
                              '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp245_global_land_metrics.csv',
                              '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/VPD_ssp585_global_land_metrics.csv',]

    window_size    = 11
    order          = 3
    smooth_type    = 'no_soomth' #'S-G_filter' #
    turning_point  =  {'calc':True, 'method':'piecewise'}
                      #{'calc':True, 'method':'cdf'}#{'calc':True, 'method':'kneed'}

    # for IGBP_type in IGBP_types:
    var_name = 'Qle'
    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'
    calculate_uncertainties_at_CMIP6_VPD(var_name=var_name, bin_by=bin_by, window_size=window_size, order=order,
             smooth_type=smooth_type, method='bin_by_vpd', model_names=model_names,
             day_time=day_time,  clarify_site=clarify_site, standardize=standardize,
             error_type=error_type, boxplot_file_names=boxplot_file_names,region_name=region_name,
             turning_point=turning_point) # IGBP_type=IGBP_type,

    # for IGBP_type in IGBP_types:
    var_name = 'GPP'
    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'
    calculate_uncertainties_at_CMIP6_VPD(var_name=var_name, bin_by=bin_by, window_size=window_size, order=order,
             smooth_type=smooth_type, method='bin_by_vpd', model_names=model_names,
             day_time=day_time,  clarify_site=clarify_site, standardize=standardize,
             error_type=error_type,boxplot_file_names=boxplot_file_names,region_name=region_name,
             turning_point=turning_point) # IGBP_type=IGBP_type,

    var_name = 'NEE'
    standardize    = None #"by_LAI"#'by_obs_mean'#'by_LAI'#None#'by_obs_mean'
    calculate_uncertainties_at_CMIP6_VPD(var_name=var_name, bin_by=bin_by, window_size=window_size, order=order,
             smooth_type=smooth_type, method='bin_by_vpd', model_names=model_names,
             day_time=day_time,  clarify_site=clarify_site, standardize=standardize,
             error_type=error_type,boxplot_file_names=boxplot_file_names,region_name=region_name,
             turning_point=turning_point) # IGBP_type=IGBP_type,
