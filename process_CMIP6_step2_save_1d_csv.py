'''
Including:
    save_CMIP6_3hourly_parallel
    save_CMIP6_3hourly_each_model
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from matplotlib.patches import Polygon
import copy
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *
import multiprocessing

def save_CMIP6_3hourly_parallel(CMIP6_3h_out_path, CMIP6_txt_path, scenario, var_name='Qle', day_time=False,
                                region={'name':'global','lat':None, 'lon':None}, is_filter=False):

    # CMIP6 model list
    CMIP6_model_list  = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'MIROC-ES2L',  'MPI-ESM1-2-LR',
                         'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6', 'MPI-ESM1-2-HR','MRI-ESM2-0'] #

    # Creates a pool of worker processes to execute tasks in parallel.
    # The with statement ensures proper cleanup of the pool after use.
    with multiprocessing.Pool() as pool:

        # Applies the process_site function to multiple arguments in parallel using the worker processes.
        # starmap is similar to map but unpacks arguments from tuples or lists.
        pool.starmap(save_CMIP6_3hourly_each_model, [(CMIP6_3h_out_path, CMIP6_txt_path, CMIP6_model, scenario,
                                                      var_name, day_time, region, is_filter) for CMIP6_model in CMIP6_model_list])

    return

def save_CMIP6_3hourly_each_model(CMIP6_3h_out_path, CMIP6_txt_path, CMIP6_model, scenario, var_name='Qle', day_time=False,
                                  region={'name':'global','lat':None, 'lon':None}, is_filter=False):

    '''
    Read data from CMIP6 nc file, remove sea pixels and select region to save as 1 D data
    '''

    # Read ACCESS-CM2 land fraction to divide land and sea
    input_file = CMIP6_3h_out_path + scenario + '_'+CMIP6_model+'.nc'

    # Open my processed CMIP6 file
    f_cmip     = nc.Dataset(input_file, mode='r')

    Qle_tmp     = f_cmip.variables['Qle'][:]
    VPD_tmp     = f_cmip.variables['VPD'][:]
    EF_tmp      = f_cmip.variables['EF'][:]
    landsea_tmp = f_cmip.variables['landsea'][:]
    ntime       = len(Qle_tmp[:,0,0])

    landsea_3d  = np.repeat(landsea_tmp[np.newaxis, :, :], ntime, axis=0)
    print('np.shape(landsea_3d)',np.shape(landsea_3d))
    print('np.shape(Qle_tmp)',np.shape(Qle_tmp))

    if day_time:
        SWdown_tmp  = f_cmip.variables['SWdown'][:]
        # mask out ocean and night time
        if is_filter:
            Qle_tmp     = np.where(  (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), Qle_tmp, np.nan)
            VPD_tmp     = np.where(  (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), VPD_tmp, np.nan)
            EF_tmp      = np.where( (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), EF_tmp, np.nan)
        else:
            Qle_tmp     = np.where((landsea_3d==1) & (SWdown_tmp>=5), Qle_tmp, np.nan)
            VPD_tmp     = np.where((landsea_3d==1) & (SWdown_tmp>=5), VPD_tmp, np.nan)
            EF_tmp      = np.where((landsea_3d==1) & (SWdown_tmp>=5), EF_tmp, np.nan)
    else:
        # mask out ocean
        if is_filter:
            Qle_tmp     = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), Qle_tmp, np.nan)
            VPD_tmp     = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), VPD_tmp, np.nan)
            EF_tmp      = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), EF_tmp, np.nan)
        else:
            Qle_tmp     = np.where(landsea_3d==1, Qle_tmp, np.nan)
            VPD_tmp     = np.where(landsea_3d==1, VPD_tmp, np.nan)
            EF_tmp      = np.where(landsea_3d==1, EF_tmp, np.nan)

    # Select region
    if region['name'] != 'global':
        print(CMIP6_model)
        # get lat and lon
        Lat_tmp  = f_cmip.variables['lat'][:]
        Lon_tmp  = f_cmip.variables['lon'][:]
        lon_2d, lat_2d = np.meshgrid(Lon_tmp, Lat_tmp)

        lon_3d  = np.repeat(lon_2d[np.newaxis, :, :], ntime, axis=0)
        lat_3d  = np.repeat(lat_2d[np.newaxis, :, :], ntime, axis=0)

        # select region
        Qle_tmp = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   Qle_tmp, np.nan)
        VPD_tmp = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   VPD_tmp, np.nan)
        EF_tmp  = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   EF_tmp, np.nan)

    # To 1D
    Qle_1d     = Qle_tmp.flatten()
    VPD_1d     = VPD_tmp.flatten()
    EF_1d      = EF_tmp.flatten()

    Qle_tmp     = None
    VPD_tmp     = None
    EF_tmp      = None
    landsea_tmp = None
    landsea_3d  = None

    # Remove NaN values
    mask_all   = (~np.isnan(Qle_1d)) & (~np.isnan(VPD_1d)) & (~np.isnan(EF_1d))
    Qle_1d     = Qle_1d[mask_all]
    VPD_1d     = VPD_1d[mask_all]
    EF_1d      = EF_1d[mask_all]
    # Note that EF and VPD may be different between CMIP models
    # Thus the used CMIP6 data points can be different (need to make sure there is
    # not significant bias in sampling between CMIP models, otherwises ignore these
    # potential bias)

    var_tmp        = pd.DataFrame(Qle_1d, columns=['Qle'])
    var_tmp['VPD'] = VPD_1d
    var_tmp['EF']  = EF_1d

    print('Check Qle_1d, VPD_1d, EF_1d', var_tmp)

    f_cmip.close()

    if is_filter:
        message = 'filtered_by_VPD_'
    else:
        message = ''

    if day_time:
        var_tmp.to_csv(f'{CMIP6_txt_path}/CMIP6_DT_{message}{var_name}_{scenario}_{CMIP6_model}_{region["name"]}.csv')
    else:
        var_tmp.to_csv(f'{CMIP6_txt_path}/CMIP6_{message}{var_name}_{scenario}_{CMIP6_model}_{region["name"]}.csv')

    return

def save_EF_annual_hist_for_scenario_parallel(CMIP6_3h_out_path, CMIP6_txt_path, scenario, var_name='Qle', day_time=False,
                                              region={'name':'global','lat':None, 'lon':None}, is_filter=False):

    # CMIP6 model list
    CMIP6_model_list  = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'MIROC-ES2L',  'MPI-ESM1-2-LR',
                         'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6', 'MPI-ESM1-2-HR','MRI-ESM2-0'] #

    # Creates a pool of worker processes to execute tasks in parallel.
    # The with statement ensures proper cleanup of the pool after use.
    with multiprocessing.Pool() as pool:

        # Applies the process_site function to multiple arguments in parallel using the worker processes.
        # starmap is similar to map but unpacks arguments from tuples or lists.
        pool.starmap(save_EF_annual_hist_for_scenario, [(CMIP6_3h_out_path, CMIP6_txt_path, CMIP6_model, scenario,
                     var_name, day_time, region, is_filter) for CMIP6_model in CMIP6_model_list])

    return

def save_EF_annual_hist_for_scenario(CMIP6_3h_out_path, CMIP6_txt_path, CMIP6_model, scenario, var_name='Qle', day_time=False,
                                  region={'name':'global','lat':None, 'lon':None}, is_filter=False):

    '''
    Read data from CMIP6 nc file, remove sea pixels and select region to save as 1 D data
    '''

    # Read EF_annual_hist
    hist_file      = CMIP6_3h_out_path  + 'historical_'+CMIP6_model+'.nc'

    # Open my processed CMIP6 file
    f_hist         = nc.Dataset(hist_file, mode='r')
    EF_annual_hist = f_hist.variables['EF_annual'][:]
    f_hist.close()

    # Read ACCESS-CM2 land fraction to divide land and sea
    input_file     = CMIP6_3h_out_path  + scenario + '_'+CMIP6_model+'.nc'

    # Open my processed CMIP6 file
    f_cmip         = nc.Dataset(input_file, mode='r')
    Qle_tmp        = f_cmip.variables['Qle'][:]
    VPD_tmp        = f_cmip.variables['VPD'][:]
    EF_tmp         = f_cmip.variables['EF'][:]
    landsea_tmp    = f_cmip.variables['landsea'][:]
    ntime          = len(Qle_tmp[:,0,0])

    landsea_3d         = np.repeat(landsea_tmp[np.newaxis, :, :], ntime, axis=0)
    EF_annual_hist_3d  = np.repeat(EF_annual_hist[np.newaxis, :, :], ntime, axis=0)

    if day_time:
        SWdown_tmp  = f_cmip.variables['SWdown'][:]
        # mask out ocean and night time
        if is_filter:
            Qle_tmp     = np.where(  (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), Qle_tmp, np.nan)
            VPD_tmp     = np.where(  (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), VPD_tmp, np.nan)
            EF_tmp      = np.where(  (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), EF_tmp, np.nan)
            EF_annual_hist_tmp = np.where( (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), EF_annual_hist_3d, np.nan)

        else:
            Qle_tmp     = np.where((landsea_3d==1) & (SWdown_tmp>=5), Qle_tmp, np.nan)
            VPD_tmp     = np.where((landsea_3d==1) & (SWdown_tmp>=5), VPD_tmp, np.nan)
            EF_tmp      = np.where((landsea_3d==1) & (SWdown_tmp>=5), EF_tmp, np.nan)
            EF_annual_hist_tmp = np.where( (landsea_3d==1) & (SWdown_tmp>=5), EF_annual_hist_3d, np.nan)
    else:
        # mask out ocean
        if is_filter:
            Qle_tmp     = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), Qle_tmp, np.nan)
            VPD_tmp     = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), VPD_tmp, np.nan)
            EF_tmp      = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), EF_tmp, np.nan)
            EF_annual_hist_tmp = np.where( (landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), EF_annual_hist_3d, np.nan)
        else:
            Qle_tmp     = np.where(landsea_3d==1, Qle_tmp, np.nan)
            VPD_tmp     = np.where(landsea_3d==1, VPD_tmp, np.nan)
            EF_tmp      = np.where(landsea_3d==1, EF_tmp, np.nan)
            EF_annual_hist_tmp = np.where(landsea_3d==1, EF_annual_hist_3d, np.nan)

    # Select region
    if region['name'] != 'global':
        print(CMIP6_model)
        # get lat and lon
        Lat_tmp  = f_cmip.variables['lat'][:]
        Lon_tmp  = f_cmip.variables['lon'][:]
        lon_2d, lat_2d = np.meshgrid(Lon_tmp, Lat_tmp)

        lon_3d  = np.repeat(lon_2d[np.newaxis, :, :], ntime, axis=0)
        lat_3d  = np.repeat(lat_2d[np.newaxis, :, :], ntime, axis=0)

        # select region
        Qle_tmp = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   Qle_tmp, np.nan)
        VPD_tmp = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   VPD_tmp, np.nan)
        EF_tmp  = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   EF_tmp, np.nan)
        EF_annual_hist_tmp  = np.where(np.all([lat_3d >= region['lat'][0],
                                       lat_3d <= region['lat'][1],
                                       lon_3d >= region['lon'][0],
                                       lon_3d <= region['lon'][1]],axis=0),
                                       EF_annual_hist_tmp, np.nan)
    # To 1D
    Qle_1d     = Qle_tmp.flatten()
    VPD_1d     = VPD_tmp.flatten()
    EF_1d      = EF_tmp.flatten()
    EF_annual_hist_1d = EF_annual_hist_tmp.flatten()

    Qle_tmp     = None
    VPD_tmp     = None
    EF_tmp      = None
    landsea_tmp = None
    landsea_3d  = None
    EF_annual_hist_tmp = None
    EF_annual_hist_3d  = None

    # Remove NaN values
    mask_all          = (~np.isnan(Qle_1d)) & (~np.isnan(VPD_1d)) & (~np.isnan(EF_1d))
    Qle_1d            = Qle_1d[mask_all]
    VPD_1d            = VPD_1d[mask_all]
    EF_1d             = EF_1d[mask_all]
    EF_annual_hist_1d = EF_annual_hist_1d[mask_all]

    # Note that EF and VPD may be different between CMIP models
    # Thus the used CMIP6 data points can be different (need to make sure there is
    # not significant bias in sampling between CMIP models, otherwises ignore these
    # potential bias)

    var_tmp        = pd.DataFrame(EF_annual_hist_1d, columns=['EF_annual_hist'])

    f_cmip.close()

    if is_filter:
        message = 'filtered_by_VPD_'
    else:
        message = ''

    if day_time:
        var_tmp.to_csv(f'{CMIP6_txt_path}/CMIP6_DT_{message}EF_annual_hist_{scenario}_{CMIP6_model}_{region["name"]}.csv')
    else:
        var_tmp.to_csv(f'{CMIP6_txt_path}/CMIP6_{message}EF_annual_hist_{scenario}_{CMIP6_model}_{region["name"]}.csv')

    return


def save_lat_lon_for_scenario_parallel(CMIP6_3h_out_path, CMIP6_txt_path, scenario, var_name='Qle', day_time=False,
                                              region={'name':'global','lat':None, 'lon':None}, is_filter=False):

    # CMIP6 model list
    CMIP6_model_list  = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'MIROC-ES2L',  'MPI-ESM1-2-LR',
                         'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6', 'MPI-ESM1-2-HR','MRI-ESM2-0'] #

    # Creates a pool of worker processes to execute tasks in parallel.
    # The with statement ensures proper cleanup of the pool after use.
    with multiprocessing.Pool() as pool:

        # Applies the process_site function to multiple arguments in parallel using the worker processes.
        # starmap is similar to map but unpacks arguments from tuples or lists.
        pool.starmap(save_lat_lon_for_scenario, [(CMIP6_3h_out_path, CMIP6_txt_path, CMIP6_model, scenario,
                     var_name, day_time, region, is_filter) for CMIP6_model in CMIP6_model_list])

    return

def save_lat_lon_for_scenario(CMIP6_3h_out_path, CMIP6_txt_path, CMIP6_model, scenario, var_name='Qle', day_time=False,
                                  region={'name':'global','lat':None, 'lon':None}, is_filter=False):

    '''
    Read data from CMIP6 nc file, remove sea pixels and select region to save as 1 D data
    '''

    # Read EF_annual_hist
    hist_file      = CMIP6_3h_out_path  + 'historical_'+CMIP6_model+'.nc'

    # Read ACCESS-CM2 land fraction to divide land and sea
    input_file     = CMIP6_3h_out_path  + scenario + '_'+CMIP6_model+'.nc'

    # Open my processed CMIP6 file
    f_cmip         = nc.Dataset(input_file, mode='r')
    Qle_tmp        = f_cmip.variables['Qle'][:]
    VPD_tmp        = f_cmip.variables['VPD'][:]
    EF_tmp         = f_cmip.variables['EF'][:]
    landsea_tmp    = f_cmip.variables['landsea'][:]
    ntime          = len(Qle_tmp[:,0,0])

    landsea_3d     = np.repeat(landsea_tmp[np.newaxis, :, :], ntime, axis=0)

    if day_time:
        SWdown_tmp  = f_cmip.variables['SWdown'][:]
        # mask out ocean and night time
        if is_filter:
            Qle_tmp     = np.where(  (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), Qle_tmp, np.nan)
            VPD_tmp     = np.where(  (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), VPD_tmp, np.nan)
            EF_tmp      = np.where(  (landsea_3d==1) & (SWdown_tmp>=5)
                                   & (VPD_tmp>0.001) & (VPD_tmp<10), EF_tmp, np.nan)
        else:
            Qle_tmp     = np.where((landsea_3d==1) & (SWdown_tmp>=5), Qle_tmp, np.nan)
            VPD_tmp     = np.where((landsea_3d==1) & (SWdown_tmp>=5), VPD_tmp, np.nan)
            EF_tmp      = np.where((landsea_3d==1) & (SWdown_tmp>=5), EF_tmp, np.nan)
    else:
        # mask out ocean
        if is_filter:
            Qle_tmp     = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), Qle_tmp, np.nan)
            VPD_tmp     = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), VPD_tmp, np.nan)
            EF_tmp      = np.where((landsea_3d==1) & (VPD_tmp>0.001) & (VPD_tmp<10), EF_tmp, np.nan)
        else:
            Qle_tmp     = np.where(landsea_3d==1, Qle_tmp, np.nan)
            VPD_tmp     = np.where(landsea_3d==1, VPD_tmp, np.nan)
            EF_tmp      = np.where(landsea_3d==1, EF_tmp, np.nan)

    # Select region
    if region['name'] != 'global':
        print(CMIP6_model)
        # get lat and lon
        Lat_tmp  = f_cmip.variables['lat'][:]
        Lon_tmp  = f_cmip.variables['lon'][:]
        lon_2d, lat_2d = np.meshgrid(Lon_tmp, Lat_tmp)

        lon_3d  = np.repeat(lon_2d[np.newaxis, :, :], ntime, axis=0)
        lat_3d  = np.repeat(lat_2d[np.newaxis, :, :], ntime, axis=0)

        # select region
        Qle_tmp = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   Qle_tmp, np.nan)
        VPD_tmp = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   VPD_tmp, np.nan)
        EF_tmp  = np.where(np.all([lat_3d >= region['lat'][0],
                                   lat_3d <= region['lat'][1],
                                   lon_3d >= region['lon'][0],
                                   lon_3d <= region['lon'][1]],axis=0),
                                   EF_tmp, np.nan)
    # To 1D
    Qle_1d     = Qle_tmp.flatten()
    VPD_1d     = VPD_tmp.flatten()
    EF_1d      = EF_tmp.flatten()
    lat_1d     = lat_3d.flatten()
    lon_1d     = lon_3d.flatten()

    Qle_tmp     = None
    VPD_tmp     = None
    EF_tmp      = None
    landsea_tmp = None
    landsea_3d  = None

    # Remove NaN values
    mask_all          = (~np.isnan(Qle_1d)) & (~np.isnan(VPD_1d)) & (~np.isnan(EF_1d))
    lat_1d_mask       = lat_1d[mask_all]
    lon_1d_mask       = lon_1d[mask_all]

    # Note that EF and VPD may be different between CMIP models
    # Thus the used CMIP6 data points can be different (need to make sure there is
    # not significant bias in sampling between CMIP models, otherwises ignore these
    # potential bias)

    var_tmp        = pd.DataFrame(lat_1d_mask, columns=['lat'])
    var_tmp['lon'] = lon_1d_mask

    f_cmip.close()

    if is_filter:
        message = 'filtered_by_VPD_'
    else:
        message = ''

    if day_time:
        var_tmp.to_csv(f'{CMIP6_txt_path}/CMIP6_DT_{message}lat_lon_{scenario}_{CMIP6_model}_{region["name"]}.csv')
    else:
        var_tmp.to_csv(f'{CMIP6_txt_path}/CMIP6_{message}lat_lon_{scenario}_{CMIP6_model}_{region["name"]}.csv')

    return


if __name__ == "__main__":

    # Read files
    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/mm3972/data/CMIP6_data/Processed_CMIP6_data/"
    CMIP6_da_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_daily/"
    CMIP6_3h_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_3hourly/"
    CMIP6_txt_path    = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6'

    # Set options
    scenarios         = ['historical']#'historical',,'ssp245'] # ['historical','ssp126','ssp245','ssp370']
    percent           = 15
    var_name          = 'Qle'
    day_time          = True
    region_name       = 'north_Am' # 'west_EU', 'north_Am' 'east_AU'
    dist_type         = 'Gamma' #'Poisson' # 'Linear', None
    is_filter         = True

    if region_name == 'global':
        region = {'name':'global', 'lat':None, 'lon':None}
    elif region_name == 'east_AU':
        region = {'name':'east_AU', 'lat':[-44.5,-10], 'lon':[129,155]}
    elif region_name == 'west_EU':
        region = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
    elif region_name == 'north_Am':
        region = {'name':'north_Am', 'lat':[25,52], 'lon':[-125,-65]}

    # Get model lists
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list = model_names['model_select_new']

    for scenario in scenarios:

        # # Processing 3 hourly
        # save_CMIP6_3hourly_parallel(CMIP6_3h_out_path, CMIP6_txt_path, scenario, var_name=var_name,
        #                             day_time=day_time, region=region, is_filter=is_filter)

        # save_EF_annual_hist_for_scenario_parallel(CMIP6_3h_out_path, CMIP6_txt_path, scenario,
        #                             var_name=var_name, day_time=day_time, region=region, is_filter=is_filter)
        save_lat_lon_for_scenario_parallel(CMIP6_3h_out_path, CMIP6_txt_path, scenario,
                                    var_name=var_name, day_time=day_time, region=region, is_filter=is_filter)
