'''
Including:
    def time_mask
    def read_CMIP6
    def calculate_EF
    def make_CMIP6_nc_file
    def make_EF_extremes_nc_file
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

def calculate_LSM_lead_CMIP6_uncertainty_daily(CMIP6_out_path, scenario, percent=15, var_name='Qle'):

    # Read ACCESS-CM2 land fraction to divide land and sea
    f_landsea = nc.Dataset('/g/data/fs38/publications/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/historical/r1i1p1f1/fx/sftlf/gn/v20191108/sftlf_fx_ACCESS-CM2_historical_r1i1p1f1_gn.nc',
                           mode='r')
    land_frac = f_landsea.variables['sftlf'][:]
    f_landsea.close()    

    # CMIP6 files
    input_files= [ CMIP6_out_path + scenario + '.nc',
                   CMIP6_out_path + scenario + '_EF_bot_'+str(percent)+'th_percent.nc',
                   CMIP6_out_path + scenario + '_EF_top_'+str(percent)+'th_percent.nc']
    
    # Get model lists 
    site_names, IGBP_types, clim_types, model_names \
               = load_default_list()
    model_list = model_names['model_select']
    Qle_mean   = np.zeros((len(model_list),3))

    # Read CMIP6 variable
    for c, input_file in enumerate(input_files):

        print('scenario',c)
        
        f_cmip    = nc.Dataset(input_file, mode='r')
        Qle       = f_cmip.variables['Qle'][:]
        VPD       = f_cmip.variables['VPD'][:]
        EF        = f_cmip.variables['EF'][:]
        f_cmip.close()    

        # Get time length
        ntime     = len(Qle[:,0,0])

        # Replace ocean pixel with nan
        for j in np.arange(ntime):
            Qle[j,:,:] = np.where(land_frac==100, Qle[j,:,:], np.nan)
            VPD[j,:,:] = np.where(land_frac==100, VPD[j,:,:], np.nan)
            EF[j,:,:]  = np.where(land_frac==100, EF[j,:,:], np.nan)
        # plt.contourf(Qle[0,:,:])
        # plt.show()

        # Remove NaN values
        Qle_1d = Qle.flatten() 
        VPD_1d = VPD.flatten() 
        EF_1d  = EF.flatten()

        mask_all = (~np.isnan(Qle_1d)) & (~np.isnan(VPD_1d)) & (~np.isnan(EF_1d))
        Qle_1d = Qle_1d[mask_all] 
        VPD_1d = VPD_1d[mask_all] 
        EF_1d  = EF_1d[mask_all]  

        # divide Qle_1d, VPD_1d and EF_1d by EF_1d values
        EF_02_mask = (EF_1d >= 0)   & (EF_1d < 0.2)
        print('data points EF<0.2',np.sum(EF_02_mask))
        
        EF_04_mask = (EF_1d >= 0.2) & (EF_1d < 0.4)
        print('data points 0.2<=EF<0.4',np.sum(EF_04_mask))
        
        EF_06_mask = (EF_1d >= 0.4) & (EF_1d < 0.6)
        print('data points 0.4<=EF<0.6',np.sum(EF_06_mask))
        print('np.any(EF_1d >= 0.6)',np.any(EF_1d >= 0.6),'np.any(EF_1d < 0.8)',np.any(EF_1d < 0.8))

        EF_08_mask = (EF_1d >= 0.6) & (EF_1d < 0.8)
        print('data points 0.6<=EF<0.8',np.sum(EF_08_mask))

        EF_10_mask = (EF_1d >= 0.8) & (EF_1d <= 1.)
        print('data points 0.8<=EF<=1.',np.sum(EF_10_mask))

        Qle_02     = Qle_1d[EF_02_mask]
        Qle_04     = Qle_1d[EF_04_mask]
        Qle_06     = Qle_1d[EF_06_mask]
        Qle_08     = Qle_1d[EF_08_mask]
        Qle_10     = Qle_1d[EF_10_mask]

        VPD_02     = VPD_1d[EF_02_mask]
        VPD_04     = VPD_1d[EF_04_mask]
        VPD_06     = VPD_1d[EF_06_mask]
        VPD_08     = VPD_1d[EF_08_mask]
        VPD_10     = VPD_1d[EF_10_mask]

        EF_02      = EF_1d[EF_02_mask]
        EF_04      = EF_1d[EF_04_mask]
        EF_06      = EF_1d[EF_06_mask]
        EF_08      = EF_1d[EF_08_mask]
        EF_10      = EF_1d[EF_10_mask]

        # Read GAM models

        for m, model_in in enumerate(model_list):
            print('model', model_in)
            bounds = [0,0.2]
            folder_name, file_message = get_PLUMBER2_curve_names(bounds)
            Qle_pred_02 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_02)
            print('model', model_in, np.mean(Qle_pred_02))
            bounds = [0.2,0.4]
            folder_name, file_message = get_PLUMBER2_curve_names(bounds)
            Qle_pred_04 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_04)

            bounds = [0.4,0.6]
            folder_name, file_message = get_PLUMBER2_curve_names(bounds)
            Qle_pred_06 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_06)

            bounds = [0.6,0.8]
            folder_name, file_message = get_PLUMBER2_curve_names(bounds)
            Qle_pred_08 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_08)

            bounds = [0.8,1.]
            folder_name, file_message = get_PLUMBER2_curve_names(bounds)
            Qle_pred_10 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_10)
            Qle_pred    = np.concatenate((Qle_pred_02,Qle_pred_04,Qle_pred_06,Qle_pred_08,Qle_pred_10))
            Qle_mean[m,c] = np.mean(Qle_pred)

    print(Qle_mean)
    
    np.savetxt(f'{CMIP6_out_path}/LSM_lead_CMIP6_uncertainty/{scenario}_uncertainty.csv',Qle_mean)

    return EF

def calculate_LSM_lead_CMIP6_uncertainty_3hourly(CMIP6_3h_out_path, scenario, var_name='Qle'):

    # Read ACCESS-CM2 land fraction to divide land and sea

    
    input_file  = CMIP6_3h_out_path + scenario + '.nc'

    model_list  = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6',
                   'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']

    # Get model lists 
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list = model_names['model_select']

    # Open my processed CMIP6 file
    f_cmip    = nc.Dataset(input_file, mode='r')

    # make data ensembles
    for i, model_name in enumerate(model_list):
    
        Qle_tmp     = f_cmip.variables[model_name+'_Qle'][:]
        VPD_tmp     = f_cmip.variables[model_name+'_VPD'][:]
        EF_tmp      = f_cmip.variables[model_name+'_EF'][:]
        landsea_tmp = f_cmip.variables[model_name+'_landsea'][:]

        Qle_tmp    = np.where(landsea_tmp==1,Qle_tmp,np.nan)
        VPD_tmp    = np.where(landsea_tmp==1,VPD_tmp,np.nan)
        EF_tmp     = np.where(landsea_tmp==1,EF_tmp,np.nan)

        # Remove NaN values
        Qle_1d     = Qle_tmp.flatten() 
        VPD_1d     = VPD_tmp.flatten() 
        EF_1d      = EF_tmp.flatten()

        mask_all   = (~np.isnan(Qle_1d)) & (~np.isnan(VPD_1d)) & (~np.isnan(EF_1d))
        Qle_1d     = Qle_1d[mask_all] 
        VPD_1d     = VPD_1d[mask_all] 
        EF_1d      = EF_1d[mask_all]  

        var_tmp    = pd.DataFrame([Qle_1d,VPD_1d,EF_1d], columns=['Qle','VPD','EF'])
        if i == 0:
            var_output = var_tmp
        else:
            var_output = pd.concat([var_output, var_tmp], ignore_index=True)

    # divide Qle_1d, VPD_1d and EF_1d by EF_1d values
    EF         = var_output['EF'][:]
    Qle        = var_output['Qle'][:]
    VPD        = var_output['VPD'][:]
    EF_02_mask = (EF >= 0)   & (EF < 0.2)
    EF_04_mask = (EF >= 0.2) & (EF < 0.4)
    EF_06_mask = (EF >= 0.4) & (EF < 0.6)
    EF_08_mask = (EF >= 0.6) & (EF < 0.8)
    EF_10_mask = (EF >= 0.8) & (EF <= 1.)

    Qle_02     = Qle[EF_02_mask]
    Qle_04     = Qle[EF_04_mask]
    Qle_06     = Qle[EF_06_mask]
    Qle_08     = Qle[EF_08_mask]
    Qle_10     = Qle[EF_10_mask]

    VPD_02     = VPD[EF_02_mask]
    VPD_04     = VPD[EF_04_mask]
    VPD_06     = VPD[EF_06_mask]
    VPD_08     = VPD[EF_08_mask]
    VPD_10     = VPD[EF_10_mask]

    EF_02      = EF[EF_02_mask]
    EF_04      = EF[EF_04_mask]
    EF_06      = EF[EF_06_mask]
    EF_08      = EF[EF_08_mask]
    EF_10      = EF[EF_10_mask]

    # Read GAM models
    Qle_pred   = {}
    for m, model_in in enumerate(model_list):

        print('model', model_in)
        bounds = [0,0.2]
        folder_name, file_message = get_PLUMBER2_curve_names(bounds)
        Qle_pred_02 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_02)
        
        bounds = [0.2,0.4]
        folder_name, file_message = get_PLUMBER2_curve_names(bounds)
        Qle_pred_04 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_04)

        bounds = [0.4,0.6]
        folder_name, file_message = get_PLUMBER2_curve_names(bounds)
        Qle_pred_06 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_06)

        bounds = [0.6,0.8]
        folder_name, file_message = get_PLUMBER2_curve_names(bounds)
        Qle_pred_08 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_08)

        bounds = [0.8,1.]
        folder_name, file_message = get_PLUMBER2_curve_names(bounds)
        Qle_pred_10 = read_best_GAM_model(var_name, model_in, folder_name, file_message, VPD_10)

        Qle_pred[model_in] = np.concatenate((Qle_pred_02,Qle_pred_04,Qle_pred_06,Qle_pred_08,Qle_pred_10))

    np.savetxt(f'{CMIP6_txt_out_path}/predicted_{var_name}_{scenario}.csv',Qle_pred)

    return


def get_PLUMBER2_curve_names(bounds):

   # Path of PLUMBER 2 dataset
    PLUMBER2_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    var_name       = 'Qle'      #'TVeg'
    time_scale     = 'daily'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'
    
    method         = 'CRV_fit_GAM_complex' # 'CRV_bins'
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

    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}

    day_time       = False  # False for daily
                            # True for half-hour or hourly
    if var_name == 'Gs' and time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # ================ 1D curve ================
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code, method=method, 
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site) # 
    
    return folder_name, file_message

if __name__ == "__main__":

    # Read files
    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/mm3972/data/CMIP6_data/Processed_CMIP6_data/"
    CMIP6_da_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_daily/"
    CMIP6_3h_out_path = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_3hourly/"
    CMIP6_txt_out_path = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6'
    scenarios         = ['historical', 'ssp245']
    # scenarios         = ['historical','ssp126','ssp245','ssp370']
    percent           = 15
    var_name          = 'Qle'
    
    for scenario in scenarios:
        # Processing daily 
        # calculate_LSM_lead_CMIP6_uncertainty_daily(CMIP6_da_out_path, scenario, percent=percent, var_name=var_name)
    
        # Processing 3 hourly 
        calculate_LSM_lead_CMIP6_uncertainty_3hourly(CMIP6_3h_out_path, CMIP6_txt_out_path, scenario, percent=15, var_name='Qle')