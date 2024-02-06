'''
Including

filter Gs
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import sys
import gc
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *

def filter_gs(Gs_tmp, Wind_tmp, zscore_threshold=4):
    
    print('Gs_tmp', Gs_tmp)
    
    Gs_tmp   = np.where(Wind_tmp<1.0, np.nan, Gs_tmp)
    Gs_tmp   = conduct_quality_control('Gs', Gs_tmp, zscore_threshold)
    # Gs_tmp   = np.where(Gs_tmp<0.0,np.nan,Gs_tmp)
    # Gs_tmp   = np.where(Gs_tmp>4.5,np.nan,Gs_tmp) 

    # VPDl_tmp = np.where(VPDl_tmp<0.05 * 1000.,0.05 * 1000.,VPDl_tmp) 
    # VPDl_tmp = np.where(VPDl_tmp>7.* 1000, 7.* 1000,VPDl_tmp) 

    return Gs_tmp

def process_site_model_parallel(site_name, model_name, var_input, filter, zscore_threshold):

    header = 'model_' if model_name != 'obs' else ''
    file_input = f'./txt/process1_output/Gs/Gs_{site_name}_{model_name}.csv'

    if os.path.exists(file_input):
        gs_input = pd.read_csv(file_input, na_values=[''], usecols=['Gs', 'VPDl', 'Wind'])
        if filter:
            Gs_tmp         = gs_input['Gs'][:]
            VPDl_tmp       = gs_input['VPDl'][:]
            Wind_tmp       = gs_input['Wind'][:]
            gs_input['Gs'] = filter_gs(Gs_tmp, Wind_tmp, zscore_threshold)
        var_input.loc[var_input['site_name'] == site_name, header+model_name] = gs_input['Gs'].values
        var_input.loc[var_input['site_name'] == site_name, model_name+'_VPDl'] = gs_input['VPDl'].values/1000.  # Pa to kPa
    else:
        print('file_input', file_input, ' does not exist')
        var_input.loc[var_input['site_name'] == site_name, header+model_name] = np.nan
        var_input.loc[var_input['site_name'] == site_name, model_name+'_VPDl'] = np.nan

    return var_input

def write_gs_to_spatial_land_days_parallel(site_names, model_names, filter=False, zscore_threshold=4):

    var_input = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', na_values=[''], usecols=[
         'time', 'CABLE_EF', 'CABLE-POP-CN_EF', 'CHTESSEL_ERA5_3_EF', 'CHTESSEL_Ref_exp1_EF', 
         'CLM5a_EF', 'GFDL_EF', 'JULES_GL9_withLAI_EF', 'JULES_test_EF', 'MATSIRO_EF', 'MuSICA_EF', 
         'NASAEnt_EF', 'NoahMPv401_EF', 'ORC2_r6593_EF', 'ORC2_r6593_CO2_EF', 'ORC3_r7245_NEE_EF', 
         'ORC3_r8120_EF', 'QUINCY_EF', 'STEMMUS-SCOPE_EF', 'obs_EF', 'VPD', 'obs_Tair', 'obs_Qair',
         'obs_Precip','obs_SWdown', 'NoahMPv401_greenness','month','hour','site_name','IGBP_type',
         'climate_type', 'half_hrs_after_precip'])

    for model_in in model_names:
        if model_in == 'obs':
            header = ''
        else:
            header = 'model_'
        var_input[header+model_in]  = np.nan
        var_input[model_in+'_VPDl'] = np.nan

    with Pool() as pool:
        results = pool.starmap(process_site_model_parallel, [(site_name, model_name, var_input.copy(), filter, zscore_threshold) 
                                                    for site_name in site_names for model_name in model_names])

    # Combine results (assuming each result is the updated var_input dataframe)
    for updated_var_input in results:
        var_input = updated_var_input

    if filter:
        var_input.to_csv(f'./txt/process1_output/Gs_all_sites_filtered.csv')
    else:
        var_input.to_csv(f'./txt/process1_output/Gs_all_sites.csv')

    return


    
def write_gs_to_spatial_land_days(site_names, model_names,filter=False, zscore_threshold=4):

    """Parallelized version of the function."""

    var_input = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', na_values=[''], usecols=[
         'time', 'CABLE_EF', 'CABLE-POP-CN_EF', 'CHTESSEL_ERA5_3_EF', 'CHTESSEL_Ref_exp1_EF', 
         'CLM5a_EF', 'GFDL_EF', 'JULES_GL9_withLAI_EF', 'JULES_test_EF', 'MATSIRO_EF', 'MuSICA_EF', 
         'NASAEnt_EF', 'NoahMPv401_EF', 'ORC2_r6593_EF', 'ORC2_r6593_CO2_EF', 'ORC3_r7245_NEE_EF', 
         'ORC3_r8120_EF', 'QUINCY_EF', 'STEMMUS-SCOPE_EF', 'obs_EF', 'VPD', 'obs_Tair', 'obs_Qair',
         'obs_Precip','obs_SWdown', 'NoahMPv401_greenness','month','hour','site_name','IGBP_type',
         'climate_type', 'half_hrs_after_precip'])
    
    for model_in in model_names:
        if model_in == 'obs':
            header = ''
        else:
            header = 'model_'
        var_input[header+model_in]  = np.nan
        var_input[model_in+'_VPDl'] = np.nan

    for site_name in site_names:

        site_mask  = (var_input['site_name'] == site_name)

        for model_in in model_names:

            print('site ', site_name, 'model', model_in)

            if model_in == 'obs':
                header = ''
            else:
                header = 'model_'

            file_input = f'./txt/process1_output/Gs/Gs_{site_name}_{model_in}.csv'

            if os.path.exists(file_input):
                
                gs_input  = pd.read_csv(file_input, na_values=[''], usecols=['Gs','VPDl','Wind'])
                if filter:
                    Gs_tmp   = gs_input['Gs'][:]
                    VPDl_tmp = gs_input['VPDl'][:]
                    Wind_tmp = gs_input['Wind'][:]
                    gs_input['Gs'] = filter_gs(Gs_tmp, Wind_tmp, zscore_threshold)
                try:
                    var_input.loc[site_mask, header+model_in] = gs_input['Gs'].values
                    var_input.loc[site_mask, model_in+'_VPDl']= gs_input['VPDl'].values/1000. # Pa to kPa
                except:
                    var_input.loc[site_mask, header+model_in] = np.nan
                    var_input.loc[site_mask, model_in+'_VPDl']= np.nan

            else:
                print('file_input',file_input,' does not exist')

                var_input.loc[site_mask, header+model_in]  = np.nan
                var_input.loc[site_mask, model_in+'_VPDl'] = np.nan
            
            gc.collect()
    if filter:
        var_input.to_csv(f'./txt/process1_output/Gs_all_sites_filtered.csv')
    else:
        var_input.to_csv(f'./txt/process1_output/Gs_all_sites.csv')

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_met_path   = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    PLUMBER2_path       = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    PLUMBER2_path_input = "/g/data/w97/mm3972/data/PLUMBER2/"

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # # === Together ===
    filter = True
    zscore_threshold = 4 
    write_gs_to_spatial_land_days(site_names, model_names['model_select'],
                                  filter=filter, zscore_threshold=zscore_threshold)