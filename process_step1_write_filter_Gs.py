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
import multiprocessing as mp
import logging

def filter_gs_each_site_model(site_name, model_name, zscore_threshold=2, gs_ref=False, gap_fill='nan' ):

    logging.info(f"Starting processing for site {site_name} model {model_name} (process ID: {os.getpid()})")
    if gs_ref:
        file_input = f'./txt/process1_output/Gs/Gs_ref_{site_name}_{model_name}.csv'
    else:
        file_input = f'./txt/process1_output/Gs/Gs_{site_name}_{model_name}.csv'

    if os.path.exists(file_input):
        gs_input = pd.read_csv(file_input, na_values=[''])

        Gs_tmp     = gs_input['Gs']
        Wind_tmp   = gs_input['Wind']
        VPD_tmp    = gs_input['VPD']
        # SWdown_tmp = gs_input['obs_SWdown']

        # =========================================================
        # filtering the data
        Gs_tmp   = np.where(Wind_tmp<1.0, np.nan, Gs_tmp)
        # Gs_tmp   = conduct_quality_control('Gs', Gs_tmp, zscore_threshold, gap_fill=gap_fill)
        Gs_tmp   = np.where(Gs_tmp<0.0,np.nan,Gs_tmp)
        Gs_tmp   = np.where(Gs_tmp>4.5,np.nan,Gs_tmp)
        Gs_tmp   = np.where(VPD_tmp<0.6,np.nan,Gs_tmp)
        # Gs_tmp   = np.where(SWdown_tmp<50,np.nan,Gs_tmp)

        # VPDl_tmp = np.where(VPDl_tmp<0.05 * 1000.,0.05 * 1000.,VPDl_tmp)
        # VPDl_tmp = np.where(VPDl_tmp>7.* 1000, 7.* 1000,VPDl_tmp)
        # =========================================================

        gs_input.loc[:,'Gs'] = Gs_tmp

        # gs_input.to_csv(f'./txt/process1_output/Gs_filter/Gs_{site_name}_{model_name}_filter_{zscore_threshold}sigma.csv')
        if gs_ref:
            gs_input.to_csv(f'./txt/process1_output/Gs_filter/Gs_ref_{site_name}_{model_name}_filter.csv')
        else:
            gs_input.to_csv(f'./txt/process1_output/Gs_filter/Gs_{site_name}_{model_name}_filter.csv')
        return
    else:
        return

def write_filter_gs_parallel(site_names, model_names, zscore_threshold=4, gs_ref=False):

    logging.basicConfig(level=logging.INFO)

    with mp.Pool() as pool:
        pool.starmap(filter_gs_each_site_model, [(site_name, model_name, zscore_threshold, gs_ref)
                                                    for site_name in site_names for model_name in model_names])
    return

def write_gs_to_spatial_land_days(site_names, model_names, filter=False, gs_ref=False):

    """Parallelized version of the function."""

    var_input = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', na_values=[''], usecols=[
         'time', 'CABLE_EF', 'CABLE-POP-CN_EF', 'CHTESSEL_Ref_exp1_EF',
         'CLM5a_EF', 'GFDL_EF', 'JULES_GL9_EF', 'JULES_GL9_withLAI_EF', 'MATSIRO_EF', 'MuSICA_EF',
         'NASAEnt_EF', 'NoahMPv401_EF', 'ORC2_r6593_EF', 'ORC3_r8120_EF', 'QUINCY_EF',
         'STEMMUS-SCOPE_EF', 'obs_EF', 'VPD', 'obs_Tair', 'obs_Qair',
         'obs_Precip','obs_SWdown', 'month', 'hour', 'site_name','IGBP_type',
         'climate_type'])

    if gs_ref:
        ref_message = '_ref'
    else:
        ref_message = ''

    for model_in in model_names:
        if model_in == 'obs':
            header = ''
        else:
            header = 'model_'
        var_input[header+model_in]  = np.nan
        var_input[model_in+'_VPDl'] = np.nan
    var_input['obs_Wind'] = np.nan

    for site_name in site_names:

        site_mask  = (var_input['site_name'] == site_name)

        for i, model_in in enumerate(model_names):

            print('site ', site_name, 'model', model_in)

            if model_in == 'obs':
                header = ''
            else:
                header = 'model_'

            if filter:
                file_input = f'./txt/process1_output/Gs_filter/Gs{ref_message}_{site_name}_{model_in}_filter.csv'
            else:
                file_input = f'./txt/process1_output/Gs/Gs{ref_message}_{site_name}_{model_in}.csv'

            if os.path.exists(file_input):

                gs_input  = pd.read_csv(file_input, na_values=[''], usecols=['Gs','VPDl','Wind'])

                try:
                    var_input.loc[site_mask, header+model_in]  = gs_input['Gs'].values
                    var_input.loc[site_mask, model_in+'_VPDl'] = gs_input['VPDl'].values/1000. # Pa to kPa
                    if i == 0:
                        var_input.loc[site_mask, 'obs_Wind']   = gs_input['Wind'].values

                except:
                    print('Missing some of gs_input["Gs"] and gs_input["VPDl"]')

            gc.collect()

    if filter:
        var_input.to_csv(f'./txt/process1_output/Gs{ref_message}_all_sites_filtered.csv')
    else:
        var_input.to_csv(f'./txt/process1_output/Gs{ref_message}_all_sites.csv')

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_met_path   = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    PLUMBER2_path       = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    PLUMBER2_path_input = "/g/data/w97/mm3972/data/PLUMBER2/"

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # === Together ===
    zscore_threshold = None
    gs_ref           = False
    # write_filter_gs_parallel(site_names, model_names['model_select_new'], zscore_threshold=zscore_threshold,gs_ref=gs_ref)

    filter = True
    write_gs_to_spatial_land_days(site_names, model_names['model_select_new'],filter=filter,gs_ref=gs_ref)
