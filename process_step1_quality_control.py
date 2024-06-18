'''
Including
    def quality_control_process1_output
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import gc
import sys
import glob
import copy
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

def write_qc_to_csv(PLUMBER2_flux_path, site_names):

    """
    Read Qle_qc and Qh_qc for flux files and write to csv.
    """

    var_input = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', na_values=[''], usecols=[
         'time', 'month', 'hour', 'site_name','IGBP_type', 'climate_type'])

    var_input['Qle_qc']         = np.nan
    var_input['Qh_qc']          = np.nan
    var_input['Qle_Qh_qc']      = np.nan
    var_input['Rnet_qc']        = np.nan
    var_input['Qle_Qh_Rnet_qc'] = np.nan

    for site_name in site_names:

        site_mask  = (var_input['site_name'] == site_name)
        file_path  = glob.glob(PLUMBER2_flux_path+"/*"+site_name+"*.nc")
        print(file_path)
        f          = nc.Dataset(file_path[0])
        Qle_qc     = f.variables['Qle_qc'][:,0,0]
        Qh_qc      = f.variables['Qh_qc'][:,0,0]
        try:
            Rnet_qc    = f.variables['Rnet_qc'][:,0,0]
            Rnet_exist = True
        except:
            print(site_name,'has not Rnet_qc using Qle_qc & Qh_qc & Qg_qc')
            Qg_qc      = f.variables['Qg_qc'][:,0,0]
            Rnet_exist = False

        if os.path.exists(file_path[0]):
            var_input.loc[site_mask, 'Qle_qc']    = np.where( Qle_qc<=2, Qle_qc, np.nan)
            var_input.loc[site_mask, 'Qh_qc']     = np.where( Qh_qc<=2,  Qh_qc,  np.nan)
            var_input.loc[site_mask, 'Qle_Qh_qc'] = np.where((Qle_qc<=2) & (Qh_qc<=2), 1, np.nan )

            if Rnet_exist:
                var_input.loc[site_mask, 'Rnet_qc']   = np.where( Rnet_qc<=2,  Rnet_qc,  np.nan)
                var_input.loc[site_mask, 'Qle_Qh_Rnet_qc'] = \
                                                    np.where((Qle_qc<=2) & (Qh_qc<=2) & (Rnet_qc<=2), 1, np.nan )
            else:
                var_input.loc[site_mask, 'Qle_Qh_Rnet_qc'] = \
                                                    np.where((Qle_qc<=2) & (Qh_qc<=2) & (Qg_qc<=2), 1, np.nan )
        else:
            print(f"{file_path[0]} doesn't exist")

        gc.collect()

    var_input.to_csv(f'./txt/process1_output/Qle_Qh_Rnet_quality_control_all_sites.csv')

    return

def quality_control_process1_output(var_name, site_names, zscore_threshold=4, gap_fill='nan'):

    # read the variables
    var_output   = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites.csv', na_values=[''])

    for site_name in site_names:
        print('site',site_name)
        site_mask = (var_output['site_name'] == site_name)
        for col_name in var_output.columns:
            if 'obs' in col_name or 'model_' in col_name:
                var_output.loc[site_mask, col_name] = \
                    conduct_quality_control(col_name, var_output[site_mask][col_name].values, zscore_threshold,gap_fill)

    var_output.to_csv(f'./txt/process1_output/{var_name}_all_sites_filter_{zscore_threshold}sigma.csv')

def quality_control_for_site(site_name, var_output, zscore_threshold,gap_fill='nan'):

    """
    Performs quality control for a single site.
    """

    logging.info(f"Starting processing for site {site_name} (process ID: {os.getpid()})")

    site_mask = (var_output['site_name'] == site_name)
    site_data = var_output[site_mask]
    for col_name in var_output.columns:
        if 'obs' in col_name: # or 'model_' in col_name:
            site_data.loc[:, col_name] = conduct_quality_control(col_name, site_data[col_name].values, zscore_threshold, gap_fill)
    return site_data

def parallel_quality_control(var_name, site_names, zscore_threshold=4, gap_fill='nan'):

    """
    Runs quality control for multiple sites in parallel.
    """

    logging.basicConfig(level=logging.INFO)

    # Read the variables
    var_output = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites.csv', na_values=[''])

    # Create a pool of workers
    with mp.Pool() as pool:
        results = pool.starmap(quality_control_for_site, [(site_name, var_output, zscore_threshold, gap_fill) for site_name in site_names])

    print('results', results)

    # Merge modified data back into var_output
    for result in results:
        var_output.update(result)

    print('var_output', var_output)

    # Save the results
    var_output.to_csv(f'./txt/process1_output/{var_name}_all_sites_filter_{zscore_threshold}sigma_check.csv')

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_met_path   = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    PLUMBER2_flux_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_path       = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    PLUMBER2_path_input = "/g/data/w97/mm3972/data/PLUMBER2/"

    # The site names
    all_site_path     = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    site_names        = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    # site_names      = ["AU-How","AU-Tum"]

    var_name          = 'Qle' #'Qle'
    zscore_threshold  = 3
    gap_fill          = 'nan'
    # quality_control_process1_output(var_name, site_names, zscore_threshold=zscore_threshold)
    # parallel_quality_control(var_name, site_names, zscore_threshold=zscore_threshold, gap_fill=gap_fill)

    write_qc_to_csv(PLUMBER2_flux_path, site_names)
