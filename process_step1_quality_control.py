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

def quality_control_process1_output(var_name, site_names, zscore_threshold=4):

    # read the variables
    var_output   = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites.csv', na_values=[''])

    for site_name in site_names:
        print('site',site_name)
        site_mask = (var_output['site_name'][:] == site_name)
        for col_name in var_output.columns:
            if 'obs' in col_name or 'model_' in col_name:
                var_output.loc[site_mask, col_name] = \
                    conduct_quality_control(col_name, var_output[site_mask][col_name].values, zscore_threshold)

    var_output.to_csv(f'./txt/process1_output/{var_name}_all_sites_filter_{zscore_threshold}sigma.csv')


def quality_control_for_site(site_name, var_output, zscore_threshold):

    """
    Performs quality control for a single site.
    """

    site_mask = (var_output['site_name'][:] == site_name)
    site_data = var_output[site_mask]
    for col_name in var_output.columns:
        if 'obs' in col_name or 'model_' in col_name:
            site_data.loc[:, col_name] = conduct_quality_control(col_name, site_data[col_name].values, zscore_threshold)
    return site_data

def parallel_quality_control(var_name, site_names, zscore_threshold=4):

    """
    Runs quality control for multiple sites in parallel.
    """

    # Read the variables
    var_output = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites.csv', na_values=[''])

    # Create a pool of workers
    with mp.Pool() as pool:
        results = pool.starmap(quality_control_for_site, [(site_name, var_output, zscore_threshold) for site_name in site_names])

    print('results', results)

    # Merge modified data back into var_output
    for result in results:
        var_output.update(result)

    print('var_output', var_output)

    # Save the results
    var_output.to_csv(f'./txt/process1_output/{var_name}_all_sites_filter_{zscore_threshold}sigma.csv')

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_met_path   = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    PLUMBER2_path       = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    PLUMBER2_path_input = "/g/data/w97/mm3972/data/PLUMBER2/"

    # The site names
    all_site_path     = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    site_names        = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    # site_names      = ["AU-How","AU-Tum"]

    var_name          = 'Qle' #'Qle'
    zscore_threshold  = 4
    # quality_control_process1_output(var_name, site_names, zscore_threshold=zscore_threshold)
    parallel_quality_control(var_name, site_names, zscore_threshold=zscore_threshold)
