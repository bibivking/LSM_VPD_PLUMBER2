'''
Including

'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (13.03.2024)"
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

def convert_TVeg_units(var_name):

    '''
    Convert from TVeg kg m-2 h-1 to W m-2
    '''

    var_input   = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites.csv', na_values=[''])

    # Convert units ...
    var_input['obs_Tair'] -= 273.15

    print("var_input['obs_Tair'][:10]",var_input['obs_Tair'][:10])

    # latent_heat_vapourisation
    lhv = (2.501 - 0.00237 * var_input['obs_Tair']) * 1E06

    for col_name in var_input.columns:
        if 'model_' in col_name:
            print("var_input[col_name][:10]",var_input[col_name][:10])
            # in process_step1, convert TVeg's unit from kg m-2 s-1 to kg m-2 h-1
            # so it should be divided by 3600 here
            var_input[col_name] = var_input[col_name]/3600*lhv

            print("var_input[col_name][:10]",var_input[col_name][:10])

    var_input.to_csv(f'./txt/process1_output/{var_name}_all_sites_Wm2.csv')

    return

def calc_nonTVeg():

    TVeg_input  = pd.read_csv(f'./txt/process1_output/TVeg_all_sites_Wm2.csv', na_values=[''])
    Qle_input   = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', na_values=[''])

    for col_name in TVeg_input.columns:
        if 'model_' in col_name:
            print("TVeg_input[col_name][:10]",TVeg_input[col_name][:10])
            TVeg_input[col_name] = Qle_input[col_name] - TVeg_input[col_name]
            print("TVeg_input[col_name][:10]",TVeg_input[col_name][:10])

    TVeg_input.to_csv(f'./txt/process1_output/nonTVeg_all_sites_Wm2.csv')

    return


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_met_path   = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    PLUMBER2_path       = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    PLUMBER2_path_input = "/g/data/w97/mm3972/data/PLUMBER2/"

    var_name            = 'TVeg'
    convert_TVeg_units(var_name)

    calc_nonTVeg()
