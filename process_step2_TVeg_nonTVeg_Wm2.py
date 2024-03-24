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

def convert_TVeg_units():

    '''
    Convert from TVeg kg m-2 h-1 to W m-2
    Constrain 0 <= TVeg <= Qle
    '''

    var_input   = pd.read_csv(f'./txt/process1_output/TVeg_all_sites.csv', na_values=[''])
    Qle_input   = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', na_values=[''])

    # Convert Tair units ...
    obs_Tair    = var_input['obs_Tair'] - 273.15
    print("obs_Tair[:10]",obs_Tair[:10])

    # latent_heat_vapourisation
    lhv = (2.501 - 0.00237 * obs_Tair) * 1E06
    print("lhv[:10]",lhv[:10])

    for i, col_name in enumerate(var_input.columns):
        
        if 'model_' in col_name:
            print('col_name is ', col_name)

            # in process_step1, convert TVeg's unit from kg m-2 s-1 to kg m-2 h-1
            # so it should be divided by 3600 here
            tmp = var_input[col_name]/3600*lhv    
            print("tmp[:10] first time ",tmp[:10])

            # set the min of TVeg as 0 
            tmp = np.where(tmp<0, 0, tmp)
            print("tmp[:10] second time ",tmp[:10])

            # set the max of TVeg as Qle
            # var_input[col_name] = np.where((Qle_input[col_name]<tmp), Qle_input[col_name], tmp)
            # If run the script again, please use the line below to avoid when Qle < 0, Trans becomes <0 as well.
            tmp = np.where((Qle_input[col_name]<tmp) & (Qle_input[col_name]>0), Qle_input[col_name], tmp)
            print("tmp[:10] third time ",tmp[:10])

            if i == 0:
                var_output           = pd.DataFrame(tmp, columns=[col_name])
            else:
                var_output[col_name] = tmp
            print("var_output[col_name][:10] ",var_output[col_name][:10])    
        else:
            if i == 0:
                var_output = pd.DataFrame(var_input[col_name].values, columns=[col_name])
            else:
                var_output[col_name] = var_input[col_name].values

    var_output.to_csv(f'./txt/process1_output/TVeg_all_sites_Wm2.csv')

    return

def calc_nonTVeg():

    TVeg_input  = pd.read_csv(f'./txt/process1_output/TVeg_all_sites_Wm2.csv', na_values=[''])
    Qle_input   = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', na_values=[''])

    for i, col_name in enumerate(TVeg_input.columns):
        if 'model_' in col_name:
            print('col_name is',col_name)
            if i == 0:
                var_output = pd.DataFrame((Qle_input[col_name] - TVeg_input[col_name]).values, columns=[col_name])
                print("var_output[col_name][:10]",var_output[col_name][:10])
            else:
                var_output[col_name] = (Qle_input[col_name] - TVeg_input[col_name]).values
                print("var_output[col_name][:10]",var_output[col_name][:10])
        else:
            if i == 0:
                var_output = pd.DataFrame(TVeg_input[col_name].values, columns=[col_name])
            else:
                var_output[col_name] = TVeg_input[col_name].values
    var_output.to_csv(f'./txt/process1_output/nonTVeg_all_sites_Wm2.csv')

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_met_path   = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    PLUMBER2_path       = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    PLUMBER2_path_input = "/g/data/w97/mm3972/data/PLUMBER2/"

    # var_name            = 'TVeg'
    convert_TVeg_units()

    calc_nonTVeg()
