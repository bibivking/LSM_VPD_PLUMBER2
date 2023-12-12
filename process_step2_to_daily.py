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

def time_step_2_daily(var_name):

    var_input    = pd.read_csv(f'./txt/all_sites/{var_name}_all_sites.csv',
                   na_values=[''])
    col_lists    = var_input.columns

    ntime        = len(var_input)
    year         = np.zeros(ntime)
    day          = np.zeros(ntime)
    for i in np.arange(ntime):
        time     = datetime.strptime(var_input['time'][i], "%Y-%m-%d %H:%M:%S")
        year[i]  = time.year
        day[i]   = time.day
    var_input['year'] = year
    var_input['day']  = day

    for col_name in col_lists:
        var_input.loc[var_input[col_name] == -9999, col_name] = np.nan


    ### CAUTION: EF should be re-calculated, when making nc file,
    ###          I filter out some EF values which make no sense,
    ###          which may make the below EF daily average biased.

    if var_name == 'Qle' or var_name == 'Qh' or var_name == 'LAI' :
        var_out = var_input.groupby(by=['year','day','site_name']).mean()
    elif var_name == 'GPP' or var_name == 'NEE' or var_name == 'TVeg':
        var_out = var_input.groupby(by=['year','day','site_name']).sum()

    var_output   = var_out.reset_index(level=['year','day','site_name'])
    var_output.to_csv(f'./txt/all_sites_daily/{var_name}_all_sites_daily.csv')

    return

if __name__ == "__main__":

    var_name = 'GPP'  #'TVeg'
    time_step_2_daily(var_name)

    var_name = 'Qle'  #'TVeg'
    time_step_2_daily(var_name)

    var_name = 'NEE'  #'TVeg'
    time_step_2_daily(var_name)
