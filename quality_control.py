import os
import sys
import glob
import netCDF4
import multiprocessing
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

def conduct_quality_control(varname, data_input,zscore_threshold=2):
    
    '''
    Please notice EF has nan values 
    '''
    z_scores    = np.abs(stats.zscore(data_input, nan_policy='omit'))
    data_output = np.where(z_scores > zscore_threshold, np.nan, data_input)

    # print('z_scores',z_scores)
    if 'EF' not in varname:
        print('EF is not in ', varname)
        # Iterate through the data to replace NaN with the average of nearby non-NaN values
        for i in range(1, len(data_output) - 1):
            if np.isnan(data_output[i]):
                prev_index = i - 1
                next_index = i + 1
                
                # find the closest non nan values
                while prev_index >= 0 and np.isnan(data_output[prev_index]):
                    prev_index -= 1
                
                while next_index < len(data_output) and np.isnan(data_output[next_index]):
                    next_index += 1
                
                # use average them 
                if prev_index >= 0 and next_index < len(data_output):
                    prev_non_nan = data_output[prev_index]
                    next_non_nan = data_output[next_index]
                    data_output[i] = (prev_non_nan + next_non_nan) / 2.0

    print('len(z_scores)',len(z_scores))
    # print('data_output',data_output)

    return data_output

def convert_into_kg_m2_s(data_input, var_units):
    
    d_2_s = 24*60*60
    if 'W' in var_units and 'm' in var_units and '2' in var_units:
        print('converting ', var_units)
        data_output = data_input * 86400 / 2454000 /d_2_s
    return data_output