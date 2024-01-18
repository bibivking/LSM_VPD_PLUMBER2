'''
Including
    def time_step_2_daily
    def update_EF_in_Qle
    def time_step_2_daily_LAI
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

def time_step_2_daily(var_name):
    var_input    = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites.csv',
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

    # drop the column 'time' since it will cause groupby error     
    var_input = var_input.drop(labels=['time'], axis=1)  

    ### CAUTION: EF should be re-calculated, when making nc file,
    ###          I filter out some EF values which make no sense,
    ###          which may make the below EF daily average biased.
    ###          Groupby ignores np.nan values to calculate 
    ###          daily average Qle & Qh, but I want to set the day
    ###          with nan timestep as nan. To achieve it, I replace  
    ###          nan with inf, groupby the data and then change back 
    ###          to nan. This method will make the days with any nan 
    ###          time step become nan.
        
    # Change nan value to np.inf
    var_input = var_input.fillna(np.inf)
    
    # Groupby
    # if var_name == 'Qle' or var_name == 'Qh' or var_name == 'LAI' :
    var_out = var_input.groupby(by=['year','month','day','site_name','IGBP_type','climate_type']).mean()
    # elif var_name == 'GPP' or var_name == 'NEE' or var_name == 'TVeg':
    #     var_out = var_input.groupby(by=['year','day','site_name','IGBP_type','climate_type']).sum()

    # Change back np.inf to nan value 
    var_out = var_out.replace(np.inf, np.nan)

    # Reset index 
    var_output   = var_out.reset_index(level=['year','month','day','site_name','IGBP_type','climate_type'])
    print(var_output)

    # Save data
    var_output.to_csv(f'./txt/process2_output/daily/{var_name}_all_sites_daily.csv')

    return

def update_EF_in_Qle():

    # Reading qle and qh files 
    var_input_Qle  = pd.read_csv(f'./txt/process2_output/daily/Qle_all_sites_daily.csv',
                     na_values=[''])

    var_input_Qh   = pd.read_csv(f'./txt/process2_output/daily/Qh_all_sites_daily.csv',
                     na_values=[''])

    # Get model name list 
    # how to get the model out list from the column names???
    model_out_list = []
    for column_name in var_input_Qle.columns:
        if "_EF" in column_name:
            model_out_list.append(column_name.split("_EF")[0])

    print('model_out_list',model_out_list)

    # Set daily Qle+Qh threshold
    Qle_Qh_threshold=10
    
    # Calculate daily EF
    for model_in in model_out_list:
        if model_in == 'obs':
            header = ''
        else:
            header = 'model_'
        Qh  = var_input_Qh[header+model_in]
        Qle = var_input_Qle[header+model_in]
        model_EF_tmp = np.where(np.all([Qh+Qle > Qle_Qh_threshold, Qh>0],axis=0), Qle/(Qh+Qle), np.nan)
        var_input_Qle[model_in+'_EF'] = np.where(model_EF_tmp<0, np.nan, model_EF_tmp)

    # Save data
    var_input_Qle.to_csv(f'./txt/process2_output/daily/Qle_all_sites_daily.csv')

    return 

def time_step_2_daily_LAI():
    
    var_input    = pd.read_csv(f'./txt/process1_output/LAI_all_sites.csv',
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

    # drop the column 'time' since it will cause groupby error     
    var_input = var_input.drop(labels=['time'], axis=1)  

    ### CAUTION: EF should be re-calculated, when making nc file,
    ###          I filter out some EF values which make no sense,
    ###          which may make the below EF daily average biased.
    ###          Groupby ignores np.nan values to calculate 
    ###          daily average Qle & Qh, but I want to set the day
    ###          with nan timestep as nan. To achieve it, I replace  
    ###          nan with inf, groupby the data and then change back 
    ###          to nan. This method will make the days with any nan 
    ###          time step become nan.
        
    # Change nan value to np.inf
    var_input = var_input.fillna(np.inf)
    
    # Groupby
    var_out = var_input.groupby(by=['year','month','day','site_name','IGBP_type','climate_type']).mean()

    # Change back np.inf to nan value 
    var_out = var_out.replace(np.inf, np.nan)

    # Reset index 
    var_output   = var_out.reset_index(level=['year','month','day','site_name','IGBP_type','climate_type'])
    print(var_output)

    # Save data
    var_output.to_csv(f'./txt/process2_output/daily/LAI_all_sites_daily.csv')

if __name__ == "__main__":

    # var_name = 'GPP'  #'TVeg'
    # time_step_2_daily(var_name)

    # var_name = 'Qle'  #'TVeg'
    # time_step_2_daily(var_name)

    # var_name = 'Qh'  #'TVeg'
    # time_step_2_daily(var_name)

    update_EF_in_Qle()


    # time_step_2_daily_LAI()

    # var_name = 'NEE'  #'TVeg'
    # time_step_2_daily(var_name)
