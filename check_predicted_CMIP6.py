import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta 

# file_name  = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/predicted_CMIP6_DT_Qle_historical_ACCESS-CM2_CABLE_global_Poisson.csv'

# var_output = pd.read_csv(file_name)
# var_tmp    = np.where(abs(var_output['CABLE'].values) > 10000., var_output['CABLE'].values, np.nan)

# print(var_tmp[~np.isnan(var_tmp)])

file_path = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6/'

file_list = [   'EC-Earth3_global.csv',
                'MPI-ESM1-2-HR_global.csv',
                'CMCC-CM2-SR5_global.csv',
                'CMCC-ESM2_global.csv',  
                'BCC-CSM2-MR_global.csv',
                'MRI-ESM2-0_global.csv',    
                'MIROC6_global.csv',
                'ACCESS-CM2_global.csv',
                'KACE-1-0-G_global.csv',
                'MPI-ESM1-2-LR_global.csv',
                'MIROC-ES2L_global.csv',]

                #'CMIP6_DT_Qle_ssp245_global.csv',]
                # 'CMIP6_DT_Qle_historical_global.csv',]
# Check the VPD ranges 

# #  ===== Checking the max & min VPD in CMIP6 data =====
# scenario = 'historical'

# for i, file_name in enumerate(file_list):

#     var_output = pd.read_csv(f'{file_path}CMIP6_DT_Qle_{scenario}_{file_name}', usecols=['VPD'])
    
#     print('file_name', file_name, 'max(var_output)', max(var_output.VPD), 'min(var_output)', min(var_output.VPD))
#     gc.collect()


#  ===== Checking the percentage filtered out in CMIP6 data =====

# scenario = 'historical'

# for i, file_name in enumerate(file_list):

#     var_output = pd.read_csv(f'{file_path}CMIP6_DT_Qle_{scenario}_{file_name}', usecols=['VPD'])
#     var_filter = pd.read_csv(f'{file_path}CMIP6_DT_filtered_by_VPD_Qle_{scenario}_{file_name}', usecols=['VPD'])
    
#     print(file_name, 'filter out ',(1-len(var_filter)/len(var_output))*100, "%")
#     gc.collect()

#  ===== Checking the percentage each bin in CMIP6 data =====
scenario = 'ssp245' #'historical'

for i, file_name in enumerate(file_list):

    var_output  = pd.read_csv(f'{file_path}CMIP6_DT_Qle_{scenario}_{file_name}', usecols=['VPD'])
    tot_num     = len(var_output) 
    num_less00001 = np.sum(var_output['VPD'].values <0.0001)
    num_less0001 = np.sum(var_output['VPD'].values <0.001)
    num_less001 = np.sum(var_output['VPD'].values <0.01)
    num_less005 = np.sum(var_output['VPD'].values <0.05)
    num_7_10   = np.sum((var_output['VPD'].values <10) & (var_output['VPD'].values >7))
    num_10_20   = np.sum((var_output['VPD'].values <=20) & (var_output['VPD'].values >10))
    num_more10  = np.sum(var_output['VPD'].values >10)

    print(file_name, 'VPD < 0.0001 account for',(num_less00001/tot_num)*100, "%")
    print('           VPD < 0.001 account for',(num_less0001/tot_num)*100, "%")
    print('           VPD < 0.01 account for',(num_less001/tot_num)*100, "%")
    print('           VPD < 0.05 account for',(num_less005/tot_num)*100, "%")
    print('           7 < VPD < 10 account for',(num_7_10/tot_num)*100, "%")
    print('           10 < VPD <= 20 account for',(num_10_20/tot_num)*100, "%")
    print('           VPD > 10 account for',(num_more10/tot_num)*100, "%")
    
    gc.collect()
