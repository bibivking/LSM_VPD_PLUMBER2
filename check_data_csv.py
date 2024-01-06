import os
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta 

file_name  = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/VPD_curve/standardized_by_daily_obs_mean_clarify_site/Qle_VPD_standardized_by_daily_obs_mean_clarify_site_error_type=bootstrap_EF_model_0-0.2_bin_by_vpd_coarse.csv'
column_name= 'MATSIRO_vals'
var_output = pd.read_csv(file_name,usecols=[column_name])
print('column_name',column_name)
print('var_output',var_output)