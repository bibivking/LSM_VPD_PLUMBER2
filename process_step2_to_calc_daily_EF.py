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
from copy import deepcopy

def time_step_2_Xday(model_names, X_day=1):

    try:
        # Read Qle_input and Qh_input
        Qle_input = pd.read_csv('./txt/process1_output/Qle_all_sites.csv',
                                na_values=['-9999'],
                                usecols=['time', 'month', 'site_name', 'model_CABLE', 'model_CABLE-POP-CN',
                                         'model_CHTESSEL_Ref_exp1', 'model_CLM5a', 'model_GFDL',
                                         'model_JULES_GL9', 'model_JULES_GL9_withLAI', 'model_MATSIRO',
                                         'model_MuSICA', 'model_NASAEnt', 'model_NoahMPv401', 'model_ORC2_r6593',
                                         'model_ORC3_r8120', 'model_QUINCY', 'model_STEMMUS-SCOPE', 'obs'])

        Qh_input = pd.read_csv('./txt/process1_output/Qh_all_sites.csv',
                               na_values=['-9999'],
                               usecols=['time', 'month', 'site_name', 'model_CABLE', 'model_CABLE-POP-CN',
                                        'model_CHTESSEL_Ref_exp1', 'model_CLM5a', 'model_GFDL',
                                        'model_JULES_GL9', 'model_JULES_GL9_withLAI', 'model_MATSIRO',
                                        'model_MuSICA', 'model_NASAEnt', 'model_NoahMPv401', 'model_ORC2_r6593',
                                        'model_ORC3_r8120', 'model_QUINCY', 'model_STEMMUS-SCOPE', 'obs'])

        # Check for 'time' column
        if 'time' not in Qle_input.columns or 'time' not in Qh_input.columns:
            raise ValueError("The input files do not contain a 'time' column")

        # Extract 'year' and 'day' from 'time' column
        Qle_input['year'] = Qle_input['time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year)
        Qle_input['day']  = Qle_input['time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day)
        Qh_input['year']  = Qle_input['year'][:]
        Qh_input['day']   = Qle_input['day'][:]

        # print(Qle_input['year'], Qle_input['day'])

        # Replace -9999 with NaN
        Qle_input.replace(-9999, np.nan, inplace=True)
        Qh_input.replace(-9999, np.nan, inplace=True)

        # Drop 'time' column
        Qle_input.drop(columns=['time'], inplace=True)
        Qh_input.drop(columns=['time'], inplace=True)

        # Groupby and calculate mean
        if not {'month', 'site_name'}.issubset(Qle_input.columns):
            raise ValueError("The input file must contain 'month' and 'site_name' columns for groupby operation")

        daily_Qle = Qle_input.groupby(['year', 'month', 'day', 'site_name']).mean()
        daily_Qh  = Qh_input.groupby(['year', 'month', 'day', 'site_name']).mean()
        
        daily_Qle = daily_Qle.reset_index()
        daily_Qh  = daily_Qh.reset_index()

        daily_EF          = deepcopy(daily_Qle)
        daily_EF_smoothed = deepcopy(daily_Qle)

        for model_name in model_names['model_select_new']:
            if model_name == 'obs':
                head = ''
            else:
                head = 'model_'
                
            daily_EF.loc[:, head + model_name] = daily_Qle[head + model_name]/(daily_Qle[head + model_name]+daily_Qh[head + model_name])

            for site_name in site_names:
                site_mask = (daily_EF_smoothed['site_name']==site_name)

                # Calculate 5-day rolling mean of efficiency factor grouped by ['year', 'month', 'day', 'site_name']
                daily_EF_smoothed.loc[site_mask, head + model_name] = daily_EF.loc[site_mask, head + model_name].rolling(window=X_day, min_periods=1).mean() 

        # Drop unnecessary columns from Qh_input
        Qle_input.drop(columns=['model_CABLE', 'model_CABLE-POP-CN', 'model_CHTESSEL_Ref_exp1', 'model_CLM5a',
                               'model_GFDL', 'model_JULES_GL9', 'model_JULES_GL9_withLAI', 'model_MATSIRO',
                               'model_MuSICA', 'model_NASAEnt', 'model_NoahMPv401', 'model_ORC2_r6593',
                               'model_ORC3_r8120', 'model_QUINCY', 'model_STEMMUS-SCOPE', 'obs'], inplace=True)

        # Merge var_output back to Qh_input
        var_output = pd.merge(Qle_input, daily_EF_smoothed, on=['year', 'month', 'day', 'site_name'], how='left')

        # Save the processed data to a new CSV file
        var_output.to_csv(f'./txt/process2_output/EF_all_sites_{X_day}_day_mean.csv', index=False)

        print(var_output)

    except Exception as e:
        print(f"Error occurred: {str(e)}")

    return

if __name__ == "__main__":

    site_names, IGBP_types, clim_types, model_names = load_default_list()
    X_day = 1
    time_step_2_Xday(model_names, site_names, X_day)
