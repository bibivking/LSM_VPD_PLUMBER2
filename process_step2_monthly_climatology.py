'''
Including
    def time_step_2_monthly

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

def time_step_2_monthly(var_name):
    model_lists  = ['model_3km27','model_6km729','model_6km729lag','model_ACASA','model_CABLE',
                    'model_CABLE-POP-CN','model_CHTESSEL_ERA5_3','model_CHTESSEL_Ref_exp1',
                    'model_CLM5a','model_GFDL','model_JULES_GL9_withLAI',
                    'model_JULES_test','model_LSTM_eb','model_LSTM_raw',
                    'model_Manabe','model_ManabeV2','model_MATSIRO','model_MuSICA',
                    'model_NASAEnt','model_NoahMPv401','model_ORC2_r6593','model_ORC2_r6593_CO2',
                    'model_ORC3_r7245_NEE','model_ORC3_r8120','model_PenmanMonteith',
                    'model_QUINCY','model_RF_eb','model_RF_raw','model_STEMMUS-SCOPE','obs']
                    
    var_input    = pd.read_csv(f'./txt/all_sites/{var_name}_all_sites.csv',
                   usecols=['model_3km27','model_6km729','model_6km729lag','model_ACASA','model_CABLE',
                           'model_CABLE-POP-CN','model_CHTESSEL_ERA5_3','model_CHTESSEL_Ref_exp1',
                           'model_CLM5a','model_GFDL','model_JULES_GL9_withLAI',
                           'model_JULES_test','model_LSTM_eb','model_LSTM_raw',
                           'model_Manabe','model_ManabeV2','model_MATSIRO','model_MuSICA',
                           'model_NASAEnt','model_NoahMPv401','model_ORC2_r6593','model_ORC2_r6593_CO2',
                           'model_ORC3_r7245_NEE','model_ORC3_r8120','model_PenmanMonteith',
                           'model_QUINCY','model_RF_eb','model_RF_raw','model_STEMMUS-SCOPE',
                           'obs','month','site_name'],na_values=[''])

    for model_name in model_lists:
        var_input.loc[var_input[model_name] == -9999, model_name] = np.nan
        
    var_out      = var_input.groupby(by=['month','site_name']).mean()
    var_output   = var_out.reset_index(level=['month', 'site_name'])
    var_output.to_csv(f'./txt/all_sites_monthly/{var_name}_all_sites_monthly.csv')

    return


if __name__ == "__main__":

    var_name       = 'GPP'  #'TVeg'

    time_step_2_monthly(var_name)