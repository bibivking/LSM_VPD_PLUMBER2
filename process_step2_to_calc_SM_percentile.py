'''
Including

filter Gs
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import gc
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import percentileofscore
from PLUMBER2_VPD_common_utils import *

def calc_SM_percentile_single_site(SM_file, model_names, SM_depth, site_name):

    '''
    1. calculate 48 time-step smoothing mean SM
    2. calculate SM percentile
    '''

    # Read Qle_input and Qh_input
    SM_input = pd.read_csv(SM_file)
    SM_tmp = SM_input[SM_input['site_name'] == site_name]

    for i, model_name in enumerate(model_names):
        if model_name in ['NASAEnt', 'QUINCY', 'obs']:
            model_name_new = 'model_mean'
        else:
            model_name_new = model_name

        SM_model = SM_tmp[model_name_new + '_SMtop' + SM_depth + 'm'].rolling(window=48, min_periods=1).mean().values
        tmp = np.array([percentileofscore(SM_model, SM, kind='rank') for SM in SM_model])

        if i == 0:
            SM_percentile = pd.DataFrame(tmp, columns=[model_name])
        else:
            SM_percentile[model_name] = tmp

    # Checks if a folder exists and creates it if it doesn't
    output_dir = f'./txt/process2_output/SMtop{SM_depth}m_percentile_sites'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the processed data to a new CSV file
    SM_percentile.to_csv(f'{output_dir}/SMtop{SM_depth}m_percentile_{site_name}.csv', index=False)

    return

def calc_SM_percentile_parallel(SM_file, model_names, SM_depth, site_names):

    # with mp.Pool() as pool:
    #     pool.starmap(calc_SM_percentile_single_site, [(SM_file, model_names, SM_depth, site_name) for site_name in site_names[160:170]])

    var_input = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', na_values=[''], usecols=['time', 'site_name'])

    for model_name in model_names:
        if model_name == 'obs':
            header = ''
        else:
            header = 'model_'
        var_input[header + model_name] = np.nan

    for site_name in site_names:
        print('site ', site_name)
        site_mask = (var_input['site_name'] == site_name)

        file_input = f'./txt/process2_output/SMtop{SM_depth}m_percentile_sites/SMtop{SM_depth}m_percentile_{site_name}.csv'

        if os.path.exists(file_input):
            sm_input = pd.read_csv(file_input, na_values=[''])

            for model_name in model_names:
                print('model', model_name)

                if model_name == 'obs':
                    header = ''
                else:
                    header = 'model_'

                try:
                    var_input.loc[site_mask, header + model_name] = sm_input[model_name].values
                except KeyError:
                    print(f'Missing some of sm_input["{model_name}"]')

                gc.collect()
        else:
            print(file_input, 'does not exist')

    var_input.to_csv(f'./txt/process2_output/SMtop{SM_depth}m_percentile_all_sites.csv', index=False)

    return

if __name__ == "__main__":
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    SM_depth = '0.3'
    SM_file  = f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/SMtop{SM_depth}m_all_sites.csv"

    # site_names = [ 'US-NR1'] # 'IT-Mal', 'RU-Fyo', 'US-AR2', 'US-Bar', 'US-Los',

    calc_SM_percentile_parallel(SM_file, model_names['model_select_new'], SM_depth, site_names)
