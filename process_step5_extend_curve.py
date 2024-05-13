'''
Bin the dataset by VPD (and EF) and save in process4_output
Including:
    def bin_VPD
    def bin_VPD_EF
    def write_var_VPD
    def write_var_VPD_EF
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
import multiprocessing as mp
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *

def calc_slope_intercept(var_name, model_list, selected_by=None, bounds=None,
                  day_time=False, IGBP_type=None, time_scale=None,
                  clim_type=None, energy_cor=False, LAI_range=None,
                  uncertain_type='UCRTN_percentile', veg_fraction=None,
                  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                  country_code=None, method='CRV_bins',dist_type='Linear'):

    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                uncertain_type=uncertain_type, clarify_site=clarify_site)

    var_input = pd.read_csv(f'./txt/process4_output/{folder_name}/fit_GammaGAM/{var_name}{file_message}_{dist_type}.csv',
                            usecols=['vpd_series','CABLE_vals','CABLE-POP-CN_vals', 'CHTESSEL_Ref_exp1_vals',
                                     'CLM5a_vals','GFDL_vals','JULES_GL9_vals','JULES_GL9_withLAI_vals',
                                     'MATSIRO_vals','MuSICA_vals','NASAEnt_vals','NoahMPv401_vals',
                                     'ORC2_r6593_vals','ORC3_r8120_vals','QUINCY_vals','STEMMUS-SCOPE_vals',
                                     'obs_vals'])

    vpd_series  = var_input['vpd_series'].values 
    vpd_last    = vpd_series[-1]

    for model_in in model_list:
        vals    = var_input[model_in+"_vals"].values
        val_max = np.max(vals)
        vpd_max = vpd_series[np.argmax(vals)]
        val_last= vals[-1]

        slope     = (val_last-val_max)/(vpd_last-vpd_max)
        intercept = val_last
        print(model_in,'val_max', val_max, 'val_last', val_last, 
              'vpd_max', vpd_max, 'vpd_last', vpd_last,
              'slope',slope, 'intercept',intercept)
    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # ======================= Default setting (dont change) =======================
    var_name       = 'Qle'       #'TVeg'
    time_scale     = 'hourly'   #'daily'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'
    method         = 'CRV_bins' # 'CRV_bins'
                                # 'CRV_fit_GAM_simple'
                                # 'CRV_fit_GAM_complex'
    standardize    = None       # 'None'
                                # 'STD_LAI'
                                # 'STD_annual_obs'
                                # 'STD_monthly_obs'
                                # 'STD_monthly_model'
                                # 'STD_daily_obs'
    LAI_range      = None
    veg_fraction   = None   #[0.7,1]

    clarify_site      = {'opt': True,
                         'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                         'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

    day_time       = False  # False for daily
                            # True for half-hour or hourly

    if time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # ====================== Custom setting ========================
    var_name       = 'Qle'
    uncertain_type = 'UCRTN_bootstrap'
    selected_by    = 'EF_model'
    method         = 'CRV_fit_GAM_complex'
    # method         = 'CRV_bins'
    dist_type      = 'Gamma' # None #'Linear' #'Poisson' # 'Gamma'

    # Get model lists
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list = model_names['model_select_new']

    IGBP_type = None
    LAI_range = None
    clim_type = None

    # 0 < EF < 0.2
    bounds         = [0,0.2] #30
    calc_slope_intercept(var_name, model_list, selected_by=selected_by, bounds=bounds,
                    day_time=day_time, IGBP_type=IGBP_type, time_scale=time_scale, 
                    clim_type=clim_type, energy_cor=energy_cor, LAI_range=LAI_range,
                    uncertain_type=uncertain_type, veg_fraction=veg_fraction, 
                    clarify_site=clarify_site, standardize=standardize,
                    country_code=country_code, method=method,dist_type=dist_type)