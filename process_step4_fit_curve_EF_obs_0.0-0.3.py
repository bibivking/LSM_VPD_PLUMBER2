#!/usr/bin/env python

import os
import gc
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from quality_control import *
from PLUMBER2_VPD_common_utils import *
from process_step4_fit_curve import *

# Path of PLUMBER 2 dataset
PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
site_names, IGBP_types, clim_types, model_names = load_default_list()

# ======================= Default setting (dont change) =======================
var_name       = "Qle"
var_name2      = "Qle"
bounds         = [0.0,0.3]
time_scale     = "hourly"
selected_by    = "EF_obs"
method         = "CRV_bins"
standardize    = None
add_Xday_mean_EF=None

dist_type      = None
VPD_num_threshold = 200
vpd_top_type   = "sample_larger_200"

uncertain_type = "UCRTN_bootstrap"
day_time       = True
IGBP_type      = None
LAI_range      = None
veg_fraction   = None
middle_day     = False

# default setting
energy_cor     = False
country_code   = None
clarify_site   = {'opt': True,
                        'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6', # rainfall problems
                                        'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1',
                                        'AU-Wrr','CN-Din','US-WCr','ZM-Mon' # models miss the simulations of them
                                        ]}
models_calc_LAI = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
if middle_day:
    message_midday = '_midday'
else:
    message_midday = ''

folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                            standardize=standardize, country_code=country_code,
                            selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                            LAI_range=LAI_range, clarify_site=clarify_site, add_Xday_mean_EF=add_Xday_mean_EF) #
file_input     = 'raw_data_'+var_name+'_VPD'+file_message+message_midday+'.csv'

write_var_VPD_parallel(var_name2, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
                            bounds=bounds, day_time=day_time, clarify_site=clarify_site, VPD_num_threshold=VPD_num_threshold,
                            standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type, vpd_top_type=vpd_top_type,
                            models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range, middle_day=middle_day,
                            country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
gc.collect()


