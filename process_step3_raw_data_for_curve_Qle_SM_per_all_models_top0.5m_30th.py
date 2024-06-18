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
from process_step3_raw_data_for_curve import *

# Path of PLUMBER 2 dataset
PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
site_names, IGBP_types, clim_types, model_names = load_default_list()

# set cause 
var_name       = "Qle"
selected_by    = "SM_per_all_models"
standardize    = None
time_scale     = "hourly"
add_Xday_mean_EF=None

low_bound      =[0,30]
high_bound     =[70,100]

add_LAI        =True
add_qc         =True
add_SMtopXm    ="0.5"
add_normalized_SMtopXm = None
add_Rnet_caused_ratio=True
quality_ctrl   =True

region_name    ="global"
veg_fraction   =None
LAI_range      =None
IGBP_type      =None
middle_day     =False
Tair_constrain =None
VPD_sensitive  =False

# default setting
day_time       = True
output_2d_grids_only = False
clarify_site   = {'opt': True,
                        'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6', # rainfall problems
                                        'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1',
                                        'AU-Wrr','CN-Din','US-WCr','ZM-Mon' # models miss the simulations of them
                                        ]}
models_calc_LAI   = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
energy_cor        = False
country_code      = None #'AU'

if country_code != None:
    site_names = load_sites_in_country_list(country_code)

# whether only provide 2d_grid csv data and stop the script
if region_name == 'global':
    region = {'name':'global', 'lat':None, 'lon':None}
    regional_sites   = None
elif region_name == 'east_AU':
    region = {'name':'east_AU', 'lat':[-44.5,-10], 'lon':[129,155]}
    regional_sites   = get_regional_site_list(region)
elif region_name == 'west_EU':
    region = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
    regional_sites   = get_regional_site_list(region)
elif region_name == 'north_Am':
    region = {'name':'north_Am', 'lat':[25,52], 'lon':[-125,-65]}
    regional_sites   = get_regional_site_list(region)

write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
                high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
                models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
                add_SMtopXm=add_SMtopXm, add_normalized_SMtopXm=add_normalized_SMtopXm,
                add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,  IGBP_type=IGBP_type,
                country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
                energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
                middle_day=middle_day, VPD_sensitive=VPD_sensitive, Tair_constrain=Tair_constrain,
                add_Xday_mean_EF=add_Xday_mean_EF)

