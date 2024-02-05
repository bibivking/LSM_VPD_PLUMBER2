#!/usr/bin/env python

"""
Note that VPD still has issue
"""

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



# Path of PLUMBER 2 dataset
PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"
PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
SM_names, soil_thicknesses = get_model_soil_moisture_info()

# The name of models
model_names   = [   "1lin","3km27", "6km729","6km729lag",
                    "ACASA", "CABLE", "CABLE-POP-CN",
                    "CHTESSEL_ERA5_3","CHTESSEL_Ref_exp1","CLM5a",
                    "GFDL","JULES_GL9_withLAI","JULES_test",
                    "LPJ-GUESS","LSTM_eb","LSTM_raw","Manabe",
                    "ManabeV2","MATSIRO","MuSICA","NASAEnt",
                    "NoahMPv401","ORC2_r6593" ,  "ORC2_r6593_CO2",
                    "ORC3_r7245_NEE", "ORC3_r8120","PenmanMonteith",
                    "QUINCY", "RF_eb","RF_raw","SDGVM","STEMMUS-SCOPE"] #"BEPS"

# The site names
all_site_path  = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
site_names     = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
site_names     = ['AR-SLu']
var_name       = 'Rnet'
for site_name in site_names:
    print('site_name',site_name)

    var_dict = check_variable_exists(PLUMBER2_path, var_name, site_name, model_names)
    print('var_dict',var_dict)