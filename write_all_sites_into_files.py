import os
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
from plot_script import *

def read_data(var_name, site_name, input_file):

    f              = nc.Dataset(input_file, mode='r')

    model_in_list  = f.variables[var_name + '_models']
    time           = nc.num2date(f.variables['CABLE_time'][:],f.variables['CABLE_time'].units, 
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True)
    ntime          = len(time)
    var_output     = pd.DataFrame(time, columns=['time'])

    model_out_list = []

    # model_in_list  = f.variables['EF_models']

    for model_in in model_in_list:
        # Set the var and time names of the model 
        model_var_name  = f"{model_in}_{var_name}"
        model_time_name = f"{model_in}_time"
        model_ntime     = len(f.variables[model_time_name])

        # If the model has full time series
        if model_ntime == ntime:
            model_out_list.append(model_in)
            if var_name == 'TVeg':
                var_output[model_in] = f.variables[model_var_name][:]*3600
            if var_name == 'NEE':
                # convert from umol/m2/s to g C/h
                s2h                  = 3600.              # s-1 to h-1
                GPP_scale            = -0.000001*12*s2h   # umol s-1 to g C h-1
                var_output[model_in] = f.variables[model_var_name][:]*GPP_scale
            else:
                var_output[model_in] = f.variables[model_var_name][:]
      
            # model_bin_name             = f"{model_in}_EF"
            # var_output[model_in+'_EF'] = f.variables[model_bin_name][:]

    if var_name == 'Qle' or var_name == 'Qh':
        var_output['obs'] = f.variables[f"obs_{var_name}"][:] 
        model_out_list.append('obs')

    var_output['obs_EF'] = f.variables["obs_EF"][:] 
        
    # Read VPD and soil moisture information
    var_output['VPD']      = f.variables['VPD'][:]
    var_output['obs_Tair'] = f.variables['obs_Tair'][:]
    var_output['obs_Qair'] = f.variables['obs_Qair'][:]

    # close the file
    f.close()

    ntime      = len(var_output)
    month      = np.zeros(ntime)
    hour       = np.zeros(ntime)
    # site       = np.full([ntime], site_name.rjust(6), dtype=str)

    for i in np.arange(ntime):
        month[i] = var_output['time'][i].month
        hour[i]  = var_output['time'][i].hour

    var_output['month']     = month
    var_output['hour']      = hour
    var_output['site_name'] = site_name
    
    # return the var values and the model list has the required output
    return var_output, model_out_list

def write_spatial_land_days(var_name, site_names, PLUMBER2_path):

    # ============= read all sites data ================
    # get veg type info
    IGBP_dict = read_IGBP_veg_type(site_names, PLUMBER2_path)

    # get climate type info
    lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_path)
    clim_class_dict    = {}

    for site_name in site_names:
        clim_class_dict[site_name] = read_climate_class(lat_dict[site_name], lon_dict[site_name])
        
    # read data
    for i, site_name in enumerate(site_names):
        # print('site_name',site_name)
        input_file = "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"+site_name+".nc"
        var_output_tmp, model_out_list = read_data(var_name, site_name, input_file)

        # Add veg type
        var_output_tmp['IGBP_type']    = IGBP_dict[site_name]

        # Add climate type
        var_output_tmp['climate_type'] = clim_class_dict[site_name]

        if i == 0:
            var_output         = var_output_tmp
            pre_model_out_list = model_out_list
        else:
            if pre_model_out_list != model_out_list: 
                # Find the elements that are only in pre_model_out_list
                only_in_pre_model_out_list = np.setdiff1d(pre_model_out_list, model_out_list, assume_unique=True)
                print('Elements only in pre_model_out_list:',only_in_pre_model_out_list)
                # Set the missing model simulation as np.nan
                for missed_model in only_in_pre_model_out_list:
                    var_output_tmp[missed_model] = np.nan
                    # var_output_tmp[missed_model+'_EF'] = np.nan

            # connect different sites data together
            var_output = var_output.append(var_output_tmp, ignore_index=True)

        var_output_tmp=None
        
    model_out_list = pre_model_out_list

    # save the dataframe
    var_output.to_csv(f'./txt/{var_name}_all_sites.csv')

    # free memory
    del(var_output)      
    del(IGBP_dict) 
    del(lat_dict) 
    del(clim_class_dict)
    del(lon_dict)


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    # site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    site_names     = ["AU-How"]
    print(site_names)

    var_name       = 'Qle'  
    write_spatial_land_days(var_name, site_names, PLUMBER2_path)



