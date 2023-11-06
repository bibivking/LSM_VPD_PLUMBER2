import os
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *

def read_CMIP6_data(site_name, file_names, scenarios, var_names, lat=None, lon=None):

    # select the site information from each CMIP6 file
    for scenario in scenarios:
        file_names_scenario = file_names[scenario]
        for var_name in var_names:
            file_names_scenario_variable = file_names_scenario[var_name]
            for file_name in file_names_scenario_variable:

                # Get model name
                f         = nc.Dataset(file_name, mode='r')
                var_tmp   = f.variables[var_name][:]
                try:
                    latitude  = f.variables['lat'][:]
                    longitude = f.variables['lon'][:]
                except:
                    latitude  = f.variables['latitude'][:]
                    longitude = f.variables['longitude'][:]

                lat_idx = np.argmin(np.abs(latitude - lat))
                lon_idx = np.argmin(np.abs(longitude - lon))
                var     = var_tmp[time_mask, lat_idx, lon_idx]


# save nc file
    """
    Read the climate_class value of the nearest pixel to lat and lon.

    Parameters:
        nc_file (str): The path to the netCDF4 file.
        lat (float): The latitude of the site.
        lon (float): The longitude of the site.

    Returns:
        int: The climate_class value of the nearest pixel.
    """
    climate_class_path = '/g/data/w97/mm3972/data/Köppen-Geiger_climate_classification/Beck_KG_V1/Beck_KG_V1_present_0p0083.nc'
    f                  = nc.Dataset(climate_class_path)

    latitude  = f.variables['latitude'][:]
    longitude = f.variables['longitude'][:]
    # print('latitude',latitude)
    # print('longitude',longitude)
    # Find the indices of the nearest pixels to lat and lon.
    lat_idx = np.argmin(np.abs(latitude - lat))
    lon_idx = np.argmin(np.abs(longitude - lon))

    # Read the climate_class value of the nearest pixel.
    # print('lat_idx, lon_idx',lat_idx, lon_idx)
    climate_class = f.variables['climate_class'][lat_idx, lon_idx]
    class_name = f.variables['class_name'][:]
    # print('climate_class',climate_class)
    # print('class_name',class_name[int(climate_class)-1])

    return class_name[int(climate_class)-1]



def calculate_VPD_EF():

    # calculate VPD and EF

if __name__ == "__main__":

    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/amu561/CMIP6_for_Mengyuan/Processed_CMIP6_data/"
    scenarios         = ['historical','ssp126','ssp245','ssp585']
    var_names         = ['hfls','hfss','hurs','tas']


    # The site names
    all_site_path     = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    site_names        = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]


    # Read variable attributions info from input
    lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_met_path)
    # get file names
    file_names           = {}
    file_names_scenario  = {}

    for scenario in scenarios:
        for var_name in var_names:
            file_names_scenario[var_name] = sorted(glob.glob(CMIP6_data_path+scenario+"/*/*.nc"))
        file_names[scenario] = file_names_scenario

    for site_name in site_names:
        # get site lat and lon
        lat, lon = get_lat_lon()

        # read CMIP6 data
        read_CMIP6_data(site_name, file_names, scenarios, var_names, lat=lat_dict[site_name], lon=lon_dict[site_name])
