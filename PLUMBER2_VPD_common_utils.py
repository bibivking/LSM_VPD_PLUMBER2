import os
import re
import sys
import glob
import netCDF4 as nc
import codecs
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

def check_variable_exists(PLUMBER2_path, varname, site_name, model_names, key_word, key_word_not=None):

    # file path
    my_dict      = {} 

    for j, model_name in enumerate(model_names):
        # print(model_name)

        # Set input file path
        file_path    = glob.glob(PLUMBER2_path+model_name +"/*"+site_name+"*.nc")
        var_exist    = False
        try:
            with nc.Dataset(file_path[0], 'r') as dataset:
                for var_name in dataset.variables:
                    # print(var_name)
                    if varname.lower() in var_name.lower():
                        my_dict[model_name] = var_name
                        var_exist = True
                    else:
                        variable  = dataset.variables[var_name]

                        # Check whether long_name exists
                        if hasattr(variable, 'long_name'):
                            long_name = variable.long_name.lower()  # Convert description to lowercase for case-insensitive search

                            # Check whether key_word exists
                            # make sure key_word in long_name and all key_word_not are not in key_word_not
                            if key_word in long_name and all(not re.search(key_not, long_name) for key_not in key_word_not):
                                # print(long_name)
                                my_dict[model_name] = var_name
                                var_exist = True
                                # print(f"The word '{key_word}' is in the description of variable '{var_name}'.")
                                break  # Exit the loop once a variable is found
                    

        except Exception as e:
            print(f"An error occurred: {e}")

        # variable doesn't exist
        if not var_exist:
            my_dict[model_name] = 'None'
    # print(my_dict)
    # f = open(f"./txt/{site_name}_{key_word}.txt", "w")
    # f.write(str(my_dict))
    # f.close()
    return my_dict

def read_lat_lon(site_names, PLUMBER2_met_path):

    # file path
    lat_dict      = {} 
    lon_dict      = {} 

    for site_name in site_names:


        # Set input file path
        file_path    = glob.glob(PLUMBER2_met_path+"/*"+site_name+"*.nc")

        f            = nc.Dataset(file_path[0], mode='r')
        lat_tmp      = f.variables['latitude'][0,0].data
        lon_tmp      = f.variables['longitude'][0,0].data
        
        # convert the array to a floating-point number
        lat_dict[site_name] = float(lat_tmp)
        lon_dict[site_name] = float(lon_tmp)
        f.close()

    # print('lat_dict',lat_dict)
    # print('lon_dict',lon_dict)

    return lat_dict, lon_dict

def read_IGBP_veg_type(site_names, PLUMBER2_met_path):

    # file path
    IGBP_dict      = {} 

    for site_name in site_names:

        # print(site_name)

        # Set input file path
        file_path    = glob.glob(PLUMBER2_met_path+"/*"+site_name+"*.nc")

        f            = nc.Dataset(file_path[0], mode='r')

        # Decode the string as Unicode
        IGBP_tmp     = codecs.decode(f.variables['IGBP_veg_short'][:].data, 'utf-8') 

        # Remove spaces from the beginning and end of the string
        IGBP_dict[site_name] = IGBP_tmp.strip()

        f.close()

    # print('IGBP_dict',IGBP_dict)

    return IGBP_dict

def read_climate_class(lat, lon):

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
    print('latitude',latitude)
    print('longitude',longitude)
    # Find the indices of the nearest pixels to lat and lon.
    lat_idx = np.argmin(np.abs(latitude - lat))
    lon_idx = np.argmin(np.abs(longitude - lon))

    # Read the climate_class value of the nearest pixel.
    print('lat_idx, lon_idx',lat_idx, lon_idx)
    climate_class = f.variables['climate_class'][lat_idx, lon_idx]
    class_name = f.variables['class_name'][:]
    print('climate_class',climate_class)
    print('class_name',class_name[int(climate_class)-1])

    return class_name[int(climate_class)-1]

def set_model_colors():

    # file path
    model_colors = {
                    "obs": 'black',
                    "1lin": 'lightcoral' ,
                    "3km27": 'indianred', 
                    "6km729": 'brown',
                    "6km729lag":'red',
                    "LSTM_eb": 'lightsalmon',
                    "LSTM_raw": 'rosybrown',
                    "RF_eb": 'sienna',
                    "RF_raw": 'tomato',
                    "ACASA": 'peru',
                    "CABLE": 'gold',
                    "CABLE-POP-CN": 'orange',
                    "CHTESSEL_ERA5_3": "olive",
                    "CHTESSEL_Ref_exp1": "olivedrab",
                    "CLM5a":"darkkhaki",
                    "GFDL": "yellowgreen",
                    "JULES_GL9_withLAI": "limegree", 
                    "JULES_test": "forestgreen",
                    "LPJ-GUESS": "turquoise",
                    "Manabe": "lightseagreen",
                    "ManabeV2": "darkcyan",
                    "MATSIRO": "deepskyblue",
                    "MuSICA": "dodgerlblue",
                    "NASAEnt": "blue",
                    "NoahMPv401":"royalblue",
                    "ORC2_r6593": "blueviolet", 
                    "ORC2_r6593_CO2":"violet",
                    "ORC3_r7245_NEE":"fuchsia", 
                    "ORC3_r8120":"orchid",
                    "PenmanMonteith": "purple",
                    "QUINCY": "mediumvioletred",
                    "SDGVM": "deeppink",
                    "STEMMUS-SCOPE": "pink"} 

    return model_colors

