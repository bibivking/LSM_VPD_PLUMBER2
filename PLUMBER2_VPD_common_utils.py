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

def check_variable_exists(PLUMBER2_path, site_name, model_names, key_word, key_word_not=None):

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

def set_model_colors():

    # file path
    model_colors = {"obs":'black',
                    "ACASA": 'darkorange',
                    "CABLE":'red',
                    "CABLE-POP-CN":'gold',
                    "CHTESSEL_Ref_exp1":'dodgerblue',
                    "CHTESSEL_ERA5_3":'lightgreen',
                    "CLM5a":"blue",
                    "GFDL":"yellowgreen",
                    "JULES_GL9":"forestgreen",
                    "LPJ-GUESS":"mediumorchid",
                    "MATSIRO":"violet",
                    "MuSICA":"moccasin",
                    "NASAEnt":"peru",
                    "NoahMPv401":"tan",
                    "ORCHIDEE_tag2.1":"teal",
                    "ORCHIDEE_tag3_2":"lightblue",
                    "QUINCY":"firebrick",
                    "SDGVM":"slategrey",
                    "STEMMUS-SCOPE":"navy",
                    "6km729":"forestgreen",
                    "6km729lag":"palevioletred",
                    "RF":"pink",
                    "3km27":"cyan",
                    "LSTM_raw":"lightseagreen",
                      } 

    return model_colors