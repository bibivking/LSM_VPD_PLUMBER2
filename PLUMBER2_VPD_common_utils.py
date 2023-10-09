import os
import re
import gc
import sys
import glob
import netCDF4 as nc
import codecs
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from datetime import datetime, timedelta


def smooth_vpd_series(values, window_size=11, order=3, type='S-G_filter'):

    """
    Smooth the data using a window with requested size.
    Input:
    array values[nx, ny]

    Type option:
    S-G_filter: Savitzky-Golay filter
    smoothing: mean
    """


    window_half     = int(window_size/2.)

    # if len(np.shape(values))==1:
    nx          = len(values)
    vals_smooth = np.full([nx], np.nan)

    if type=='S-G_filter':
        vals_smooth = savgol_filter(values, window_size, order,mode='nearest')
    elif type=='smoothing':
        for j in np.arange(window_half,nx-window_half):
            vals_smooth[j] = np.nanmean(values[j-window_half:j+window_half])

    # elif len(np.shape(values))==2:
    #     nx              = len(values[:,0])
    #     ny              = len(values[0,:])
    #     vals_smooth     = np.full([nx,ny], np.nan)

    #     for i in np.arange(nx):
    #         if type=='S-G_filter':
    #             vals_smooth[i,:] = savgol_filter(values[i,:], window_size, order)
    #         elif type=='smoothing':
    #             for j in np.arange(window_half,ny-window_half):
    #                 vals_smooth[i,j] = np.nanmean(values[i,j-window_half:j+window_half])

    return vals_smooth


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

def check_variable_units(PLUMBER2_path, varname, site_name, model_names, key_word, key_word_not=None):

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
                        my_dict[model_name] = dataset.variables[var_name].units
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
                                my_dict[model_name] = dataset.variables[var_name].units
                                var_exist = True
                                # print(f"The word '{key_word}' is in the description of variable '{var_name}'.")
                                break  # Exit the loop once a variable is found


        except Exception as e:
            print(f"An error occurred: {e}")

        # variable doesn't exist
        if not var_exist:
            my_dict[model_name] = 'None'

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

def set_model_colors():

    # file path
    model_colors = {
                    "obs": 'black',
                    "obs_cor": 'dimgrey',
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
                    "JULES_GL9_withLAI": "limegreen",
                    "JULES_test": "forestgreen",
                    "LPJ-GUESS": "turquoise",
                    "Manabe": "lightseagreen",
                    "ManabeV2": "darkcyan",
                    "MATSIRO": "deepskyblue",
                    "MuSICA": "dodgerblue",
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

def conduct_quality_control(varname, data_input,zscore_threshold=2):

    '''
    Please notice EF has nan values
    '''

    z_scores    = np.abs(stats.zscore(data_input, nan_policy='omit'))
    data_output = np.where(z_scores > zscore_threshold, np.nan, data_input)

    # print('z_scores',z_scores)
    if 'EF' not in varname:
        print('EF is not in ', varname)
        # Iterate through the data to replace NaN with the average of nearby non-NaN values
        for i in range(1, len(data_output) - 1):
            if np.isnan(data_output[i]):
                prev_index = i - 1
                next_index = i + 1

                # find the closest non nan values
                while prev_index >= 0 and np.isnan(data_output[prev_index]):
                    prev_index -= 1

                while next_index < len(data_output) and np.isnan(data_output[next_index]):
                    next_index += 1

                # use average them
                if prev_index >= 0 and next_index < len(data_output):
                    prev_non_nan = data_output[prev_index]
                    next_non_nan = data_output[next_index]
                    data_output[i] = (prev_non_nan + next_non_nan) / 2.0

    print('len(z_scores)',len(z_scores))
    # print('data_output',data_output)

    return data_output

def convert_into_kg_m2_s(data_input, var_units):

    d_2_s = 24*60*60
    if 'W' in var_units and 'm' in var_units and '2' in var_units:
        print('converting ', var_units)
        data_output = data_input * 86400 / 2454000 /d_2_s
    return data_output

def convert_from_umol_m2_s_into_gC_m2_s(data_input, var_units):

    # convert from umol/m2/s to gC/m2/s
    umol_2_mol  = 0.000001
    mol_2_gC    = 12
    print('converting ', var_units)
    data_output = data_input*umol_2_mol*mol_2_gC

    return data_output
