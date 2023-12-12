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
from scipy.interpolate import griddata
from pygam import LinearGAM, PoissonGAM
from scipy import stats, interpolate
from scipy.signal import savgol_filter
from datetime import datetime, timedelta


def calculate_VPD_by_RH(rh, tair):

    '''
    calculate vpd
    '''
    print('calculating VPD')
    # set nan values
    rh   = np.where(rh<-1.,np.nan,rh)
    tair = np.where(tair<-100.,np.nan,tair)

    DEG_2_KELVIN = 273.15
    PA_TO_KPA    = 0.001

    # convert back to Pa
    # tair         -= DEG_2_KELVIN

    # saturation vapor pressure
    es = 100.0 * 6.112 * np.exp((17.67 * tair) / (243.5 + tair))

    # actual vapor pressure
    ea = rh/100. * es

    vpd = (es - ea) * PA_TO_KPA
    # vpd = np.where(vpd < 0.05, 0.05, vpd)

    return vpd

def load_default_list():

    # The site names
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]

    IGBP_types     = ['GRA','OSH', 'SAV', 'WSA', 'CSH', 'DBF', 'ENF', 'EBF', 'MF', 'WET', 'CRO']
    clim_types     = ['Af', 'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Csa', 'Csb', 'Cwa',
                      'Dfa', 'Dfb', 'Dfc', 'Dsb', 'Dsc', 'Dwa', 'Dwb', 'ET']

    model_names_all = [ "1lin","3km27", "6km729","6km729lag",
                        "LSTM_eb","LSTM_raw", "RF_eb","RF_raw",
                        "Manabe", "ManabeV2","PenmanMonteith",
                        "CABLE", "CABLE-POP-CN","CHTESSEL_ERA5_3",
                        "CHTESSEL_Ref_exp1","CLM5a","GFDL",
                        "JULES_GL9_withLAI","JULES_test","MATSIRO",
                        "NoahMPv401","ORC2_r6593", "ORC2_r6593_CO2",
                        "ORC3_r7245_NEE", "ORC3_r8120","STEMMUS-SCOPE",
                        "ACASA","LPJ-GUESS","MuSICA",
                        "NASAEnt","QUINCY", "SDGVM",] #"BEPS"

    model_names_select= ["CABLE", "CABLE-POP-CN","CHTESSEL_ERA5_3",
                         "CHTESSEL_Ref_exp1","CLM5a","GFDL",
                         "JULES_GL9_withLAI","JULES_test","LPJ-GUESS",
                         "MATSIRO","MuSICA","NASAEnt",
                         "NoahMPv401","ORC2_r6593", "ORC2_r6593_CO2",
                         "ORC3_r7245_NEE", "ORC3_r8120","QUINCY",
                         "SDGVM","STEMMUS-SCOPE",] #"BEPS"

    empirical_model  = ["1lin","3km27", "6km729","6km729lag",
                        "LSTM_eb","LSTM_raw", "RF_eb","RF_raw",]
    hydrological_model  = ["Manabe", "ManabeV2","PenmanMonteith",]
    land_surface_model=["CABLE", "CABLE-POP-CN","CHTESSEL_ERA5_3",
                        "CHTESSEL_Ref_exp1","CLM5a","GFDL",
                        "JULES_GL9_withLAI","JULES_test","MATSIRO",
                        "NoahMPv401","ORC2_r6593", "ORC2_r6593_CO2",
                        "ORC3_r7245_NEE", "ORC3_r8120","STEMMUS-SCOPE",]
    ecosystem_model  = ["ACASA","LPJ-GUESS","MuSICA",
                        "NASAEnt","QUINCY", "SDGVM",]

    model_names      = {
                        'all_model': model_names_all,
                        'model_select':model_names_select,
                        'empirical_model':empirical_model,
                        'hydrological_model':hydrological_model,
                        'land_surface_model':land_surface_model,
                        'ecosystem_model':ecosystem_model}

    return site_names, IGBP_types, clim_types, model_names

def load_sites_in_country_list(country_code):

    # The site names
    if country_code != None:
        PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
        all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*"+country_code+"*.nc"))
        site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    else:
        site_names     = None
    return site_names

def fit_GAM(x_top, x_bot, x_interval, x_values,y_values,n_splines=4,spline_order=2):

    x_series   = np.arange(x_bot, x_top, x_interval)

    # x_values in gridsearch should be of shape [n_samples, m_features], for this case
    # m_features=1, so reshape x_values to [len(x_values),1]
    x_values_array = x_values.to_numpy()
    x_values       = x_values_array.reshape(len(x_values),1)

    # calculate mean value
    gam          = PoissonGAM(n_splines=n_splines,spline_order=spline_order).gridsearch(x_values, y_values) # n_splines=22
    # gam          = LinearGAM(n_splines=n_splines,spline_order=spline_order).gridsearch(x_values, y_values) # n_splines=22
    y_pred       = gam.predict(x_series)
    y_int        = gam.confidence_intervals(x_series, width=.95)

    return x_series, y_pred, y_int

def fit_spline(x_top, x_bot, x_interval, x_values,y_values,n_splines=4,spline_order=2):

    x_series   = np.arange(x_bot, x_top, x_interval)

    # x_values in gridsearch should be of shape [n_samples, m_features], for this case
    # m_features=1, so reshape x_values to [len(x_values),1]
    x_values_array = x_values.to_numpy()
    x_values       = x_values_array.reshape(len(x_values),1)

    tck  = interpolate.splrep(x_values, y_values, s=len(x_series))
    y_pred = interpolate.BSpline(*tck)(x_series)

    return x_series, y_pred

def smooth_vpd_series(values, window_size=11, order=3, smooth_type='S-G_filter',deriv=0):

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

    if smooth_type=='S-G_filter':
        vals_smooth = savgol_filter(values, window_size, order, mode='nearest',deriv=deriv)
    elif smooth_type=='smoothing':
        for j in np.arange(window_half,nx-window_half):
            vals_smooth[j] = np.nanmean(values[j-window_half:j+window_half])
        vals_smooth = np.roll(vals_smooth, 10, fill_value='NaN')
    # elif

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
                    if varname.lower() in var_name.lower():
                        variable  = dataset.variables[var_name]
                        if hasattr(variable, 'long_name'):
                            long_name = variable.long_name.lower()
                            if key_word in long_name and all(not re.search(key_not, long_name) for key_not in key_word_not):
                                my_dict[model_name] = var_name
                                var_exist = True
                        else:
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

def regrid_data(lat_in, lon_in, lat_out, lon_out, input_data, method='linear',threshold=None):

    if len(np.shape(lat_in)) == 1:
        lon_in_2D, lat_in_2D = np.meshgrid(lon_in,lat_in)
        lon_in_1D            = np.reshape(lon_in_2D,-1)
        lat_in_1D            = np.reshape(lat_in_2D,-1)
    elif len(np.shape(lat_in)) == 2:
        lon_in_1D            = np.reshape(lon_in,-1)
        lat_in_1D            = np.reshape(lat_in,-1)
    else:
        print("ERROR: lon_in has ", len(np.shape(lat_in)), "dimensions")

    if len(np.shape(lat_out)) == 1:
        lon_out_2D, lat_out_2D = np.meshgrid(lon_out,lat_out)
    elif len(np.shape(lat_out)) == 2:
        lon_out_2D            = lon_out
        lat_out_2D            = lat_out
    else:
        print("ERROR: lon_out has ", len(np.shape(lat_in)), "dimensions")

    value_tmp = np.reshape(input_data,-1)

    # Check NaN - input array shouldn't have NaN
    if threshold is None:
        mask_values     = ~np.isnan(value_tmp)
    else:
        #print("np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0) ",np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0))
        mask_values     = np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0)

    value     = value_tmp[mask_values]
    # ======= CAUTION =======
    lat_in_1D = lat_in_1D[mask_values]  # here I make nan in values as the standard
    lon_in_1D = lon_in_1D[mask_values]
    #print("shape value = ", np.shape(value))
    #print("shape lat_in_1D = ", np.shape(lat_in_1D))
    #print("shape lon_in_1D = ", np.shape(lon_in_1D))
    # =======================
    #print("value =",value)
    #print("lat_in_1D =",lat_in_1D)
    #print("lon_in_1D =",lon_in_1D)
    Value = griddata((lon_in_1D, lat_in_1D), value, (lon_out_2D, lat_out_2D), method=method)

    return Value

def set_model_colors():

    # file path
    model_colors = {
                    'obs': 'black',
                    'CABLE':'darkblue',
                    'CABLE-POP-CN':'blue',
                    'CHTESSEL_ERA5_3':'cornflowerblue',
                    'CHTESSEL_Ref_exp1':'c',
                    'CLM5a':'deepskyblue',
                    'GFDL':'aquamarine',
                    'JULES_GL9_withLAI':'lime',
                    'JULES_test':'forestgreen',
                    'LPJ-GUESS':'darkolivegreen',
                    'MATSIRO':'limegreen',
                    'MuSICA':'yellowgreen',
                    'NASAEnt':'yellow'  ,
                    'NoahMPv401':'gold',
                    'ORC2_r6593':'orange',
                    'ORC2_r6593_CO2':'red',
                    'ORC3_r7245_NEE':'coral',
                    'ORC3_r8120':'pink',
                    'QUINCY':'deeppink',
                    'SDGVM':'violet' ,
                    'STEMMUS-SCOPE':'darkviolet' ,
                    }

    # model_colors = {
    #                 'obs': 'black',
    #                 'obs_cor': 'dimgrey',
    #                 '1lin': 'lightcoral' ,
    #                 '3km27': 'indianred',
    #                 '6km729': 'firebrick',
    #                 '6km729lag':'red',
    #                 'LSTM_eb': 'coral',
    #                 'LSTM_raw': 'pink',
    #                 'RF_eb': 'tomato',
    #                 'RF_raw': 'deeppink',
    #                 'Manabe':'violet',
    #                 'ManabeV2':'darkviolet',
    #                 'PenmanMonteith': 'purple',
    #                 'CABLE':'darkblue',
    #                 'CABLE-POP-CN':'blue',
    #                 'CHTESSEL_ERA5_3':'cornflowerblue',
    #                 'CHTESSEL_Ref_exp1':'dodgerblue',
    #                 'CLM5a':'deepskyblue',
    #                 'GFDL':'c',
    #                 'JULES_GL9_withLAI':'aquamarine',
    #                 'JULES_test':'lightseagreen',
    #                 'MATSIRO':'darkcyan',
    #                 'NoahMPv401':'darkolivegreen',
    #                 'ORC2_r6593':'forestgreen',
    #                 'ORC2_r6593_CO2':'limegreen',
    #                 'ORC3_r7245_NEE':'lime',
    #                 'ORC3_r8120':'lightgreen',
    #                 'STEMMUS-SCOPE':'yellowgreen',
    #                 'ACASA':'yellow',
    #                 'LPJ-GUESS':'orange',
    #                 'MuSICA':'gold',
    #                 'NASAEnt': 'goldenrod',
    #                 'QUINCY':'peru',
    #                 'SDGVM':'sandybrown',
    #                 }


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

def get_model_soil_moisture_info():

    # Set models with simulated soil mositure
    SM_names      = { 'CABLE':'SoilMoist', 'CABLE-POP-CN':'SoilMoist',
                      'CHTESSEL_ERA5_3':'SoilMoist', 'CHTESSEL_Ref_exp1':'SoilMoist',
                      'CLM5a':'mrlsl', 'GFDL':'SoilMoist', 'JULES_GL9_withLAI':'SoilMoist',
                      'JULES_test':'SoilMoist', 'MATSIRO':'SoilMoistV', 'MuSICA':'SoilMoist',
                      'NoahMPv401':'SoilMoist', 'ORC2_r6593':'SoilMoist','ORC3_r7245_NEE':'SoilMoist',
                      'ORC3_r8120':'SoilMoist','STEMMUS-SCOPE':'SoilMoist'} #  'SDGVM':'RootMoist',

    # Set soil layer thickness
    soil_thicknesses = {'CABLE':[0.022,0.058, 0.154, 0.409, 1.085, 2.872],
                        'CABLE-POP-CN':[0.022,0.058, 0.154, 0.409, 1.085, 2.872],
                        'CHTESSEL_ERA5_3':[0.07, 0.21, 0.72, 1.89],
                        'CHTESSEL_Ref_exp1':[0.07, 0.21, 0.72, 1.89],
                        'JULES_test': [0.1, 0.25, 0.65, 2.0],
                        'JULES_GL9_withLAI': [0.1, 0.25, 0.65, 2.0],
                        'MATSIRO': [0.05, 0.2, 0.75, 1., 2.],
                        'NoahMPv401': [0.10, 0.30, 0.60,1.],
                        'CLM5a':[0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4,
                                 0.44, 0.54, 0.64, 0.74, 0.84, 0.94, 1.04, 1.14, 2.39, 4.675534, 7.63519,
                                 11.14, 15.11543],
                        'GFDL': [0.02,0.04,0.04,0.05,0.05,0.1,0.1,0.2,0.2,0.2,
                                  0.4,0.4,0.4,0.4,0.4,1,1,1,1.5,2.5],
                        'MuSICA': [0.01695884, 0.01970336, 0.02289203, 0.02659675, 0.03090101,
                                    0.03590186, 0.041712, 0.04846244, 0.05630532, 0.06541745, 0.07600423,
                                    0.08830432, 0.102595, 0.1191984, 0.1384887, 0.160901, 0.1869402,
                                    0.2171936, 0.2523429, 0.2931806],
                        'ORC2_r6593':[0.0009775171, 0.003910068, 0.009775171, 0.02150538, 0.04496579,
                                      0.09188661, 0.1857283, 0.3734115, 0.7487781, 1.499511, 2],
                        'ORC3_r7245_NEE':[0.0009775171, 0.003910068, 0.009775171, 0.02150538, 0.04496579,
                                          0.09188661, 0.1857283, 0.3734115, 0.7487781, 1.499511, 2],
                        'ORC3_r8120':[0.0009775171, 0.003910068, 0.009775171, 0.02150538, 0.04496579,
                                      0.09188661, 0.1857283, 0.3734115, 0.7487781, 1.499511, 2],
                        'STEMMUS-SCOPE':[ 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                                        0.02, 0.02, 0.02, 0.02, 0.02, 0.025, 0.025, 0.025, 0.025, 0.05, 0.05,
                                        0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
                                        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15,
                                        0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]}
    return SM_names, soil_thicknesses

def bootstrap_ci( data, statfunction=np.average, alpha = 0.05, n_samples = 100):

    """
    Purpose calculate confidence interval by bootstrap method
    Code source: https://stackoverflow.com/questions/44392978/compute-a-confidence-interval-from-sample-data-assuming-unknown-distribution/66008548#66008548
    """
    
    import warnings

    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0],))    
    
    alphas = np.array([alpha/2, 1 - alpha/2])
    nvals  = np.round((n_samples - 1) * alphas).astype(int)
    if np.any(nvals < 10) or np.any(nvals >= n_samples-10):
        warnings.warn("Some values used extremal samples; results are probably unstable. "
                      "Try to increase n_samples")

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()
    
    boot_indexes = bootstrap_ids(data, n_samples)
    print('boot_indexes',boot_indexes)
    stat         = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    print('stat',stat)
    stat.sort(axis=0)

    return stat[nvals]



