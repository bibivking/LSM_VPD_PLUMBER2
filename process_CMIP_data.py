import os
import sys
import gc
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

def time_mask(time_tmp, time_s, time_e, seconds=None):

    '''
    Checked on 14 Dec 2021, no problem was identified
    '''

    # #print("In time_mask")
    Time    = time_tmp - datetime(2000,1,1,0,0,0)
    Time_s  = time_s - datetime(2000,1,1,0,0,0)
    Time_e  = time_e - datetime(2000,1,1,0,0,0)
    # print('Time',Time)
    # print('Time_s',Time_s)
    if seconds == None:
        time_cood = (Time>=Time_s) & (Time<Time_e)
    else:
        time_cood = []
        for j in np.arange(len(Time)):
            if seconds[0] >= seconds[1]:
                if_seconds = (Time[j].seconds >= seconds[0]) | (Time[j].seconds < seconds[1])
            else:
                if_seconds = (Time[j].seconds >= seconds[0]) & (Time[j].seconds < seconds[1])
            time_cood.append( (Time[j]>=Time_s) & (Time[j]<Time_e) & if_seconds)

    return time_cood

def read_CMIP6_data(site_name, file_names, scenarios, var_names, lat=None, lon=None, time_s=None, time_e=None):

    # select the site information from each CMIP6 file

    for scenario in scenarios:

        file_names_scenario = file_names[scenario]
        output_file         = CMIP6_out_path+site_name+'_'+scenario+'.nc'
        print('Output file is ', output_file)

        for var_name in var_names:

            file_names_scenario_variable = file_names_scenario[var_name]
            model_out_list               = []

            for file_name in file_names_scenario_variable:

                # print('file_name',file_name)
                # ! ncdump -h {file_name}
                model_out_name = file_name.split("/")[9]
                model_out_list.append(model_out_name)

                # Get model name
                f         = nc.Dataset(file_name, mode='r')

                # Read lat and lon
                try:
                    latitude  = f.variables['lat'][:]
                    longitude = f.variables['lon'][:]
                except:
                    latitude  = f.variables['latitude'][:]
                    longitude = f.variables['longitude'][:]

                # Read time
                time_tmp  = nc.num2date(f.variables['time'][:],f.variables['time'].units,
                            only_use_cftime_datetimes=False, calendar=f.variables['time'].calendar) # only_use_python_datetimes=True,

                # To solve the inconsistancy in time coordinate
                for i, t in enumerate(time_tmp):
                    year   = t.year
                    month  = t.month
                    day    = t.day
                    hour   = t.hour
                    minute = t.minute
                    second = t.second
                    microsecond = t.microsecond
                    time_tmp[i] = datetime(year, month, day, hour, minute, second, microsecond)

                # select time periods
                time_cood = time_mask(time_tmp, time_s, time_e)

                # make new time cooridate
                time_tmp  = time_tmp[time_cood]

                # Read variable
                var_tmp = f.variables[var_name][:]
                var_units = f.variables[var_name].units
                var_long_name = f.variables[var_name].long_name
                lat_idx = np.argmin(np.abs(latitude - lat))
                lon_idx = np.argmin(np.abs(longitude - lon))
                var     = var_tmp[time_cood, lat_idx, lon_idx]

                # Make nc file
                if not os.path.exists(output_file):

                    # make output file
                    f = nc.Dataset(output_file, 'w', format='NETCDF4')

                    ### Create nc file ###
                    # f.history           = "Created by: %s" % (os.path.basename(__file__))
                    f.creation_date     = "%s" % (datetime.now())
                    f.description       = 'CMIP6 '+scenario+' at '+site_name+', made by MU Mengyuan'
                    f.Conventions       = "CF-1.0"

                    # set time dimensions
                    ntime               = len(var)
                    Time_name           = 'time'
                    f.createDimension(Time_name, ntime)


                    time_output = []
                    for t_tmp in time_tmp:
                        time_output.append((t_tmp - datetime(2000,1,1,0,0,0)).days)

                    Time                = f.createVariable(Time_name, 'f4', (Time_name))
                    Time.standard_name  = Time_name
                    Time.units          = 'days since 2000-01-01 00:00:00'
                    Time[:]             = time_output

                    Var_name            = model_out_name+"_"+var_name
                    Var                 = f.createVariable(Var_name, 'f4', (Time_name))
                    Var.standard_name   = Var_name
                    Var.units           = var_units
                    Var.long_name       = var_long_name
                    Var[:]              = var[:]

                    f.close()

                else:
                    # add to the exist nct file
                    f = nc.Dataset(output_file, 'r+', format='NETCDF4')

                    # set dimensions
                    ntime               = len(var)
                    Time_name           = 'time'
                    Var_name            = model_out_name+"_"+var_name

                    # if it doesn't exist then create
                    if Var_name not in f.variables:
                        Var                = f.createVariable(Var_name, 'f4', (Time_name))
                        Var.standard_name  = Var_name
                        Var.units          = var_units
                        Var.long_name      = var_long_name
                        Var[:]             = var[:]
                    else:
                        f.variables[Var_name][:]    = var[:]
                        f.variables[Var_name].units = var_units

                    f.close()


            # Add model list
            f = nc.Dataset(output_file, 'r+', format='NETCDF4')
            model_list_name = var_name+"_models"
            model_out_num   = len(model_out_list)
            model_names_array= np.array(model_out_list, dtype="S20")
            if model_list_name not in f.variables:
                # set model names dimension
                f.createDimension(model_list_name, model_out_num)

                # create variables
                model               = f.createVariable(model_list_name, "S20", (model_list_name))
                model.standard_name = model_list_name
                model[:]            = model_names_array
            f.close()
            gc.collect()

    return

def add_EF_to_nc_file(output_file, zscore_threshold=2, Qle_Qh_threshold=10):
    # output_file = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6/AR-SLu_historical.nc'
    # zscore_threshold=2
    # Qle_Qh_threshold=10

    # Set input file path
    f_out                = nc.Dataset(output_file,'r+')
    latent_models        = f_out.variables['hfls_models'][:]
    sensible_models      = f_out.variables['hfss_models'][:]

    model_out_names      = []
    model_out_num        = 0

    # check whether both latent and sensible fluxes exist in the model
    for latent_model in latent_models:
        if latent_model in sensible_models:

            model_out_names.append(latent_model)
            model_out_num  = model_out_num + 1

            # print(latent_model, 'has both Qle and Qh')

            f_out          = nc.Dataset(output_file,'r+')
            model_sensible = f_out.variables[latent_model+'_hfls'][:]
            model_latent   = f_out.variables[latent_model+'_hfss'][:]

            model_EF_tmp   = np.where(np.all([model_sensible+model_latent > Qle_Qh_threshold , model_sensible>0],axis=0),
                                      model_latent/(model_sensible+model_latent), np.nan)
            model_EF_tmp   = np.where(model_EF_tmp<0,np.nan,model_EF_tmp)
            model_EF_tmp   = conduct_quality_control('EF',model_EF_tmp,zscore_threshold)

            # Try to access the variable
            try:
                # if model_EF exists, update to the new values
                f_out.variables[latent_model+'_EF'][:] = model_EF_tmp
            except:
                # if model_EF doesn't exist, create the model_EF
                model_EF                = f_out.createVariable(latent_model+'_EF', 'f4', ('time'))
                model_EF.standard_name  = latent_model+"_EF"
                model_EF.long_name      = "Evaporative fraction (hfls/(hfls+hfss)) in "+latent_model
                model_EF.units          = "-"
                model_EF[:]             = model_EF_tmp
            f_out.close()

    # output the model has both Qle and Qh
    f_out               = nc.Dataset(output_file,'r+')

    try:
        # if model_EF exists, update to the new values
        print('EF_models:', f_out.variables['EF_models'])
    except:
        # set model names dimension
        f_out.createDimension("EF_models", model_out_num)

        # Form the model names array
        model_names_array   = np.array(model_out_names, dtype="S20")

        # create variables
        model               = f_out.createVariable("EF_models", "S20", ("EF_models"))
        model.standard_name = "EF_models"
        model[:]            = model_names_array

    f_out.close()

    return

def add_vpd_to_nc_file(output_file):
    # output_file = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6/AR-SLu_historical.nc'

    # Set input file path
    f_out                = nc.Dataset(output_file,'r+')
    RH_models            = f_out.variables['hurs_models'][:]
    Tair_models          = f_out.variables['tas_models'][:]

    model_out_names      = []
    model_out_num        = 0

    # check whether both latent and sensible fluxes exist in the model
    for RH_model in RH_models:
        if RH_model in Tair_models:

            model_out_names.append(RH_model)
            model_out_num  = model_out_num + 1

            # print(latent_model, 'has both Qle and Qh')

            f_out          = nc.Dataset(output_file,'r+')
            model_RH       = f_out.variables[RH_model+'_hurs'][:]
            model_Tair     = f_out.variables[RH_model+'_tas'][:]
            # print('model_Tair',model_Tair)
            model_VPD_tmp  = calculate_VPD_by_RH(model_RH, model_Tair)

            # Try to access the variable
            try:
                # if model_EF exists, update to the new values
                f_out.variables[RH_model+'_VPD'][:] = model_VPD_tmp
            except:
                # if model_EF doesn't exist, create the model_EF
                model_VPD                = f_out.createVariable(RH_model+'_VPD', 'f4', ('time'))
                model_VPD.standard_name  = RH_model+"_VPD"
                model_VPD.long_name      = "Vapor Pressure Deficit in "+RH_model
                model_VPD.units          = "-"
                model_VPD[:]             = model_VPD_tmp
            f_out.close()

    # output the model has both Qle and Qh
    f_out               = nc.Dataset(output_file,'r+')

    try:
        # if model_EF exists, update to the new values
        print('VPD_models:', f_out.variables['VPD_models'])
    except:
        # set model names dimension
        f_out.createDimension("VPD_models", model_out_num)

        # Form the model names array
        model_names_array   = np.array(model_out_names, dtype="S20")

        # create variables
        model               = f_out.createVariable("VPD_models", "S20", ("VPD_models"))
        model.standard_name = "VPD_models"
        model[:]            = model_names_array

    f_out.close()

    return

if __name__ == "__main__":

    # Read files
    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/amu561/CMIP6_for_Mengyuan/Processed_CMIP6_data/"
    CMIP6_out_path    = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6/"
    scenarios         = ['historical']#,'ssp126','ssp245','ssp585']
    var_names         = ['hfls','hfss','hurs','tas']

    # The site names
    all_site_path     = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    site_names        = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    zscore_threshold  = 2
    Qle_Qh_threshold  = 10

    # Read variable attributions info from input
    lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_met_path)

    # get file names
    file_names           = {}
    file_names_scenario  = {}

    for scenario in scenarios:
        for var_name in var_names:
            file_names_scenario[var_name] = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/{var_name}/*/*/*.nc'))
        file_names[scenario] = file_names_scenario

    lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_met_path)
    time_s             = datetime(1950,1,1,0,0,0)
    time_e             = datetime(2015,1,1,0,0,0)
    # time_s             = datetime(2060,1,1,0,0,0)
    # time_e             = datetime(2100,1,1,0,0,0)

    # for site_name in site_names:
        # get site lat and lon
        # lat, lon = read_lat_lon(site_names, PLUMBER2_met_path)
    site_name = site_names[0]
    # read CMIP6 data
    read_CMIP6_data(site_name, file_names, scenarios, var_names,
                    lat=lat_dict[site_name], lon=lon_dict[site_name],
                    time_s=time_s, time_e=time_e)
    #
    # lat=lat_dict[site_name]
    # lon=lon_dict[site_name]

    for scenario in scenarios:
        output_file = CMIP6_out_path+site_name+'_'+scenario+'.nc'
        add_EF_to_nc_file(output_file, zscore_threshold=zscore_threshold, Qle_Qh_threshold=Qle_Qh_threshold)
        add_vpd_to_nc_file(output_file)
