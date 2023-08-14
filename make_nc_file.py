#!/usr/bin/env python

"""
Note that VPD still has issue
"""
import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from check_vars import check_variable_exists

def qair_to_vpd(qair, tair, press):
    '''
    calculate vpd
    '''
    DEG_2_KELVIN = 273.15
    PA_TO_KPA    = 0.001
    PA_TO_HPA    = 0.01

    # convert back to Pa
    press        = press/PA_TO_HPA
    tair         = tair-DEG_2_KELVIN

    # saturation vapor pressure
    es = 100.0 * 6.112 * np.exp((17.67 * tair) / (243.5 + tair))

    # vapor pressure
    ea = (qair * press) / (0.622 + (1.0 - 0.622) * qair)

    vpd = (es - ea) * PA_TO_KPA
    # vpd = np.where(vpd < 0.05, 0.05, vpd)

    return vpd

def make_nc_file(PLUMBER2_path, var_name_dict, model_names, site_name, output_file):

    nmodel          = len(model_names)
    model_out_num   = 0
    model_out_names = []
    model_vars      = {}
    model_times     = {}
    model_time_units= {}
    model_time_calendars= {}

    for j, model_name in enumerate(model_names):

        # if the variable exists
        if var_name_dict[model_name] != 'None':

            model_out_names.append(model_name)
            model_out_num  = model_out_num + 1

            # Set input file path
            file_path    = glob.glob(PLUMBER2_path+model_name +"/*"+site_name+"*.nc")

            print('j=',j, "model_name=",model_name, 'file_path=',file_path)

            # Change var_name for the two models 
            var_name_tmp = var_name_dict[model_name]

            # Open input file
            f = nc.Dataset(file_path[0])

            if model_out_num == 0:
                # Read var info from input             
                var_unit       = f.variables[var_name_tmp].units
                var_long_name  = f.variables[var_name_tmp].long_name

            # # Set output array
            # ntime          = len(Time)

            # # Read variable values
            # Var            = np.zeros((nmodel,ntime))

            Var_tmp        = f.variables[var_name_tmp][:]      
            
            # Reset the default values
            if '_FillValue' in f.variables[var_name_tmp].ncattrs():
                var_FillValue  = f.variables[var_name_tmp]._FillValue
            elif '_fillValue' in f.variables[var_name_tmp].ncattrs():
                var_FillValue  = f.variables[var_name_tmp]._fillValue
            elif 'missing_value' in f.variables[var_name_tmp].ncattrs():
                var_FillValue  = f.variables[var_name_tmp].missing_value

            if var_FillValue !=None:
                Var_tmp = np.where(Var_tmp == var_FillValue, np.nan, Var_tmp )

            # Read time info from input
            model_vars[model_name]           = Var_tmp.reshape(-1)
            model_times[model_name]          = f.variables['time'][:]
            model_time_units[model_name]     = f.variables['time'].units
            model_time_calendars[model_name] = f.variables['time'].calendar

            # Close the input file
            f.close()
            var_FillValue= None

        # make output file
        f = nc.Dataset(output_file, 'w', format='NETCDF4')

        ### Create nc file ###
        f.history           = "Created by: %s" % (os.path.basename(__file__))
        f.creation_date     = "%s" % (datetime.now())
        f.description       = 'PLUMBER2 '+var_name+'at '+site_name+', made by MU Mengyuan'

        # set dimensions
        f.createDimension('model', model_out_num)
        
        f.createDimension('time', ntime)
        f.Conventions       = "CF-1.0"

        model_names_array = np.array(model_names, dtype="S20")

        # create variables
        model               = f.createVariable('model', "S20", ('model'))
        model.standard_name = "model name"
        model[:]            = model_names_array

        time                = f.createVariable('time', 'f4', ('time'))
        time.standard_name  = "time"
        time.units          = time_unit
        time.calendar       = time_calendar
        time[:]             = Time

        var                = f.createVariable(var_name, 'f4', ('model','time'))
        var.standard_name  = var_name
        var.units          = var_unit
        var.long_name      = var_long_name
        var[:]             = Var

        f.close()
    return

def add_Qle_obs_to_nc_file(PLUMBER2_flux_path, site_name, output_file):

    # Set input file path
    file_path          = glob.glob(PLUMBER2_flux_path +"/*"+site_name+"*.nc")
    print('file_path', file_path)

    f_in               = nc.Dataset(file_path[0])
    Qle                = f_in.variables['Qle'][:]
    Qle                = np.where(Qle == -9999., np.nan, Qle)
    f_in.close()

    f_out              = nc.Dataset(output_file,'r+')
    obs                = f_out.createVariable('Qle_obs', 'f4', ('time'))
    obs.standard_name  = "Qle_obs"
    obs.long_name      = "Latent heat flux from surface"
    obs.units          = "W/m2"
    obs[:]             = Qle
    f_out.close()

    return

def add_met_to_nc_file(PLUMBER2_met_path, site_name, output_file):

    # Set input file path
    file_path = glob.glob(PLUMBER2_met_path +"/*"+site_name+"*.nc")
    f_in      = nc.Dataset(file_path[0])
    Qair      = f_in.variables['Qair'][:]
    Tair      = f_in.variables['Tair'][:]
    Psurf     = f_in.variables['Psurf'][:]
    f_in.close()

    Tair      = np.where(Tair == -9999., np.nan, Tair)
    Qair      = np.where(Qair == -9999., np.nan, Qair)
    Psurf     = np.where(Psurf == -9999., np.nan, Psurf)

    VPD       = qair_to_vpd(Qair, Tair, Psurf)

    f_out               = nc.Dataset(output_file,'r+')

    tair                = f_out.createVariable('Tair_obs', 'f4', ('time'))
    tair.standard_name  = "Tair_obs"
    tair.long_name      = "Near surface air temperature"
    tair.units          = "K"
    tair[:]             = Tair

    qair                = f_out.createVariable('Qair_obs', 'f4', ('time'))
    qair.standard_name  = "Qair_obs"
    qair.long_name      = "Near surface specific humidity"
    qair.units          = "kg/kg"
    qair[:]             = Qair

    psurf                = f_out.createVariable('Psurf_obs', 'f4', ('time'))
    psurf.standard_name  = "Psurf_obs"
    psurf.long_name      = "Surface air pressure"
    psurf.units          = "Pa"
    psurf[:]             = Psurf

    vpd                  = f_out.createVariable('VPD', 'f4', ('time'))
    vpd.standard_name    = "VPD"
    vpd.long_name        = "Vapor pressure deficit"
    vpd.units            = "kPa"
    vpd[:]               = VPD

    f_out.close()

    return


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"

    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    # The name of models
    model_names   = [ "CABLE","3km27","CABLE-POP-CN","CHTESSEL_Ref_exp1",
                      "GFDL","MATSIRO","NASAEnt","ORCHIDEE_tag2.1",
                      "QUINCY","ACASA","CHTESSEL_ERA5_3","CLM5a",
                      "JULES_GL9","LSTM_raw","MuSICA","NoahMPv401","ORCHIDEE_tag3_2",
                      "RF","STEMMUS-SCOPE"] # "LPJ-GUESS","SDGVM", "BEPS",

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    print(all_site_path)
    site_names     = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    print(site_names)

    var_name     = "Qle"
    key_word     = "trans"
    key_word_not = "evap"

    for site_name in site_names:

        output_file  = "/g/data/w97/mm3972/scripts/PLUMBER2/VPD_impact/nc_files/"+site_name+".nc"

        var_name_dict = check_variable_exists(PLUMBER2_path, site_name, model_names, key_word, key_word_not)

        make_nc_file(PLUMBER2_path, var_name_dict, model_names, site_name, output_file)

        add_Qle_obs_to_nc_file(PLUMBER2_flux_path, site_name, output_file)

        add_met_to_nc_file(PLUMBER2_met_path, site_name, output_file)
        