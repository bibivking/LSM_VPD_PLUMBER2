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
from PLUMBER2_VPD_common_utils import check_variable_exists

def make_nc_file(PLUMBER2_path, var_name_dict, model_names, site_name, output_file, varname, zscore_threshold=2):

    '''
    Make the netcdf file
    '''

    # initilization
    model_out_num        = 0
    model_out_names      = []
    model_vars           = {}
    model_var_units      = {}
    model_times          = {}
    model_time_units     = {}
    model_time_calendars = {}

    # Read in data
    for j, model_name in enumerate(model_names):

        print(var_name_dict[model_name])

        # if the variable exists
        if var_name_dict[model_name] != 'None':

            # add the model name into output list
            model_out_names.append(model_name)
            model_out_num  = model_out_num + 1

            # Set input file path
            file_path      = glob.glob(PLUMBER2_path+model_name +"/*"+site_name+"*.nc")
            print('j=',j, "model_name=",model_name, 'file_path=',file_path)

            # Change var_name for the different models
            var_name_tmp   = var_name_dict[model_name]
            print('len(var_name_tmp)',len(var_name_tmp))

            # Open input file
            f = nc.Dataset(file_path[0])

            # Read variable attributions info from input
            if not model_vars:
                var_long_name = f.variables[var_name_tmp].long_name

            # Read variable from input
            Var_tmp = f.variables[var_name_tmp][:]

            # Reset missing value
            for attr in ['_FillValue', '_fillValue', 'missing_value']:
                if hasattr(f.variables[var_name_tmp], attr):
                    var_FillValue = getattr(f.variables[var_name_tmp], attr)
                    break
            else:
                var_FillValue = None

            # set missing values as nan
            if var_FillValue is not None:
                Var_tmp = np.where(Var_tmp == var_FillValue, np.nan, Var_tmp)

            #  ==== check whether the model use patches ====
            # Check if the variable has patch dimensions (coordinates)
            patch = None
            veget = None

            if hasattr(f.variables[var_name_tmp], 'dimensions'):
                if 'patch' in f.variables[var_name_tmp].dimensions:
                    print('model_name', model_name, 'site_name', site_name,'has patch demension' )
                    patch = f.dimensions['patch'].size
                    print('patch',patch)

            if hasattr(f.variables[var_name_tmp], 'dimensions'):
                if 'veget' in f.variables[var_name_tmp].dimensions:
                    print('model_name', model_name, 'site_name', site_name,'has veget demension' )
                    veget = f.dimensions['veget'].size
                    print('veget',veget)


            if patch is not None:
                if patch > 1:
                    # if model uses patches
                    # read patch fraction
                    patchfrac   = f.variables['patchfrac']
                    print('model_name', model_name, 'site_name', site_name, 'patch = ', patch, 'patchfrac =', patchfrac )

                    # initlize Var_tmp_tmp
                    Var_tmp_tmp = Var_tmp[:,0]*patchfrac[0]

                    # calculate the patch fraction weighted pixel value
                    for i in np.arange(1,patch):
                        Var_tmp_tmp = Var_tmp_tmp + Var_tmp[:,i]*patchfrac[i]
                else:
                    # if model doesn't use patches
                     Var_tmp_tmp = Var_tmp.reshape(-1)

            if veget is not None:
                if veget > 1:
                    # if model uses patches
                    # read veget fraction
                    vegetfrac   = f.variables['vegetfrac']
                    print('model_name', model_name, 'site_name', site_name, 'veget = ', veget, 'vegetfrac =', vegetfrac )

                    # initlize Var_tmp_tmp
                    Var_tmp_tmp = np.zeros(len(Var_tmp[:,0]))

                    # calculate the veget fraction weighted pixel value for each time step
                    for i in np.arange(len(Var_tmp[:,0])):
                        for j in np.arange(0,veget):
                            Var_tmp_tmp[i] = Var_tmp_tmp[i] + Var_tmp[i,j]*vegetfrac[i,j]

                else:
                    # if model doesn't use patches
                     Var_tmp_tmp = Var_tmp.reshape(-1)

            if patch == None and veget == None:
                # if model doesn't use patches
                Var_tmp_tmp = Var_tmp.reshape(-1)


            # Read variable units
            model_var_units[model_name] = f.variables[var_name_tmp].units

            print('model_name=',model_name,len(Var_tmp_tmp))
            # ===== Quality Control =====
            # use average of the previous and later values to replace outlier
            Var_tmp_tmp                 = conduct_quality_control(varname, Var_tmp_tmp,zscore_threshold)

            # ===== Convert Units =====
            # use average of the previous and later values to replace outlier
            if varname == 'TVeg':
                if 'kg' not in model_var_units[model_name] or 's' not in model_var_units[model_name]:
                    print('model_var_units[model_name]',model_var_units[model_name])
                    Var_tmp_tmp  = convert_into_kg_m2_s(Var_tmp_tmp,model_var_units[model_name])

                # unify units
                model_var_units[model_name] = "kg/m^2/s"

            # assign values
            model_vars[model_name]      = Var_tmp_tmp

            # Set time var name for different models
            time_name_in = 'time_counter' if 'ORC' in model_name else 'time'
            print('time_name_in=',time_name_in)

            # Read time info from input
            model_times[model_name]          = f.variables[time_name_in][:]
            model_time_units[model_name]     = f.variables[time_name_in].units

            # Check whether calendar exist
            if hasattr(f.variables[time_name_in], 'calendar'):
                model_time_calendars[model_name] = f.variables[time_name_in].calendar
            else:
                model_time_calendars[model_name] = 'None'

            # Close the input file
            f.close()
            var_FillValue = None
            Var_tmp       = None

    # Form the model names array
    model_names_array = np.array(model_out_names, dtype="S20")

    # Put out data
    for i, model_out_name in enumerate(model_out_names):
        # print('processing ',site_name,'site', model_out_name, ' model')

        # check whether the nc file exists if not creat it
        if not os.path.exists(output_file):

            # make output file
            f = nc.Dataset(output_file, 'w', format='NETCDF4')

            ### Create nc file ###
            f.history           = "Created by: %s" % (os.path.basename(__file__))
            f.creation_date     = "%s" % (datetime.now())
            f.description       = 'PLUMBER2 '+varname+' at '+site_name+', made by MU Mengyuan'

            f.Conventions       = "CF-1.0"

            # set time dimensions
            ntime               = len(model_times[model_out_name])
            time_name           = model_out_name+'_time'
            var_name            = model_out_name+"_"+varname
            f.createDimension(time_name, ntime)

            # set model names dimension
            model_list_name     = varname+"_models"
            f.createDimension(model_list_name, model_out_num)

            # create variables
            model               = f.createVariable(model_list_name, "S20", (model_list_name))
            model.standard_name = model_list_name
            model[:]            = model_names_array

            time                = f.createVariable(time_name, 'f4', (time_name))
            time.standard_name  = time_name
            time.units          = model_time_units[model_out_name]
            if model_time_calendars[model_out_name] != 'None':
                time.calendar   = model_time_calendars[model_out_name]
            time[:]             = model_times[model_out_name]

            var                 = f.createVariable(var_name, 'f4', (time_name))
            var.standard_name   = var_name
            var.units           = model_var_units[model_out_name]
            var.long_name       = var_long_name
            var[:]              = model_vars[model_out_name]

            f.close()
            time  = None
            var   = None
            model = None

        else:
            # add to the exist nct file
            f = nc.Dataset(output_file, 'r+', format='NETCDF4')

            # set dimensions
            ntime               = len(model_times[model_out_name])
            time_name           = model_out_name+'_time'
            var_name            = model_out_name+"_"+varname

            if time_name not in f.variables:
                # create a time dimension
                f.createDimension(time_name, ntime)

                # set time variable
                time                = f.createVariable(time_name, 'f4', (time_name))
                time.standard_name  = time_name
                time.units          = model_time_units[model_out_name]
                if model_time_calendars[model_out_name] != 'None':
                    time.calendar   = model_time_calendars[model_out_name]
                time[:]             = model_times[model_out_name]

            # check whether the model namelists for this variable exists
            model_list_name = varname+"_models"

            # if it doesn't exist then create
            if model_list_name not in f.variables:
                # set model names dimension
                f.createDimension(model_list_name, model_out_num)
                # create variables
                model               = f.createVariable(model_list_name, "S20", (model_list_name))
                model.standard_name = model_list_name
                model[:]            = model_names_array

            var                = f.createVariable(var_name, 'f4', (time_name))
            var.standard_name  = var_name
            var.units          = model_var_units[model_out_name]
            var.long_name      = var_long_name
            # print('len(model_vars[model_out_name])',len(model_vars[model_out_name]))
            # print('ntime',ntime)
            # print('type(model_vars[model_out_name])',type(model_vars[model_out_name]))
            var[:]             = model_vars[model_out_name]

            f.close()
            time = None
            var  = None
            model= None

    return

def add_Qle_obs_to_nc_file(PLUMBER2_flux_path, site_name, output_file):

    # Set input file path
    file_path          = glob.glob(PLUMBER2_flux_path +"/*"+site_name+"*.nc")
    print('file_path', file_path)

    f_in               = nc.Dataset(file_path[0])
    Qle                = f_in.variables['Qle'][:]
    Qle                = np.where(Qle == -9999., np.nan, Qle)

    f_out              = nc.Dataset(output_file,'r+')
    obs                = f_out.createVariable('obs_Qle', 'f4', ('CABLE_time'))
    obs.standard_name  = "obs_latent_heat"
    obs.long_name      = "Latent heat flux from surface"
    obs.units          = "W/m2"
    obs[:]             = Qle

    try:
        Qle_cor                = f_in.variables['Qle_cor'][:]
        Qle_cor                = np.where(Qle_cor == -9999., np.nan, Qle_cor)
        obs_cor                = f_out.createVariable('obs_Qle_cor', 'f4', ('CABLE_time'))
        obs_cor.standard_name  = "obs_latent_heat_cor"
        obs_cor.long_name      = "Latent heat flux from surface, energy balance corrected"
        obs_cor.units          = "W/m2"
        obs_cor[:]             = Qle_cor
    except:
        print('No Qle_cor at ', site_name)

    f_in.close()
    f_out.close()

    return

def add_Qh_obs_to_nc_file(PLUMBER2_flux_path, site_name, output_file):

    # Set input file path
    file_path          = glob.glob(PLUMBER2_flux_path +"/*"+site_name+"*.nc")
    print('file_path', file_path)

    f_in               = nc.Dataset(file_path[0])
    Qh                 = f_in.variables['Qh'][:]
    Qh                 = np.where(Qh == -9999., np.nan, Qh)

    f_out              = nc.Dataset(output_file,'r+')
    obs                = f_out.createVariable('obs_Qh', 'f4', ('CABLE_time'))
    obs.standard_name  = "obs_sensible_heat"
    obs.long_name      = "Sensible heat flux from surface"
    obs.units          = "W/m2"
    obs[:]             = Qh

    try:
        Qh_cor                 = f_in.variables['Qh_cor'][:]
        Qh_cor                 = np.where(Qh_cor == -9999., np.nan, Qh_cor)
        obs_cor                = f_out.createVariable('obs_Qh_cor', 'f4', ('CABLE_time'))
        obs_cor.standard_name  = "obs_sensible_heat_cor"
        obs_cor.long_name      = "Sensible heat flux from surface, energy balance corrected"
        obs_cor.units          = "W/m2"
        obs_cor[:]             = Qh_cor
    except:
        print('No Qh_cor at ', site_name)

    f_in.close()
    f_out.close()

    return

def add_SM_50cm_CABLE_to_nc_file(PLUMBER2_path, site_name, output_file):

    # Set input file path
    file_path          = glob.glob(PLUMBER2_path +"CABLE/*"+site_name+"*.nc")
    print('file_path', file_path)

    f_in               = nc.Dataset(file_path[0])
    SM                 = f_in.variables['SoilMoist'][:]
    SM                 = np.where(SM < 0., np.nan, SM)
    f_in.close()

    SM_50cm            = (SM[:,0,:,:]*0.022 + SM[:,1,:,:]*0.058 + SM[:,2,:,:]*0.154 + SM[:,3,:,:]*0.266)/0.5
    f_out              = nc.Dataset(output_file,'r+')
    obs                = f_out.createVariable('obs_SM50cm', 'f4', ('CABLE_time'))
    obs.standard_name  = "obs_SM50cm"
    obs.long_name      = "Soil moisture in the top 50 cm"
    obs.units          = "m^3/m^3"
    obs[:]             = SM_50cm

    f_out.close()

    return

def add_met_to_nc_file(PLUMBER2_met_path, site_name, output_file):

    # Set input file path
    file_path = glob.glob(PLUMBER2_met_path +"/*"+site_name+"*.nc")
    f_in      = nc.Dataset(file_path[0])
    Qair      = f_in.variables['Qair'][:]
    Tair      = f_in.variables['Tair'][:]
    Psurf     = f_in.variables['Psurf'][:]
    VPD       = f_in.variables['VPD'][:]/10.

    f_in.close()

    Tair      = np.where(Tair  < 0., np.nan, Tair)
    Qair      = np.where(Qair  < 0., np.nan, Qair)
    Psurf     = np.where(Psurf < 0., np.nan, Psurf)

    f_out               = nc.Dataset(output_file,'r+')

    tair                = f_out.createVariable('obs_Tair', 'f4', ('CABLE_time'))
    tair.standard_name  = "obs_Tair"
    tair.long_name      = "Near surface air temperature"
    tair.units          = "K"
    tair[:]             = Tair

    qair                = f_out.createVariable('obs_Qair', 'f4', ('CABLE_time'))
    qair.standard_name  = "obs_Qair"
    qair.long_name      = "Near surface specific humidity"
    qair.units          = "kg/kg"
    qair[:]             = Qair

    psurf                = f_out.createVariable('obs_Psurf', 'f4', ('CABLE_time'))
    psurf.standard_name  = "obs_Psurf"
    psurf.long_name      = "Surface air pressure"
    psurf.units          = "Pa"
    psurf[:]             = Psurf

    vpd                  = f_out.createVariable('VPD', 'f4', ('CABLE_time'))
    vpd.standard_name    = "VPD"
    vpd.long_name        = "Vapor pressure deficit"
    vpd.units            = "kPa"
    vpd[:]               = VPD

    f_out.close()

    return

def add_EF_to_nc_file(output_file, zscore_threshold=2, Qle_Qh_threshold=10):

    # Set input file path
    f_out                 = nc.Dataset(output_file,'r+')

    obs_sensible          = f_out.variables['obs_Qh'][:]
    obs_latent            = f_out.variables['obs_Qle'][:]
    obs_EF_tmp            = np.where(np.all([obs_sensible+obs_latent > Qle_Qh_threshold, obs_sensible>0],axis=0),
                                     obs_latent/(obs_sensible+obs_latent), np.nan)
    obs_EF_tmp            = np.where(obs_EF_tmp<0, np.nan, obs_EF_tmp)
    obs_EF_tmp            = conduct_quality_control('EF',obs_EF_tmp,zscore_threshold)
    latent_models         = f_out.variables['Qle_models'][:]
    sensible_models       = f_out.variables['Qh_models'][:]

    # Try to access the variable
    try:
        # if obs_EF exists, update to the new values
        f_out.variables['obs_EF'][:] = obs_EF_tmp
    except:
        # if obs_EF doesn't exist, create the obs_EF
        obs_EF                = f_out.createVariable('obs_EF', 'f4', ('CABLE_time'))
        obs_EF.standard_name  = "obs_evaporative_fraction"
        obs_EF.long_name      = "Evaporative fraction (Qle/(Qle+Qh)) in obs"
        obs_EF.units          = "-"
        obs_EF[:]             = obs_EF_tmp

    f_out.close()

    model_out_names      = []
    model_out_num        = 0

    # check whether both latent and sensible fluxes exist in the model
    for latent_model in latent_models:
        if latent_model in sensible_models:

            model_out_names.append(latent_model)
            model_out_num  = model_out_num + 1

            print(latent_model, 'has both Qle and Qh')

            f_out          = nc.Dataset(output_file,'r+')
            model_sensible = f_out.variables[latent_model+'_Qh'][:]
            model_latent   = f_out.variables[latent_model+'_Qle'][:]
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
                model_EF                = f_out.createVariable(latent_model+'_EF', 'f4', (latent_model+'_time'))
                model_EF.standard_name  = latent_model+"_EF"
                model_EF.long_name      = "Evaporative fraction (Qle/(Qle+Qh)) in "+latent_model
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

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"

    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

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
    all_site_path  = sorted(glob.glob(PLUMBER2_met_path+"/US*.nc"))
    site_names     = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    # site_names     = ['AU-Sam','AU-Stp','AU-TTE','AU-Tum','AU-Whr','AU-Wrr','AU-Ync'] 
    #['AR-SLu']# 'AU-Tum',

    print(site_names)

    for site_name in site_names:
        print('site_name',site_name)
        output_file      = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"+site_name+".nc"
        zscore_threshold = 3 # beyond 3 standard deviation, out of 99.7%
                             # beyond 4 standard deviation, out of 99.349%

        varname       = "TVeg"
        key_word      = "trans"
        key_word_not  = ["evap","transmission","pedo","electron",]
        trans_dict    = check_variable_exists(PLUMBER2_path, varname, site_name, model_names, key_word, key_word_not)
        print(trans_dict)
        make_nc_file(PLUMBER2_path, trans_dict, model_names, site_name, output_file, varname, zscore_threshold)
        gc.collect()

        varname       = "Qle"
        key_word      = 'latent'
        key_word_not  = ['None']
        qle_dict      = check_variable_exists(PLUMBER2_path, varname, site_name, model_names, key_word, key_word_not)
        print(qle_dict)
        make_nc_file(PLUMBER2_path, qle_dict, model_names, site_name, output_file, varname, zscore_threshold)
        gc.collect()

        varname       = "Qh"
        key_word      = 'sensible'
        key_word_not  = ['vegetation','soil','corrected']
        qh_dict       = check_variable_exists(PLUMBER2_path, varname, site_name, model_names, key_word, key_word_not)
        print(qh_dict)
        make_nc_file(PLUMBER2_path, qh_dict, model_names, site_name, output_file, varname, zscore_threshold)
        gc.collect()

        varname       = "NEE"
        key_word      = 'exchange'
        key_word_not  = ['None']
        nee_dict      = check_variable_exists(PLUMBER2_path, varname, site_name, model_names, key_word, key_word_not)
        print(nee_dict)
        make_nc_file(PLUMBER2_path, nee_dict, model_names, site_name, output_file, varname, zscore_threshold)
        gc.collect()

        add_Qle_obs_to_nc_file(PLUMBER2_flux_path, site_name, output_file)
        gc.collect()

        add_Qh_obs_to_nc_file(PLUMBER2_flux_path, site_name, output_file)
        gc.collect()

        add_met_to_nc_file(PLUMBER2_met_path, site_name, output_file)
        gc.collect()

        Qle_Qh_threshold=10
        add_EF_to_nc_file(output_file, zscore_threshold, Qle_Qh_threshold)
        gc.collect()
