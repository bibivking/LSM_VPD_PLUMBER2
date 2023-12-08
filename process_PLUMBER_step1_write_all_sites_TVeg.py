import os
import sys
import gc
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
                var_output['model_'+model_in] = f.variables[model_var_name][:]*3600
            elif var_name == 'NEE' or var_name == 'GPP':
                # convert from umol/m2/s to g C/h
                s2h = 3600.              # s-1 to h-1
                var_output['model_'+model_in] = f.variables[model_var_name][:]*s2h
            else:
                var_output['model_'+model_in] = f.variables[model_var_name][:]

            # read model EF
            model_bin_name             = f"{model_in}_EF"
            try:
                var_output[model_in+'_EF'] = f.variables[model_bin_name][:]
            except:
                var_output[model_in+'_EF'] = np.nan
        elif model_ntime == int(ntime/2):
            print('model ', model_in, ' is hourly, model_ntime is', model_ntime, ' ntime is', ntime)
            model_out_list.append(model_in)

            # put the value of hourly data to the first half hour
            hourly_mask = np.full(ntime,False)
            for t in ntime:
                if t % 2 == 0:
                    hourly_mask[t] = True
            if var_name == 'TVeg':
                var_output['model_'+model_in] = np.nan
                var_output.loc[hourly_mask,'model_'+model_in] =f.variables[model_var_name][:]*3600
            elif var_name == 'NEE' or var_name == 'GPP':
                # convert from umol/m2/s to g C/h
                s2h = 3600.              # s-1 to h-1
                var_output['model_'+model_in] = np.nan
                var_output.loc[hourly_mask,'model_'+model_in] =f.variables[model_var_name][:]*s2h
            else:
                var_output['model_'+model_in] = np.nan
                var_output.loc[hourly_mask,'model_'+model_in] =f.variables[model_var_name][:]

            # read model EF
            model_bin_name             = f"{model_in}_EF"
            try:
                var_output[model_in+'_EF']                  = np.nan
                var_output.loc[hourly_mask, model_in+'_EF'] = f.variables[model_bin_name][:]
            except:
                var_output[model_in+'_EF'] = np.nan

    # read obs values
    if var_name == 'Qle' or var_name == 'Qh':
        var_output['obs'] = f.variables[f"obs_{var_name}"][:]
        model_out_list.append('obs')
        try:
            var_output['obs_cor'] = f.variables[f"obs_{var_name}_cor"][:]
        except:
            var_output['obs_cor'] = np.nan
        model_out_list.append('obs_cor')

    if var_name == 'NEE' or var_name == 'GPP':
        s2h               = 3600.              # s-1 to h-1
        var_output['obs'] = f.variables[f"obs_{var_name}"][:]*s2h

    var_output['obs_EF'] = f.variables["obs_EF"][:]

    # Read VPD and soil moisture information
    var_output['VPD']        = f.variables['VPD'][:]
    var_output['obs_Tair']   = f.variables['obs_Tair'][:]
    var_output['obs_Qair']   = f.variables['obs_Qair'][:]
    var_output['obs_Precip'] = f.variables['obs_Precip'][:]
    var_output['obs_SWdown'] = f.variables['obs_SWdown'][:]

    greenness_file = '/g/data/w97/mm3972/data/PLUMBER2/NoahMPv401/NoahMPv401_UAlb_r1a_'+site_name+'.nc'
    f_green        = nc.Dataset(greenness_file, mode='r')
    var_output['NoahMPv401_greenness'] = f_green.variables['Greenness'][:,0,0]

    # close the file
    f.close()
    f_green.close()

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

def read_LAI_obs(site_name, PLUMBER2_met_path):

    input_file     = glob.glob(PLUMBER2_met_path + site_name+"*.nc")[0]

    f              = nc.Dataset(input_file, mode='r')
    LAI_obs        = f.variables['LAI'][:,0,0]
    LAI_obs        = np.where(LAI_obs==-9999., np.nan, LAI_obs)
    print('LAI_obs',LAI_obs)

    return LAI_obs

def read_LAI_model(site_name, model_with_LAI, model_LAI_name, PLUMBER2_path_input):

    file_path      = glob.glob(PLUMBER2_path_input + model_with_LAI +"/*"+site_name+"*.nc")
    if not file_path:
        LAI_model  = np.nan
    else:
        f              = nc.Dataset(file_path[0])
        LAI_model_tmp  = f.variables[model_LAI_name][:]
        veget          = None

        # Reset missing value
        for attr in ['_FillValue', '_fillValue', 'missing_value']:
            if hasattr(f.variables[model_LAI_name], attr):
                var_FillValue = getattr(f.variables[model_LAI_name], attr)
                LAI_model_tmp = np.where(LAI_model_tmp==var_FillValue, np.nan, LAI_model_tmp)
                break

        if hasattr(f.variables[model_LAI_name], 'dimensions'):
            if 'veget' in f.variables[model_LAI_name].dimensions:
                # print('model_name', model_name, 'site_name', site_name,'has veget demension' )
                veget = f.dimensions['veget'].size

        if veget is not None:
            if veget > 1:
                # if model uses patches
                # read veget fraction
                vegetfrac   = f.variables['vegetfrac']
                # print('model_name', model_name, 'site_name', site_name, 'veget = ', veget, 'vegetfrac =', vegetfrac )

                # initlize Var_tmp_tmp
                ntime     = len(LAI_model_tmp[:,0,0,0])
                LAI_model = np.zeros(ntime)

                # calculate the veget fraction weighted pixel value for each time step
                for i in np.arange(ntime):
                    for j in np.arange(0,veget):
                        LAI_model[i] = LAI_model[i] + LAI_model_tmp[i,j]*vegetfrac[i,j]
            else:
                # if patch == 1
                LAI_model = LAI_model_tmp.reshape(-1)
        else:
            # if model doesn't use patches
            LAI_model = LAI_model_tmp.reshape(-1)
        f.close()
        
    print('LAI_model',LAI_model)
    
    return LAI_model

def calc_hours_after_precip(precip, valid_daily_precip=1,site_name=None):

    s2h     = 60*60

    ntime   = len(precip)
    prec_24 = np.zeros(ntime)

    # calculate 24 hours total precipitation
    for t in np.arange(0,ntime):
        if t < 24:
            prec_24[t] = np.nansum(precip[0:t])*s2h
        else:
            prec_24[t] = np.nansum(precip[t-24:t])*s2h

    # check whether 24-h precipation pass the threshold
    valid_prec = np.where(prec_24 > valid_daily_precip, 1, 0)

    # calcualte hours without precipitation
    accul_hours      = 0
    half_hrs_after_precip = np.zeros(ntime)

    for t in np.arange(ntime):
        accul_hours         = np.where(valid_prec[t] == 1,  0, accul_hours+1)
        half_hrs_after_precip[t] = accul_hours

    # Check the plot
    if 0:
        fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[8,6],sharex=True, sharey=False, squeeze=True) #
        plot       = ax.plot(half_hrs_after_precip, lw=1.0, color='black', alpha=0.3)
        plot       = ax.plot(prec_24*10, lw=1.0, color='green', alpha=0.5)
        plot       = ax.plot(precip*s2h*10, lw=1.0, color='blue', alpha=0.5)

        fig.savefig('./plots/check_'+site_name+'_rain_free_days.png',bbox_inches='tight',dpi=300) # '_30percent'

    return half_hrs_after_precip

def write_spatial_land_days(var_name, site_names, PLUMBER2_path, PLUMBER2_met_path, add_LAI=False, country_code=None):

    # ============= read all sites data ================
    # get veg type info
    IGBP_dict          = read_IGBP_veg_type(site_names, PLUMBER2_met_path)
    print('IGBP_dict has been read')

    # get climate type info
    lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_met_path)
    print('lat_dict and lon_dict have been read')

    clim_class_dict    = {}
    for site_name in site_names:
        clim_class_dict[site_name] = read_climate_class(lat_dict[site_name], lon_dict[site_name])
        gc.collect()
    print('clim_class_dict has been read')

    # read data
    for i, site_name in enumerate(site_names):

        print('site_name',site_name)

        input_file = PLUMBER2_path+site_name+".nc"

        var_output_tmp, model_out_list = read_data(var_name, site_name, input_file)

        # Add veg type
        var_output_tmp['IGBP_type']    = IGBP_dict[site_name]

        # Add climate type
        var_output_tmp['climate_type'] = clim_class_dict[site_name]

        # Add hours after previous valid rainfall
        var_output_tmp['half_hrs_after_precip'] = calc_hours_after_precip(var_output_tmp['obs_Precip'],valid_daily_precip=1,site_name=site_name) # No rain: Less than 1.0 mm, Light rain: 1.0 mm to 10.0 mm

        if add_LAI:
            var_output_tmp['obs_LAI'] = read_LAI_obs(site_name, PLUMBER2_met_path)

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
                    var_output_tmp['model_'+missed_model] = np.nan
                    # var_output_tmp[missed_model+'_EF'] = np.nan

            # connect different sites data together
            var_output = var_output.append(var_output_tmp, ignore_index=True)

        # save the dataframe
        var_output_tmp=None
        gc.collect()


    if add_LAI:
        if country_code !=None:
            var_output.to_csv(f'./txt/all_sites/{var_name}_all_sites_with_LAI_'+country_code+'.csv') # , mode='a', index=False
        else:
            var_output.to_csv(f'./txt/all_sites/{var_name}_all_sites_with_LAI.csv') # , mode='a', index=False
    else:
        if country_code !=None:
            var_output.to_csv(f'./txt/all_sites/{var_name}_all_sites_'+country_code+'.csv') # , mode='a', index=False
        else:
            var_output.to_csv(f'./txt/all_sites/{var_name}_all_sites.csv') # , mode='a', index=False

def add_obs_LAI_to_write_spatial_land_days(var_name, site_names, PLUMBER2_met_path):

    var_output   = pd.read_csv(f'./txt/all_sites/{var_name}_all_sites.csv',usecols=['time','month',
                               'hour','site_name','IGBP_type','climate_type','half_hrs_after_precip'],
                               na_values=[''])
    ntime        = len(var_output)

    var_output['obs_LAI'] = np.nan

    for i, site_name in enumerate(site_names):
        site_mask = (var_output['site_name'] == site_name)
        LAI_tmp =  read_LAI_obs(site_name, PLUMBER2_met_path)
        var_output.loc[site_mask,'obs_LAI'] = LAI_tmp

    var_output.to_csv(f'./txt/all_sites/LAI_all_sites.csv')
    return

def add_model_LAI_to_write_spatial_land_days(var_name, site_names, models_calc_LAI, model_LAI_names, PLUMBER2_path_input):

    var_output   = pd.read_csv(f'./txt/all_sites/LAI_all_sites.csv',na_values=[''])
    ntime        = len(var_output)

    for model_with_LAI in models_calc_LAI:
        var_output[model_with_LAI+'_LAI'] = np.nan

    for i, site_name in enumerate(site_names):
        site_mask = (var_output['site_name'] == site_name)

        for model_with_LAI in models_calc_LAI:
            model_LAI_name = model_LAI_names[model_with_LAI]

            LAI_tmp =  read_LAI_model(site_name, model_with_LAI, model_LAI_name, PLUMBER2_path_input)
            model_ntime = len(LAI_tmp)

            if model_ntime == ntime:
                # if the model output is half-hourly
                var_output.loc[site_mask, model_with_LAI+'_LAI'] = LAI_tmp
            elif model_ntime == int(ntime/2):
                print('model ', model_in, ' is hourly, model_ntime is', model_ntime, ' ntime is', ntime)

                # put the value of hourly data to the first half hour
                LAI_new = np.full(ntime,np.nan)
                for t in ntime:
                    if t % 2 == 0:
                        try:
                            LAI_new[t] = LAI_tmp[int(t/2.)]
                        except:
                            # in case, the site misses LAI data and it is set as np.nan
                            LAI_new[t] = LAI_tmp
                var_output.loc[site_mask, model_with_LAI+'_LAI'] =LAI_new
        gc.collect()
    var_output.to_csv(f'./txt/all_sites/LAI_all_sites.csv')
    return

def add_model_SMtop1m_to_write_spatial_land_days(var_name, site_names, SM_names, PLUMBER2_path):

    # read the variables
    var_output   = pd.read_csv(f'./txt/all_sites/{var_name}_all_sites.csv',usecols=['time','month',
                               'hour','site_name','IGBP_type','climate_type','half_hrs_after_precip'],
                               na_values=[''])

    #
    # ntime        = len(var_output)

    # Initlize the model SMtop1m
    for model_name in SM_names:
        var_output[model_name+'_SMtop1m'] = np.nan

    # Loop accross all sites
    for i, site_name in enumerate(site_names):
        # Get site mask
        site_mask = (var_output['site_name'] == site_name)

        # Read site nc file
        file_path      = glob.glob(PLUMBER2_path + "*"+site_name+"*.nc")
        f_in           = nc.Dataset(file_path[0])

        for model_name in SM_names:
            # if the model output is half-hourly
            try:
                var_output.loc[site_mask, model_name+'_SMtop1m'] = f_in.variables[model_name+'_SMtop1m'][:]
            except:
                # for the few sites missing some models simulations
                var_output.loc[site_mask, model_name+'_SMtop1m'] = f_in.variables['model_mean_SMtop1m'][:]

        var_output.loc[site_mask, 'model_mean_SMtop1m'] = f_in.variables['model_mean_SMtop1m'][:]

    var_output.to_csv(f'./txt/all_sites/SMtop1m_all_sites.csv')
    return

def add_greenness_to_write_spatial_land_days(var_name, site_names):

    var_output   = pd.read_csv(f'./txt/all_sites/LAI_all_sites.csv',na_values=[''])
    ntime        = len(var_output)
    var_output['NoahMPv401_greenness'] = np.nan

    # total_length = 17137992
    # green_frac   = np.zeros(total_length)
    for i, site_name in enumerate(site_names):

        site_mask      = (var_output['site_name'] == site_name)
        greenness_file = '/g/data/w97/mm3972/data/PLUMBER2/NoahMPv401/NoahMPv401_UAlb_r1a_'+site_name+'.nc'
        f_green        = nc.Dataset(greenness_file, mode='r')
        var_output.loc[site_mask,'NoahMPv401_greenness'] = f_green.variables['Greenness'][:,0,0]

    var_output.to_csv(f'./txt/all_sites/LAI_all_sites.csv')

    return

def check_LAI(var_name, site_names, PLUMBER2_met_path):

    var_output            = pd.read_csv(f'./txt/all_sites/LAI_all_sites.csv',na_values=[''])
    ntime                 = len(var_output)

    print("np.any(var_output['obs_LAI'] !=0)",np.any(var_output['obs_LAI'] !=0))

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_met_path   = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    PLUMBER2_path       = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    PLUMBER2_path_input = "/g/data/w97/mm3972/data/PLUMBER2/"

    # The site names
    all_site_path     = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    site_names        = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    # site_names      = ["AU-How","AU-Tum"]

    var_name          = 'TVeg'
    add_LAI           = False
    models_calc_LAI   = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','Noah-MP']
    model_LAI_names   = {'ORC2_r6593':'lai','ORC2_r6593_CO2':'lai','ORC3_r7245_NEE':'lai','ORC3_r8120':'lai',
                         'GFDL':'lai', 'SDGVM':'lai','QUINCY':'LAI','Noah-MP':'LAI'} #

    SM_names, soil_thicknesses = get_model_soil_moisture_info()

    country_code      = None #'AU'
    # site_names  = load_sites_in_country_list(country_code)
    write_spatial_land_days(var_name, site_names, PLUMBER2_path, PLUMBER2_met_path, add_LAI)

    # add_model_SMtop1m_to_write_spatial_land_days(var_name, site_names, SM_names, PLUMBER2_path)

    # # === Together ===
    # add_obs_LAI_to_write_spatial_land_days(var_name, site_names, PLUMBER2_met_path)
    # add_model_LAI_to_write_spatial_land_days(var_name, site_names, models_calc_LAI, model_LAI_names, PLUMBER2_path_input)
    # add_greenness_to_write_spatial_land_days(var_name, site_names)
    # # ================

    # check_LAI(var_name, site_names, PLUMBER2_met_path)
