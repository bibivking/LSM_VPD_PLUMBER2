'''
Including:
    def time_mask
    def read_CMIP6
    def calculate_EF
    def make_CMIP6_nc_file
    def make_EF_extremes_nc_file
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from matplotlib.patches import Polygon
import copy
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

def read_CMIP6(fname, model_name, var_name, time_s, time_e, regrid_to=None):

    # Open file
    f         = nc.Dataset(fname, mode='r')

    # Read variable
    var_tmp   = f.variables[var_name][:]

    # Read time
    time_tmp  = nc.num2date(f.variables['time'][:], f.variables['time'].units,
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
    time      = time_tmp[time_cood]
    var       = var_tmp[time_cood,:,:]


    # Regrid ACCESS-ESM1-5 to ACCESS-CM2
    if regrid_to == "ACCESS-CM2" and model_name == 'ACCESS-ESM1-5':

        # Read lat_in and lon_in
        lat_in  = f.variables['lat'][:]
        lon_in  = f.variables['lon'][:]

        # Read lat_out and lon_out
        f_cm    = nc.Dataset('/g/data/w97/mm3972/data/CMIP6_data/Processed_CMIP6_data/ssp370/hfls/ACCESS-CM2/r1i1p1f1/hfls_ACCESS-CM2_ssp370_r1i1p1f1_2015_2100_ssp370_regrid_setgrid.nc', mode='r')
        lat_out = f_cm.variables['lat'][:]
        lon_out = f_cm.variables['lon'][:]

        # Set dimensions
        ntime = len(var[:,0,0])
        nlat  = len(lat_out)
        nlon  = len(lon_out)
        Var   = np.zeros((ntime,nlat,nlon))

        for i in np.arange(ntime):
            Var[i,:,:]    = regrid_data(lat_in, lon_in, lat_out, lon_out, var[i,:,:], method='nearest')
        f_cm.close()
    else:
        Var = var
    f.close()

    return time, Var

def calculate_EF(Qle, Qh):

    # Set daily Qle+Qh percent
    # Qle_Qh_threshold=10
    # EF_tmp = np.where(np.all([Qh+Qle > Qle_Qh_threshold, Qh>0],axis=0), Qle/(Qh+Qle), np.nan)
    
    EF_tmp = np.where(np.all([Qle>0, Qh>0],axis=0), Qle/(Qh+Qle), np.nan)
    EF     = np.where(EF_tmp<0, np.nan, EF_tmp)

    return EF

def make_CMIP6_multiple_nc_file(CMIP6_data_path, output_file, scenario, time_s, time_e):

    '''
    Process multiple CMIP6 models 
    '''

    print('scenario',scenario)

    # Get CMIP6 model list
    model_list = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6',
                  'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']
    
    landsea_list = {'ACCESS-CM2':'/g/data/fs38/publications/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/historical/r1i1p1f1/fx/sftlf/gn/v20191108',
                    'BCC-CSM2-MR':'/g/data/oi10/replicas/CMIP6/GMMIP/BCC/BCC-CSM2-MR/hist-resIPO/r1i1p1f1/fx/sftlf/gn/v20190613',
                    'CMCC-CM2-SR5':'/g/data/oi10/replicas/CMIP6/CMIP/CMCC/CMCC-CM2-SR5/historical/r1i1p1f1/fx/sftlf/gn/v20200616',
                    'CMCC-ESM2':'/g/data/oi10/replicas/CMIP6/CMIP/CMCC/CMCC-ESM2/historical/r1i1p1f1/fx/sftlf/gn/v20210114',
                    'EC-Earth3':'/g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical/r1i1p1f1/fx/sftlf/gr/v20200310', 
                    'KACE-1-0-G':'/g/data/oi10/replicas/CMIP6/CMIP/NIMS-KMA/KACE-1-0-G/historical/r1i1p1f1/Lmon/mrsos/gr/v20191002',
                    'MIROC6':'/g/data/oi10/replicas/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/fx/sftlf/gn/v20190311',
                    'MIROC-ES2L':'/g/data/oi10/replicas/CMIP6/CMIP/MIROC/MIROC-ES2L/historical/r1i1p1f2/fx/sftlf/gn/v20190823',
                    'MPI-ESM1-2-HR':'/g/data/oi10/replicas/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/fx/sftlf/gn/v20190710',
                    'MPI-ESM1-2-LR':'/g/data/oi10/replicas/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710', 
                    'MRI-ESM2-0':'/g/data/oi10/replicas/CMIP6/CMIP/MRI/MRI-ESM2-0/historical/r1i1p1f1/fx/sftlf/gn/v20190603'}
    

    # ======Loop each model ======
    for model_name in model_list:

        print('model', model_name)

        # === Get variable from Processed_CMIP6_data ===
        # latent heat flux (hfls) files
        fname_hfls  = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hfls/{model_name}/*/*.nc'))[0]
        
        # sensible heat flux (hfss) files
        fname_hfss  = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hfss/{model_name}/*/*.nc'))[0]

        # air temperature (tas) files
        fname_tas   = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/tas/{model_name}/*/*.nc'))[0]

        # shortwave radiation flux (rsds) files
        fname_rsds  = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/rsds/{model_name}/*/*.nc'))[0]
        
        # air pressure (ps) files
        fname_ps    = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/ps/{model_name}/*/*.nc'))[0]

        # specific humidity (huss) files
        fname_huss  = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/huss/{model_name}/*/*.nc'))[0]

        # # relative humidity (hurs) files
        # fname_hurs  = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hurs/{model_name}/*/*.nc'))[0]

        # === Get the same griding and time period data ===
        time, var_hfls  = read_CMIP6(fname_hfls, model_name, 'hfls', time_s, time_e)
        time, var_hfss  = read_CMIP6(fname_hfss, model_name, 'hfss', time_s, time_e)
        time, var_tas   = read_CMIP6(fname_tas,  model_name, 'tas',  time_s, time_e)
        time, var_rsds  = read_CMIP6(fname_rsds, model_name, 'rsds', time_s, time_e)
        time, var_ps    = read_CMIP6(fname_ps,   model_name, 'ps',   time_s, time_e)
        time, var_huss  = read_CMIP6(fname_huss, model_name, 'huss', time_s, time_e)
        # time, var_hurs  = read_CMIP6(fname_hurs, model_name, 'hurs', time_s, time_e)

        # === Obtain landsea ===
        fname_landsea   = sorted(glob.glob(f'{landsea_list[model_name]}/*.nc'))[0]
        landsea_varname = (fname_landsea).split("/")[12]

        # Read landsea_lat and landsea_lon
        f_landsea = nc.Dataset(fname_landsea, mode='r')
        landsea_lat = f_landsea.variables['lat'][:]
        landsea_lon = f_landsea.variables['lon'][:]

        # Read landsea var
        if len(np.shape(f_landsea.variables[landsea_varname])) == 2:            
            landsea_var = f_landsea.variables[landsea_varname][:]
        elif len(np.shape(f_landsea.variables[landsea_varname])) == 3:          
            landsea_var = f_landsea.variables[landsea_varname][0,:,:]
        
        # Adjust the original CMIP6 landsea since Anna's R script changed lat & lon
        # in the processed CMIP6 files to lat [-90 ~ 90] and lon [-180,180].
            
        # Adjust lat from 90 ~ -90 to -90 ~ 90
        if landsea_lat[0] > landsea_lat[-1]:
            landsea_var[:] = landsea_var[::-1,:]

        # Adjust lon from 0 ~ 360 to -180 ~ 180
        if landsea_lon[0] >= 0:  
            landsea_lon[:] = np.where(landsea_lon>180, landsea_lon-360, landsea_lon)
            landsea_var[:] = landsea_var[:,np.argsort(landsea_lon)]
        
        # Set missing values as 0
        for attr in ['_FillValue', '_fillValue', 'missing_value']:
            if hasattr(f_landsea.variables[landsea_varname], attr):
                var_FillValue = getattr(f_landsea.variables[landsea_varname], attr)
                landsea_var   = np.where(landsea_var==var_FillValue, 0, landsea_var)

        # Set land == 1 and ocean == 0 
        landsea_var   = np.where(landsea_var>0, 1, landsea_var)

        f_landsea.close()    

        # === Calculate VPD === 
        var_vpd       = calculate_VPD_by_Qair(var_huss, var_tas, var_ps)

        # === Calcuate EF ===
        var_EF        = calculate_EF(var_hfls,  var_hfss)
        
        # === Create nc file and set time ===
        if not os.path.exists(output_file):

            # Make time series
            time_init   = datetime(1970,1,1,0,0,0)
            time_series = []
            for t in time:
                time_series.append((t-time_init).total_seconds())
            print(time_series)

            # Create the nc file
            f = nc.Dataset(output_file, 'w', format='NETCDF4')

            f.history           = "Created by: %s" % (os.path.basename(__file__))
            f.creation_date     = "%s" % (datetime.now())
            f.description       = 'CMIP6 '+scenario+' ensemble mean, made by MU Mengyuan'
            f.Conventions       = "CF-1.0"

            # Set time dimension
            f.createDimension('time', len(time))

            Time                = f.createVariable('time', 'f4', ('time'))
            Time.standard_name  = 'time'
            Time.units          = "seconds since 1970-01-01 00:00:00"
            Time[:]             = time_series
            f.close()    
        
        # === Set other variables ===
        f = nc.Dataset(output_file, 'r+', format='NETCDF4')

        # Read lat and lon dimension
        f_lat_lon    = nc.Dataset(fname_hfls, mode='r')
        lat          = f_lat_lon.variables['lat'][:]
        lon          = f_lat_lon.variables['lon'][:]
        f_lat_lon.close()

        # Set lat and lon out
        f.createDimension(model_name+'_lat', len(lat))
        f.createDimension(model_name+'_lon', len(lon))

        # Latitude
        Lat                = f.createVariable('lat', 'f4', ('lat'))
        Lat.standard_name  = 'latitude'
        Lat[:]             = lat

        # Longitude
        Lon                = f.createVariable('lon', 'f4', ('lon'))
        Lon.standard_name  = 'longitude'
        Lon[:]             = lon

        # Latent heat flux
        Qle                = f.createVariable(model_name+'_Qle', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        Qle.standard_name  = 'Latent heat flux'
        Qle.units          = 'W m-2'
        Qle[:]             = var_hfls

        # Sensible heat flux
        Qh                = f.createVariable(model_name+'_Qh', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        Qh.standard_name  = 'Sensible heat flux'
        Qh.units          = 'W m-2'
        Qh[:]             = var_hfss

        # Relative humidity
        # RH                = f.createVariable(model_name+'_RH', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        # RH.standard_name  = 'Relative humidity'
        # RH.units          = '%'
        # RH[:]             = var_hurs

        # Air temperature
        Tair                = f.createVariable(model_name+'_Tair', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        Tair.standard_name  = 'Near-Surface Air Temperature'
        Tair.units          = 'K'
        Tair[:]             = var_tas

        # Air pressure
        Press                = f.createVariable(model_name+'_Press', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        Press.standard_name  = 'Surface Air Pressure'
        Press.units          = 'Pa'
        Press[:]             = var_ps

        # Air pressure
        SWdown                = f.createVariable(model_name+'_SWdown', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        SWdown.standard_name  = 'Surface Downwelling Shortwave Radiation'
        SWdown.units          = 'W m-2'
        SWdown[:]             = var_rsds

        Qair                = f.createVariable(model_name+'_Qair', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        Qair.standard_name  = 'Near-Surface Specific Humidity'
        Qair.units          = 'W m-2'
        Qair[:]             = var_huss

        VPD                = f.createVariable(model_name+'_VPD', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        VPD.standard_name  = 'Vapor pressure deficit'
        VPD.units          = 'kPa'
        VPD[:]             = var_vpd
        
        EF                 = f.createVariable(model_name+'_EF', 'f4', ('time', model_name+'_lat', model_name+'_lon'))
        EF.standard_name   = 'Evaporative fraction'
        EF.units           = 'fraction'
        EF[:]              = var_EF

        Landsea               = f.createVariable(model_name+'_landsea', 'f4', (model_name+'_lat', model_name+'_lon'))
        Landsea.standard_name = 'Landsea mask (land=1, sea=0)'
        Landsea.units         = '1/0'
        Landsea[:]            = landsea_var

        f.close()

        # Free memory
        lat       = None
        lon       = None        
        var_hfls  = None
        var_hfss  = None
        var_tas   = None
        var_rsds  = None
        var_ps    = None
        var_huss  = None
        var_vpd   = None
        var_EF    = None
        landsea_var  = None

        Lat       = None
        Lon       = None        
        Qle       = None
        Qh        = None
        Tair      = None
        SWdown    = None
        Press     = None
        Qair      = None
        VPD       = None
        EF        = None
        Landsea   = None

def make_CMIP6_ACCESS_nc_file(CMIP6_data_path, CMIP6_out_path, scenario, time_s, time_e, regrid_to="ACCESS-CM2"):

    '''
    Process ACCESS-CM2 and ACCESS-ESM1-5 raw CMIP6 data
    '''

    # Open files
    fname_cm_hfls   = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hfls/ACCESS-CM2/*/*.nc'))[0]
    fname_esm_hfls  = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hfls/ACCESS-ESM1-5/*/*.nc'))[0]

    fname_cm_hfss   = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hfss/ACCESS-CM2/*/*.nc'))[0]
    fname_esm_hfss  = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hfss/ACCESS-ESM1-5/*/*.nc'))[0]

    fname_cm_hurs   = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hurs/ACCESS-CM2/*/*.nc'))[0]
    fname_esm_hurs  = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hurs/ACCESS-ESM1-5/*/*.nc'))[0]

    fname_cm_tas    = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/tas/ACCESS-CM2/*/*.nc'))[0]
    fname_esm_tas   = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/tas/ACCESS-ESM1-5/*/*.nc'))[0]

    # Set lat and lon out
    f              = nc.Dataset(fname_cm_hurs, mode='r')
    lat_out        = f.variables['lat'][:]
    lon_out        = f.variables['lon'][:]
    f.close()

    # Get the same griding and time period data
    time, var_cm_hfls  = read_CMIP6(fname_cm_hfls, 'ACCESS-CM2', 'hfls', time_s, time_e, regrid_to)
    time, var_esm_hfls = read_CMIP6(fname_esm_hfls,'ACCESS-ESM1-5', 'hfls', time_s, time_e, regrid_to)

    time, var_cm_hfss  = read_CMIP6(fname_cm_hfss, 'ACCESS-CM2', 'hfss', time_s, time_e, regrid_to)
    time, var_esm_hfss = read_CMIP6(fname_esm_hfss,'ACCESS-ESM1-5', 'hfss', time_s, time_e, regrid_to)

    time, var_cm_hurs  = read_CMIP6(fname_cm_hurs, 'ACCESS-CM2', 'hurs', time_s, time_e, regrid_to)
    time, var_esm_hurs = read_CMIP6(fname_esm_hurs,'ACCESS-ESM1-5', 'hurs', time_s, time_e, regrid_to)

    time, var_cm_tas   = read_CMIP6(fname_cm_tas, 'ACCESS-CM2', 'tas', time_s, time_e, regrid_to)
    time, var_esm_tas  = read_CMIP6(fname_esm_tas,'ACCESS-ESM1-5', 'tas', time_s, time_e, regrid_to)

    # Caclulate ensemble average
    var_hfls_mean = (var_cm_hfls + var_esm_hfls)/2
    var_hfss_mean = (var_cm_hfss + var_esm_hfss)/2
    var_hurs_mean = (var_cm_hurs + var_esm_hurs)/2
    var_tas_mean  = (var_cm_tas + var_esm_tas)/2

    # Calculate VPD
    VPD_mean      = calculate_VPD_by_RH(var_hurs_mean,  var_tas_mean)

    # Calcuate EF
    EF_mean       = calculate_EF(var_hfls_mean,  var_hfss_mean)

    # put all data in nc file
    output_file   = CMIP6_out_path + scenario + '.nc'
    print('output_file is ',output_file)

    print('time', time)
    time_init   = datetime(1970,1,1,0,0,0)
    time_series = []
    for t in time:
        time_series.append((t-time_init).days)
    print(time_series)

    f = nc.Dataset(output_file, 'w', format='NETCDF4')

    ### Create nc file ###
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = 'CMIP6 '+scenario+' ensemble mean, made by MU Mengyuan'
    f.Conventions       = "CF-1.0"

    # set dimensions
    f.createDimension('time', len(time))
    f.createDimension('lat', len(lat_out))
    f.createDimension('lon', len(lon_out))

    Time                = f.createVariable('time', 'f4', ('time'))
    Time.standard_name  = 'time'
    Time.units          = "days since 1970-01-01 00:00:00"
    Time[:]             = time_series

    Lat                = f.createVariable('lat', 'f4', ('lat'))
    Lat.standard_name  = 'latitude'
    Lat[:]             = lat_out

    Lon                = f.createVariable('lon', 'f4', ('lon'))
    Lon.standard_name  = 'longitude'
    Lon[:]             = lon_out

    Qle                = f.createVariable('Qle', 'f4', ('time', 'lat', 'lon'))
    Qle.standard_name  = 'Latent heat flux'
    Qle.units          = 'W m-2'
    Qle[:]             = var_hfls_mean

    Qh                = f.createVariable('Qh', 'f4', ('time', 'lat', 'lon'))
    Qh.standard_name  = 'Sensible heat flux'
    Qh.units          = 'W m-2'
    Qh[:]             = var_hfss_mean

    RH                = f.createVariable('RH', 'f4', ('time', 'lat', 'lon'))
    RH.standard_name  = 'Relative humidity'
    RH.units          = '%'
    RH[:]             = var_hurs_mean

    Tair                = f.createVariable('Tair', 'f4', ('time', 'lat', 'lon'))
    Tair.standard_name  = 'Near-Surface Air Temperature'
    Tair.units          = 'K'
    Tair[:]             = var_tas_mean

    VPD                = f.createVariable('VPD', 'f4', ('time', 'lat', 'lon'))
    VPD.standard_name  = 'Vapor pressure deficit'
    VPD.units          = 'kPa'
    VPD[:]             = VPD_mean

    EF                 = f.createVariable('EF', 'f4', ('time', 'lat', 'lon'))
    EF.standard_name   = 'Evaporative fraction'
    EF.units           = 'fraction'
    EF[:]              = EF_mean

    f.close()

def make_EF_extremes_nc_file(CMIP6_out_path, scenario, percent=15):

    '''
    Output top/bom xx percentiles of the CMIP6 nc files 
    '''

    # Reading data
    input_file     = CMIP6_out_path + scenario + '.nc'

    # Read EF
    print('Read EF')
    f              = nc.Dataset(input_file, mode='r')
    time           = f.variables['time'][:]
    lat_out        = f.variables['lat'][:]
    lon_out        = f.variables['lon'][:]
    ef             = f.variables['EF'][:]
    qle            = f.variables['Qle'][:]
    vpd            = f.variables['VPD'][:]
    f.close()

    # calculate EF percentile, ignoring nan
    print('calculate EF percentile')
    p_bot_ef       = np.nanpercentile(ef, percent, axis=0)
    p_top_ef       = np.nanpercentile(ef, 100-percent, axis=0)


    # save data with EF percentile range
    ef_bot         = np.where(ef<p_bot_ef, ef, np.nan)
    ef_top         = np.where(ef>p_top_ef, ef, np.nan)
    qle_bot        = np.where(ef<p_bot_ef, qle, np.nan)
    qle_top        = np.where(ef>p_top_ef, qle, np.nan)
    vpd_bot        = np.where(ef<p_bot_ef, vpd, np.nan)
    vpd_top        = np.where(ef>p_top_ef, vpd, np.nan)

    gc.collect()

    # # Decide how many data wanted in the new file
    # new_length     = round( ntime * (percent/100) )
    # index_bot      = new_length
    # index_top      = ntime - new_length

    output_files   = [ CMIP6_out_path + scenario + '_EF_bot_'+str(percent)+'th_percent.nc',
                       CMIP6_out_path + scenario + '_EF_top_'+str(percent)+'th_percent.nc']

    for output_file in output_files:

        f = nc.Dataset(output_file, 'w', format='NETCDF4')

        ### Create nc file ###
        f.history           = "Created by: %s" % (os.path.basename(__file__))
        f.creation_date     = "%s" % (datetime.now())

        if 'bot' in output_file:
            f.description   = 'bottom '+str(percent)+' percent of EF in CMIP6 '+scenario+' ensemble mean, made by MU Mengyuan'
        elif 'top' in output_file:
            f.description   = 'top '+str(percent)+' percent of EF in CMIP6 '+scenario+' ensemble mean, made by MU Mengyuan'

        f.Conventions       = "CF-1.0"

        # set dimensions
        f.createDimension('lat',  len(lat_out))
        f.createDimension('lon',  len(lon_out))
        f.createDimension('time', len(time))

        Time               = f.createVariable('time', 'f4', ('time'))
        Time.standard_name = 'time'
        Time[:]            = time
        
        Lat                = f.createVariable('lat', 'f4', ('lat'))
        Lat.standard_name  = 'latitude'
        Lat[:]             = lat_out

        Lon                = f.createVariable('lon', 'f4', ('lon'))
        Lon.standard_name  = 'longitude'
        Lon[:]             = lon_out

        Qle                = f.createVariable('Qle', 'f4', ('time', 'lat', 'lon'))
        Qle.standard_name  = 'Latent heat flux'
        Qle.units          = 'W m-2'

        if 'bot' in output_file:
            Qle[:]         = qle_bot[:]
        elif 'top' in output_file:
            Qle[:]         = qle_top[:]

        VPD                = f.createVariable('VPD', 'f4', ('time', 'lat', 'lon'))
        VPD.standard_name  = 'Vapor pressure deficit'
        VPD.units          = 'kPa'

        if 'bot' in output_file:
            VPD[:]         = vpd_bot[:]
        elif 'top' in output_file:
            VPD[:]         = vpd_top[:]

        EF                 = f.createVariable('EF', 'f4', ('time', 'lat', 'lon'))
        EF.standard_name   = 'Evaporative fraction'
        EF.units           = 'fraction'

        if 'bot' in output_file:
            EF[:]         = ef_bot[:]
        elif 'top' in output_file:
            EF[:]         = ef_top[:]

        Num               = f.createVariable('numbers', 'f4', ('lat', 'lon'))

        if 'bot' in output_file:
            Num.standard_name = 'number of data points'
            Num[:]            = np.sum(np.where(np.isnan(ef_bot[:]),0,1),axis=0)
        elif 'top' in output_file:
            Num.standard_name = 'number of data points'
            Num[:]            = np.sum(np.where(np.isnan(ef_top[:]),0,1),axis=0)
    
        Lat = None
        Lon = None
        Qle = None
        VPD = None
        EF  = None

        f.close()

    return

if __name__ == "__main__":

    # Read files
    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/mm3972/data/CMIP6_3hr_data/Processed_CMIP6_data/"
    CMIP6_out_path    = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_3hourly/"
    scenarios         = ['historical']# , 'ssp245']

    # read CMIP6 data
    for scenario in scenarios:

        if scenario == 'historical':
            time_s  = datetime(1985,1,1,0,0,0)
            time_e  = datetime(2015,1,1,0,0,0)
        else:
            time_s  = datetime(2070,1,1,0,0,0)
            time_e  = datetime(2100,1,1,0,0,0)

        # make_CMIP6_ACCESS_nc_file(CMIP6_data_path, CMIP6_out_path, scenario, time_s, time_e)
        # percent = 15
        # make_EF_extremes_nc_file(CMIP6_out_path, scenario, percent=percent)
        # gc.collect()

        output_file = CMIP6_out_path + scenario + '.nc'
        make_CMIP6_multiple_nc_file(CMIP6_data_path, output_file, scenario, time_s, time_e)
