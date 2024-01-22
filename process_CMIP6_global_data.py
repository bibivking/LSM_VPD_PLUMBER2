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

def read_CMIP6(fname, model_name, var_name, time_s, time_e):

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
    if model_name == 'ACCESS-ESM1-5':

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

def make_CMIP6_nc_file(CMIP6_data_path, CMIP6_out_path, scenario, time_s, time_e):

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
    time, var_cm_hfls  = read_CMIP6(fname_cm_hfls, 'ACCESS-CM2', 'hfls', time_s, time_e)
    time, var_esm_hfls = read_CMIP6(fname_esm_hfls,'ACCESS-ESM1-5', 'hfls', time_s, time_e)

    time, var_cm_hfss  = read_CMIP6(fname_cm_hfss, 'ACCESS-CM2', 'hfss', time_s, time_e)
    time, var_esm_hfss = read_CMIP6(fname_esm_hfss,'ACCESS-ESM1-5', 'hfss', time_s, time_e)

    time, var_cm_hurs  = read_CMIP6(fname_cm_hurs, 'ACCESS-CM2', 'hurs', time_s, time_e)
    time, var_esm_hurs = read_CMIP6(fname_esm_hurs,'ACCESS-ESM1-5', 'hurs', time_s, time_e)

    time, var_cm_tas   = read_CMIP6(fname_cm_tas, 'ACCESS-CM2', 'tas', time_s, time_e)
    time, var_esm_tas  = read_CMIP6(fname_esm_tas,'ACCESS-ESM1-5', 'tas', time_s, time_e)

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

    # Reading data
    input_file     = CMIP6_out_path + scenario + '.nc'

    # Read EF
    print('Read EF')
    f              = nc.Dataset(input_file, mode='r')
    ef             = f.variables['EF'][:]
    f.close()

    # Sorting EF
    print('Sorting EF')
    ef_sorted      = np.sort(ef, axis=0)
    rearranged     = np.argsort(ef, axis=0)
    ef             = None
    gc.collect()

    # Read qle
    print('Read qle')
    f              = nc.Dataset(input_file, mode='r')
    qle            = f.variables['Qle'][:]
    f.close()
    qle_rearranged = qle[rearranged]
    qle            = None
    gc.collect()

    # Read vpd
    print('Read vpd')
    f              = nc.Dataset(input_file, mode='r')
    vpd            = f.variables['VPD'][:]
    f.close()
    vpd_rearranged = vpd[rearranged]
    vpd            = None
    gc.collect()

    # Read ntime, lat, lon
    print('Read ntime, lat, lon')
    f              = nc.Dataset(input_file, mode='r')
    ntime          = len(f.variables['time'][:])
    lat_out        = f.variables['lat'][:]
    lon_out        = f.variables['lon'][:]
    f.close()
    gc.collect()

    # Decide how many data wanted in the new file
    new_length     = round( ntime * (percent/100) )
    index_bot      = new_length
    index_top      = ntime - new_length

    output_files   = [ CMIP6_out_path + scenario + '_EF_bot_'+percent+'percent.nc',
                        CMIP6_out_path + scenario + '_EF_top_'+percent+'percent.nc']

    for output_file in output_files:

        f = nc.Dataset(output_file, 'w', format='NETCDF4')

        ### Create nc file ###
        f.history           = "Created by: %s" % (os.path.basename(__file__))
        f.creation_date     = "%s" % (datetime.now())

        if 'bot' in output_file:
            f.description   = 'bottom '+percent+' percent of EF in CMIP6 '+scenario+' ensemble mean, made by MU Mengyuan'
        elif 'top' in output_file:
            f.description   = 'top '+percent+' percent of EF in CMIP6 '+scenario+' ensemble mean, made by MU Mengyuan'

        f.Conventions       = "CF-1.0"

        # set dimensions
        f.createDimension('lat',  len(lat_out))
        f.createDimension('lon',  len(lon_out))
        f.createDimension('rank', new_length)

        Lat                = f.createVariable('lat', 'f4', ('lat'))
        Lat.standard_name  = 'latitude'
        Lat[:]             = lat_out

        Lon                = f.createVariable('lon', 'f4', ('lon'))
        Lon.standard_name  = 'longitude'
        Lon[:]             = lon_out

        Qle                = f.createVariable('Qle', 'f4', ('rank', 'lat', 'lon'))
        Qle.standard_name  = 'Latent heat flux'
        Qle.units          = 'W m-2'

        if 'bot' in output_file:
            Qle[:]         = qle_rearranged[:index_bot,:,:]
        elif 'top' in output_file:
            Qle[:]         = qle_rearranged[index_top:,:,:]

        VPD                = f.createVariable('VPD', 'f4', ('rank', 'lat', 'lon'))
        VPD.standard_name  = 'Vapor pressure deficit'
        VPD.units          = 'kPa'

        if 'bot' in output_file:
            VPD[:]         = vpd_rearranged[:index_bot,:,:]
        elif 'top' in output_file:
            VPD[:]         = vpd_rearranged[index_top:,:,:]

        EF                 = f.createVariable('EF', 'f4', ('rank', 'lat', 'lon'))
        EF.standard_name   = 'Evaporative fraction'
        EF.units           = 'fraction'

        if 'bot' in output_file:
            EF[:]         = ef_sorted[:index_bot,:,:]
        elif 'top' in output_file:
            EF[:]         = ef_sorted[index_top:,:,:]

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
    CMIP6_data_path   = "/g/data/w97/mm3972/data/CMIP6_data/Processed_CMIP6_data/"
    CMIP6_out_path    = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6_daily/"
    scenarios         = ['historical','ssp126','ssp245','ssp370']

    # read CMIP6 data
    for scenario in scenarios:

        if scenario == 'historical':
            time_s  = datetime(1985,1,1,0,0,0)
            time_e  = datetime(2015,1,1,0,0,0)
        else:
            time_s  = datetime(2070,1,1,0,0,0)
            time_e  = datetime(2100,1,1,0,0,0)

        # make_CMIP6_nc_file(CMIP6_data_path, CMIP6_out_path, scenario, time_s, time_e)
        # percent = 15
        # make_EF_extremes_nc_file(CMIP6_out_path, scenario, percent=percent)
        gc.collect()
