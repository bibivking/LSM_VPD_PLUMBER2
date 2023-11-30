import os
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

def make_landsea_mask_file(CMIP6_data_path, CMIP6_out_path, model_out_list):

    # use CABLE landsea file
    CABLE_mask_file = '/g/data/w97/mm3972/model/cable/src/CABLE-AUX/offline/mmy_gridinfo_Global/gridinfo_mmy_MD_elev_orig_std_avg-sand_landmask.nc'
    f_mask          = nc.Dataset(CABLE_mask_file, mode='r')
    landsea_tmp     = f_mask.variables['landsea'][:]
    lat_in          = f_mask.variables['latitude'][:]
    lon_in_tmp      = f_mask.variables['longitude'][:]

    # Change the central of latitude
    landsea         = copy.deepcopy(landsea_tmp)

    landsea[:,:360] = landsea_tmp[:,360:]
    landsea[:,360:] = landsea_tmp[:,:360]
    lon_in          = np.arange(-179.75, 180.25, 0.5)

    print('lon_in_tmp', lon_in_tmp)
    print('lon_in', lon_in)

    # loop each CMIP model
    for model_name in model_out_list:

        print('model_name', model_name)

        # Set output names
        output_file     = CMIP6_out_path + model_name + '_landsea_mask.nc'

        # Get output lat and lon
        file_names_hurs = sorted(glob.glob(f'{CMIP6_data_path}historical/hurs/{model_name}/*/*.nc'))[0]
        print('file_names_hurs',file_names_hurs)
        f_hurs          = nc.Dataset(file_names_hurs, mode='r')

        # Read lat and lon
        try:
            lat_out  = f_hurs.variables['lat'][:]
            lon_out  = f_hurs.variables['lon'][:]
        except:
            lat_out  = f_hurs.variables['latitude'][:]
            lon_out  = f_hurs.variables['longitude'][:]

        landsea_regrid  = regrid_data(lat_in, lon_in, lat_out, lon_out, landsea, method='nearest')

        f = nc.Dataset(output_file, 'w', format='NETCDF4')

        ### Create nc file ###
        f.history           = "Created by: %s" % (os.path.basename(__file__))
        f.creation_date     = "%s" % (datetime.now())
        f.description       = 'CMIP6 '+model_name+' landsea maskfile, made by MU Mengyuan'
        f.Conventions       = "CF-1.0"

        # set time dimensions
        f.createDimension('lat', len(lat_out))
        f.createDimension('lon', len(lon_out))

        Landsea                = f.createVariable('landsea', 'f4', ('lat','lon'))
        Landsea.standard_name  = 'landsea (0:land; 1:sea)'
        Landsea[:]             = landsea_regrid

        Lat                = f.createVariable('lat', 'f4', ('lat'))
        Lat.standard_name  = 'latitude'
        Lat[:]             = lat_out

        Lon                = f.createVariable('lon', 'f4', ('lon'))
        Lon.standard_name  = 'longitude'
        Lon[:]             = lon_out

        f.close()

    return

def calculate_model_mean_land_VPD(CMIP6_data_path, CMIP6_out_path, scenario, model_out_list):

    # select the site information from each CMIP6 file
    for model_name in model_out_list:
        print('model_name',model_name)

        file_hurs   = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hurs/{model_name}/*/*.nc'))[0]
        file_tas    = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/tas/{model_name}/*/*.nc'))[0]
        file_mask   = CMIP6_out_path + model_name + '_landsea_mask.nc'

        # Get model name
        f_hurs   = nc.Dataset(file_hurs, mode='r')
        f_tas    = nc.Dataset(file_tas, mode='r')

        rh       = f_hurs.variables['hurs'][:]
        tair     = f_tas.variables['tas'][:]
        VPD_tmp  = calculate_VPD_by_RH(rh, tair)

        # Read lat and lon
        try:
            latitude  = f_hurs.variables['lat'][:]
            longitude = f_hurs.variables['lon'][:]
        except:
            latitude  = f_hurs.variables['latitude'][:]
            longitude = f_hurs.variables['longitude'][:]

        # Get land sea mask
        f_mask          = nc.Dataset(file_mask, mode='r')
        landsea         = f_mask.variables['landsea'][:]

        # Read time
        time_tmp  = nc.num2date(f_hurs.variables['time'][:],f_hurs.variables['time'].units,
                    only_use_cftime_datetimes=False, calendar=f_hurs.variables['time'].calendar) # only_use_python_datetimes=True,

        # To solve the inconsistancy in time coordinate
        for i, t in enumerate(time_tmp):
            year   = t.year
            month  = t.month
            day    = t.day
            hour   = t.hour
            minute = t.minute
            second = t.second
            microsecond    = t.microsecond
            time_tmp[i]    = datetime(year, month, day, hour, minute, second, microsecond)
            VPD_tmp[i,:,:] = np.where(landsea == 0, VPD_tmp[i,:,:], np.nan)

        # select time periods
        time_cood = time_mask(time_tmp, time_s, time_e)

        # make new time cooridate
        time      = time_tmp[time_cood]

        # Read variable
        VPD       = VPD_tmp[time_cood,:,:]
        VPD_mean  = np.nanmean(VPD, axis=(1,2))
        VPD_top   = np.zeros(len(VPD_mean))
        VPD_bot   = np.zeros(len(VPD_mean))

        for t in np.arange(len(VPD_mean)):
            mask_temp = ~ np.isnan(VPD[t,:,:])
            VPD_top[t] = np.percentile(VPD[t][mask_temp], 75)
            VPD_bot[t] = np.percentile(VPD[t][mask_temp], 25)

        # print('VPD_mean', VPD_mean)
        # print('VPD_top', VPD_top)
        # print('VPD_bot', VPD_bot)

        data_out              = pd.DataFrame(time, columns=['time'])
        data_out['VPD_mean']  = VPD_mean
        data_out['VPD_top']   = VPD_top
        data_out['VPD_bot']   = VPD_bot

        data_out.to_csv(f'./txt/CMIP6/{model_name}_VPD_{scenario}_global_land.csv')

        f_hurs.close()
        f_tas.close()
        f_mask.close()

        gc.collect()
    return

def calculate_model_VPD_metrics(CMIP6_data_path, CMIP6_out_path, scenario, model_out_list, time_s, time_e,
                                region=None, region_name='AU'):

    # select the site information from each CMIP6 file
    for j, model_name in enumerate(model_out_list):
        print('model_name',model_name)

        file_hurs   = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hurs/{model_name}/*/*.nc'))[0]
        file_tas    = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/tas/{model_name}/*/*.nc'))[0]
        file_mask   = CMIP6_out_path + model_name + '_landsea_mask.nc'

        # Get model name
        f_hurs   = nc.Dataset(file_hurs, mode='r')
        f_tas    = nc.Dataset(file_tas, mode='r')

        rh       = f_hurs.variables['hurs'][:]
        tair     = f_tas.variables['tas'][:]
        VPD_tmp  = calculate_VPD_by_RH(rh, tair)

        # Read lat and lon
        try:
            latitude  = f_hurs.variables['lat'][:]
            longitude = f_hurs.variables['lon'][:]
        except:
            latitude  = f_hurs.variables['latitude'][:]
            longitude = f_hurs.variables['longitude'][:]

        # Get land sea mask
        f_mask          = nc.Dataset(file_mask, mode='r')
        landsea         = f_mask.variables['landsea'][:]

        # Read time
        time_tmp  = nc.num2date(f_hurs.variables['time'][:],f_hurs.variables['time'].units,
                    only_use_cftime_datetimes=False, calendar=f_hurs.variables['time'].calendar) # only_use_python_datetimes=True,

        f_hurs.close()
        f_tas.close()
        f_mask.close()

        # To solve the inconsistancy in time coordinate
        for i, t in enumerate(time_tmp):
            year   = t.year
            month  = t.month
            day    = t.day
            hour   = t.hour
            minute = t.minute
            second = t.second
            microsecond    = t.microsecond
            time_tmp[i]    = datetime(year, month, day, hour, minute, second, microsecond)
            VPD_tmp[i,:,:] = np.where(landsea == 0, VPD_tmp[i,:,:], np.nan)

        # select time periods
        time_cood = time_mask(time_tmp, time_s, time_e)

        # make new time cooridate
        time      = time_tmp[time_cood]

        # select region
        if region != None:
            lat_mask  = (latitude > region[0][0]) & (latitude < region[0][1])
            lon_mask  = (longitude > region[1][0]) & (longitude < region[1][1])

            lon_mask_2D, lat_mask_2D = np.meshgrid(lon_mask,lat_mask)
            # Read variable
            lat_lon_mask = lon_mask_2D & lat_mask_2D
            VPD          = VPD_tmp[time_cood]
            ntime        = len(VPD[:,0,0])
            for t in np.arange(ntime):
                VPD[t,:,:]=np.where(lat_lon_mask,VPD[t,:,:],np.nan)
        else:
            VPD       = VPD_tmp[time_cood,:,:]

        if j == 0:
            VPD_all = np.zeros((len(model_out_list)+1,len(time)))
        VPD_all[j,:]= np.nanmean(VPD, axis=(1,2))

        mask_temp = ~ np.isnan(VPD)
        VPD_mean  = np.nanmean(VPD)
        VPD_std   = np.nanstd(VPD)
        VPD_5     = np.percentile(VPD[mask_temp], 5)
        VPD_25    = np.percentile(VPD[mask_temp], 25)
        VPD_75    = np.percentile(VPD[mask_temp], 75)
        VPD_95    = np.percentile(VPD[mask_temp], 95)
        if j ==0:
            VPD_metrics = pd.DataFrame({model_name: np.array([VPD_mean, VPD_std, VPD_5, VPD_25, VPD_75, VPD_95])})
        else:
            VPD_metrics[model_name] = np.array([VPD_mean, VPD_std, VPD_5, VPD_25, VPD_75, VPD_95])

    # Calculate global mean
    VPD_mean  = np.nanmean(VPD_all)
    VPD_std   = np.nanstd(VPD_all)
    VPD_5     = np.percentile(VPD_all, 5)
    VPD_25    = np.percentile(VPD_all, 25)
    VPD_75    = np.percentile(VPD_all, 75)
    VPD_95    = np.percentile(VPD_all, 95)
    VPD_metrics['all_model'] = np.array([VPD_mean, VPD_std, VPD_5, VPD_25, VPD_75, VPD_95])
    if region_name!=None:
        VPD_metrics.to_csv(f'./txt/CMIP6/VPD_{scenario}_global_land_metrics_'+region_name+'.csv')
    else:
        VPD_metrics.to_csv(f'./txt/CMIP6/VPD_{scenario}_global_land_metrics.csv')
    return

def calculate_future_VPD_warming_level(CMIP6_out_path):

    # select the site information from each CMIP6 file
    warming_level_path = '/g/data/w97/mm3972/data/Global_warming_levels/'
    warming_levels     = ['baseline_1990_2019','1deg','2deg']

    for warming_level in warming_levels:

        print('warming_level',warming_level)

        all_data_1d = []

        file_names  = sorted(glob.glob(f'{warming_level_path}{warming_level}/vpd/*/*/*.nc'))

        # read all models for this scenario
        for file_name in file_names:
            # Get model name
            model_name  = file_name.split("/")[9]
            print(model_name)

            # read mask file
            file_mask = CMIP6_out_path + model_name + '_landsea_mask.nc'
            f_mask    = nc.Dataset(file_mask, mode='r')
            landsea   = f_mask.variables['landsea'][:]

            # read data
            f_vpd     = nc.Dataset(file_name, mode='r')
            vpd_tmp   = f_vpd.variables['vpd'][:]
            ntime     = len(vpd_tmp[:,0,0])
            # vpd       = np.zeros(ntime)

            print('np.shape(vpd_tmp)',np.shape(vpd_tmp))
            print('vpd_tmp[0,0,0]',vpd_tmp[0,0,0])

            cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r
            fig1, ax1  = plt.subplots(nrows=1, ncols=1, figsize=[5,5],sharex=False, sharey=False, squeeze=True) #
            ax1.contourf(vpd_tmp[0,:,:],cmap=cmap)
            fig1.savefig("./plots/Check_"+model_name+".png",bbox_inches='tight',dpi=300)


            for t in np.arange(ntime):
                vpd  = np.nanmean(np.where( (vpd_tmp[t,:,:]== vpd_tmp[0,0,0]), np.nan,  vpd_tmp[t,:,:]))
                # vpd  = np.where( (temp> 100.),temp, np.nan)
                # vpd   = np.nanmean( np.where( (landsea==0) , temp, np.nan) )
                all_data_1d.append(vpd)
                # print('np.any(np.where( landsea==0, vpd_tmp[t,:,:], np.nan))',np.any(np.where( landsea==0, vpd_tmp[t,:,:], np.nan)))
                # print('vpd[t]',vpd[t])

        print('np.shape(all_data_1d)',np.shape(all_data_1d))

        # Calculate metrics
        all_data_1d   = np.array(all_data_1d)
        print('np.shape(all_data_1d)',np.shape(all_data_1d))
        Mean          = np.mean(all_data_1d)
        print('Mean',Mean)
        Percentile_75 = np.percentile(all_data_1d, 75)
        Percentile_25 = np.percentile(all_data_1d, 25)
        Percentile_95 = np.percentile(all_data_1d, 95)
        Percentile_5  = np.percentile(all_data_1d, 5)

        if warming_level == 'baseline_1990_2019':
            metrics = pd.DataFrame([Mean, Percentile_75, Percentile_25, Percentile_95, Percentile_5], columns=[warming_level])
        else:
            metrics[warming_level] = [Mean,Percentile_75,Percentile_25, Percentile_95, Percentile_5]

    metrics.to_csv('./txt/CMIP6/metrics_land_VPD_at_different_warming_levels.csv')

    return

def plot_VPD_time_series(model_out_list, scenarios):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[6,4],sharex=False, sharey=False, squeeze=True) #
    # fig, ax = plt.subplots(figsize=[10, 7])
    # plt.subplots_adjust(wspace=0.09, hspace=0.02)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color']     = almost_black
    plt.rcParams['xtick.color']     = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']      = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # Set the colors for different models
    model_colors = set_model_colors()

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # ============== read data ==============
    colors       = {'historical':'black', 'ssp126':'forestgreen','ssp245':'orange','ssp585':'darkred'}

    labels       = ['historical', 'ssp126','ssp245','ssp585']
    for s, scenario in enumerate(scenarios):

        # Set color
        line_color = colors[scenario]

        for i, model_name in enumerate(model_out_list):

            VPD_values = pd.read_csv(f'./txt/CMIP6/{model_name}_VPD_{scenario}_global_land.csv')

            ntime      = len(VPD_values)
            nyear      = int(ntime/12)
            print('nyear',nyear)
            # get time series
            time_tmp   = VPD_values['time']
            year_s     = pd.to_datetime(time_tmp[0]).year
            year_e     = pd.to_datetime(time_tmp[ntime-1]).year
            time_series= np.arange(year_s,year_e+1,1)

            # to yearly
            vpd        = np.zeros(nyear)
            vpd_tmp    = VPD_values['VPD_mean']
            for t in np.arange(nyear):
                t_s    = t*12
                t_e    = (t+1)*12
                vpd[t] = np.mean(vpd_tmp[t_s:t_e])

            plot = ax.plot(time_series, vpd, lw=0.4, color=line_color, alpha=0.3)
            if i == 0:
                vpd_all_models  = np.zeros((len(model_out_list),nyear))
            vpd_all_models[i,:] = vpd

        # plot the model emsemble
        plot = ax.plot(time_series, np.mean(vpd_all_models,axis=0), lw=2.0, color=line_color, alpha=1.0, label=labels[s])

        # ===== Drawing the box whisker =====
        if 1:
            warming_levels = ['baseline_1990_2019','1deg','2deg']
            line_colors    = ['black','tomato','firebrick']
            metric_file    = './txt/CMIP6/metrics_land_VPD_at_different_warming_levels.csv'

            box_metrics    = pd.read_csv(metric_file)

            for i, warming_level in enumerate(warming_levels):
                mean, p75,  p25, maximum, minimum = box_metrics[warming_level]

                xaxis_s = 2102 + i*10
                xaxis_e = 2108 + i*10

                # Draw the box
                ax.add_patch(Polygon([[xaxis_s, p25], [xaxis_s, p75],[xaxis_e, p75], [xaxis_e, p25]],
                                              closed=True, color=line_colors[i], fill=True, alpha=0.8, linewidth=0.1))

                # Draw the mean line
                ax.plot([xaxis_s,xaxis_e], [mean,mean], color = almost_black, linewidth=0.5)

                # Draw the p25 p75
                ax.plot([xaxis_s, xaxis_e], [p25, p25], color = almost_black, linewidth=0.5)
                ax.plot([xaxis_s, xaxis_e], [p75, p75], color = almost_black, linewidth=0.5)

                ax.plot([xaxis_s, xaxis_s], [p25, p75], color = almost_black, linewidth=0.5)
                ax.plot([xaxis_e, xaxis_e], [p25, p75], color = almost_black, linewidth=0.5)

                # Draw the max and min
                ax.plot([xaxis_s+4, xaxis_e-4], [minimum, minimum], color = almost_black, linewidth=0.5)
                ax.plot([xaxis_s+4, xaxis_e-4], [maximum, maximum], color = almost_black, linewidth=0.5)
                ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p75, maximum], color = almost_black, linewidth=0.5)
                ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p25, minimum], color = almost_black, linewidth=0.5)

    ax.legend(fontsize=10, frameon=False)
    ax.set_xticks([1900,1910,1920,1930,1940,1950,1960,1970,1980, 1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100, 2105, 2115, 2125])
    ax.set_xticklabels(['1900','1910','1920','1930','1940','1950','1960','1970','1980', '1990','2000',
                        '2010','2020','2030','2040','2050','2060','2070','2080','2090','2100',
                        'baseline','1$\mathregular{^{o}}$C','2$\mathregular{^{o}}$C'],fontsize=12,rotation=90)
    ax.set_xlim(1950,2130)

    ax.set_ylabel("VPD (kPa)", fontsize=12)
    # ax[0,0].tick_params(axis='y', labelsize=12)
    # ax[0,0].set_ylim(-50,80)

    fig.savefig("./plots/Fig_CMIP_VPD_time_series.png",bbox_inches='tight',dpi=300) # '_30percent'


if __name__ == "__main__":

    # Read files
    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    CMIP6_data_path   = "/g/data/w97/amu561/CMIP6_for_Mengyuan/Processed_CMIP6_data/"
    CMIP6_out_path    = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/CMIP6/"
    scenarios         = ['historical','ssp126','ssp245','ssp585']
    region            = [[-44,-10], [112,154]]
    region_name       = 'AU'
    # get file names
    file_names           = {}
    file_names_scenario  = {}


    # # read CMIP6 data
    # for scenario in scenarios:
    #
    #     if scenario == 'historical':
    #         time_s             = datetime(1900,1,1,0,0,0)
    #         time_e             = datetime(2015,1,1,0,0,0)
    #     else:
    #         time_s             = datetime(2015,1,1,0,0,0)
    #         time_e             = datetime(2100,1,1,0,0,0)

    #     file_names_scenario = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hurs/*/*/*.nc'))
    #
    #     model_out_list      = []
    #     for file_name in file_names_scenario:
    #         model_out_name = file_name.split("/")[9]
    #         if model_out_name in ['NorESM2-LM','CESM2-WACCM']:
    #             print('Ignore ',model_out_name,'since different time cooridate in Tair and RH')
    #         else:
    #             model_out_list.append(model_out_name)
    #
    #     # make_landsea_mask_file(CMIP6_data_path, CMIP6_out_path, model_out_list)
    #     # calculate_model_mean_land_VPD(CMIP6_data_path, CMIP6_out_path, scenario, model_out_list)
    #
    # calculate_future_VPD_warming_level(CMIP6_out_path)
    # plot_VPD_time_series(model_out_list, scenarios)
    #

    # read CMIP6 data
    for scenario in scenarios:

        if scenario == 'historical':
            time_s             = datetime(1985,1,1,0,0,0)
            time_e             = datetime(2015,1,1,0,0,0)
        else:
            time_s             = datetime(2070,1,1,0,0,0)
            time_e             = datetime(2100,1,1,0,0,0)

        file_names_scenario = sorted(glob.glob(f'{CMIP6_data_path}{scenario}/hurs/*/*/*.nc'))

        model_out_list      = []
        for file_name in file_names_scenario:
            model_out_name = file_name.split("/")[9]
            if model_out_name in ['NorESM2-LM','CESM2-WACCM']:
                print('Ignore ',model_out_name,'since different time cooridate in Tair and RH')
            else:
                model_out_list.append(model_out_name)
        print('model_out_list',model_out_list)
        calculate_model_VPD_metrics(CMIP6_data_path, CMIP6_out_path, scenario, model_out_list, time_s, time_e,
                                    region=region, region_name=region_name)
