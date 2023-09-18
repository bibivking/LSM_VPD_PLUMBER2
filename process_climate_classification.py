import os
import sys
import glob
from osgeo import gdal
import netCDF4 as nc
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from cartopy.feature import NaturalEarthFeature, OCEAN
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from PLUMBER2_VPD_common_utils import *

def read_clim_class(clim_class_path, clim_class_out):

    # for opening the raster read-only and saving it on f variable.
    f  = gdal.Open(clim_class_path, gdal.GA_ReadOnly)
    print('f',f)

    # Copy the transformation to a variable, it will be useful later.
    gt = f.GetGeoTransform()
    print('gt',gt)

    # Get the projection
    projection = f.GetProjection()

    # Read the bands of your raster using GetRasterBand
    clim_class = f.ReadAsArray()
    # clim_class = f.GetRasterBand(1)
    print( 'clim_class', clim_class)

    # Read the size of your array
    size1, size2 = clim_class.shape

    # Calculate lat and lon
    print('calculate lat and lon')
    lat = np.zeros((size1))
    lon = np.zeros((size2))
    
    for y in np.arange(size1):
        lat[y] = gt[3] + y * gt[5]
    for x in np.arange(size2):
        lon[x] = gt[0] + x * gt[1] 


    # Convert the clim_class to a float32 array 
    # var = clim_class.astype(np.float32) 
    # var = np.where(var < 0, -9999. ,var)

    # =================== Make nc file ===================
    f                   = nc.Dataset(clim_class_out, 'w', format='NETCDF4')

    ### Create nc file ###
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = 'Transfer '+clim_class_path+' to netcdf file, created by MU Mengyuan'

    # set dimensions
    print('set dimensions')
    f.createDimension('latitude', size1)
    f.createDimension('longitude', size2)
    f.createDimension('class', 30)
    f.createDimension('RGB', 3)
    f.Conventions       = "CF-1.0"

    # create variables
    print('create variables')

    print('create latitude')
    latitude            = f.createVariable('latitude', 'f4', ('latitude'))
    latitude.long_name  = "latitude"
    latitude.units      = "degree_north"
    latitude._CoordinateAxisType = "Lat"
    latitude[:]         = lat

    print('create longitude')
    longitude           = f.createVariable('longitude', 'f4', ('longitude'))
    longitude.long_name = "longitude"
    longitude.units     = "degree_east"
    longitude._CoordinateAxisType = "Lon"
    longitude[:]        = lon

    print('create climate_class')
    var_out             = f.createVariable('climate_class', 'f4', ('latitude', 'longitude'),fill_value=0.)
    var_out.long_name   = "Köppen-Geiger climate classification"
    var_out[:]          = clim_class
    
    # Form the class names array
    class_name_tmp = np.array([
                            'Af', 'Am', 'Aw', 'BWh', 
                            'BWk', 'BSh','BSk', 'Csa',
                            'Csb','Csc', 'Cwa', 'Cwb',
                            'Cwc', 'Cfa', 'Cfb', 'Cfc',
                            'Dsa', 'Dsb', 'Dsc', 'Dsd',
                            'Dwa', 'Dwb', 'Dwc', 'Dwd',
                            'Dfa', 'Dfb', 'Dfc', 'Dfd',
                            'ET', 'EF'
                            ], dtype="S3")

    print('create class_name')
    class_name           = f.createVariable('class_name',  "S3", ('class'))
    class_name.long_name = "name of the climate classification"
    class_name[:]        = class_name_tmp

    print('create class_color')
    class_color           = f.createVariable('class_color', 'f4', ('class','RGB'))
    class_color.long_name = "RGB of every climate classification"
    class_color[:]        = [  [0, 0, 255],     [0, 120, 255], [70, 170, 250], [255, 0, 0],
                               [255, 150, 150], [245, 165, 0], [255, 220, 100],[255, 255, 0],
                               [200, 200, 0],   [150, 150, 0], [150, 255, 150],[100, 200, 100],
                               [50, 150, 50],   [200, 255, 80],[100, 255, 80], [50, 200, 0],
                               [255, 0, 255],   [200, 0, 200], [150, 50, 150], [150, 100, 150],
                               [170, 175, 255], [90, 120, 220],[75, 80, 180],  [50, 0, 135],
                               [0, 255, 255],   [55, 200, 255],[0, 125, 125],  [0, 70, 95],
                               [178, 178, 178], [102, 102, 102] ]

    print('create class_long_name')   
    # Form the class long names array
    class_long_name_tmp = np.array([ 'Tropical, rainforest',  'Tropical, monsoon', 'Tropical, savannah', 'Arid, desert, hot',
                                  'Arid, desert, cold', 'Arid, steppe, hot', 'Arid, steppe, cold', 'Temperate, dry summer, hot summer',
                                  'Temperate, dry summer, warm summer', 'Temperate, dry summer, cold summer', 'Temperate, dry winter, hot summer','Temperate, dry winter, warm summer',
                                  'Temperate, dry winter, cold summer', 'Temperate, no dry season, hot summer', 'Temperate, no dry season, warm summer', 'Temperate, no dry season, cold summer',
                                  'Cold, dry summer, hot summer', 'Cold, dry summer, warm summer', 'Cold, dry summer, cold summer','Cold, dry summer, very cold winter',
                                  'Cold, dry winter, hot summer', 'Cold, dry winter, warm summer', 'Cold, dry winter, cold summer', 'Cold, dry winter, very cold winter',
                                  'Cold, no dry season, hot summer', 'Cold, no dry season, warm summer', 'Cold, no dry season, cold summer', 'Cold, no dry season, very cold winter',
                                  'Polar, tundra', 'Polar, frost' ], dtype="S40")            

    class_long_name           = f.createVariable('class_long_name', 'S40', ('class'))
    class_long_name.long_name = "long name of the climate classification"
    class_long_name[:]        = class_long_name_tmp

    f.close()
    return

def plot_clim_class(site_names, PLUMBER2_met_path, clim_class_path_low_res):

    # for opening the raster read-only and saving it on f variable.
    f  = gdal.Open(clim_class_path_low_res, gdal.GA_ReadOnly)
    print('f',f)

    # Copy the transformation to a variable, it will be useful later.
    gt = f.GetGeoTransform()
    print('gt',gt)

    # Get the projection
    projection = f.GetProjection()

    # Read the bands of your raster using GetRasterBand
    clim_class = f.ReadAsArray()
    # clim_class = f.GetRasterBand(1)
    print( 'clim_class', clim_class)

    # Read the size of your array
    size1, size2 = clim_class.shape
    print('size1,size2=',size1,size2)
    print(clim_class)

    # Calculate lat and lon
    print('calculate lat and lon')
    lat = np.zeros((size1))
    lon = np.zeros((size2))
    
    for y in np.arange(size1):
        lat[:] = gt[3] + y * gt[5]
    for x in np.arange(size2):
        lon[:] = gt[0] + x * gt[1] 

    # ____ plotting ____
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5,5],sharex=True, sharey=True, squeeze=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0., hspace=-0.05) # left=0.15,right=0.95,top=0.85,bottom=0.05,

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 10
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

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # ======================= Set colormap =======================
    cmap  = plt.cm.gist_ncar #rainbow

    colors = [  [0, 0, 255, 0.4],     [0, 120, 255, 0.8], [70, 170, 250, 0.8], [255, 0, 0, 0.6],
                [255, 150, 150, 0.7], [245, 165, 0, 0.7], [255, 220, 100, 0.7],[255, 255, 0, 0.7],
                [200, 200, 0, 0.6],   [150, 150, 0, 0.6], [150, 255, 150, 0.6],[100, 200, 100, 0.6],
                [50, 150, 50, 0.9],   [200, 255, 80, 0.9],[100, 255, 80, 0.9], [50, 200, 0, 0.9],
                [255, 0, 255, 0.7],   [200, 0, 200, 0.7], [150, 50, 150, 0.7], [150, 100, 150, 0.7],
                [170, 175, 255, 0.5], [90, 120, 220, 0.5],[75, 80, 180, 0.5],  [50, 0, 135, 0.5],
                [0, 255, 255, 0.6],   [55, 200, 255, 0.6],[0, 125, 125, 0.6],  [0, 70, 95, 0.6],
                [178, 178, 178, 0.4], [102, 102, 102, 0.4] ]
    custom_cmap = ListedColormap(colors)

    ax.coastlines(resolution="50m",linewidth=0.5)
    # ax.add_feature(states, linewidth=.5, edgecolor="black")
    clevs = np.arange(-0.5,30.5,1)
    extent=(-180, 180, -90, 90)
    clim_class = np.where(clim_class==0,np.nan,clim_class)
    plot1 = ax.imshow(clim_class[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=29.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False, 
    ax.add_feature(OCEAN,edgecolor='none', facecolor="white")

    # plot1 = ax.contourf(lon, lat, clim_class, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='neither')

    cbar  = plt.colorbar(plot1, ax=ax, ticklocation="right", pad=0.01, orientation="horizontal",
                        aspect=40, shrink=1.) # cax=cax,
   
    cbar.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    cbar.set_ticklabels(['Af', 'Am', 'Aw', 'BWh', 
                            'BWk', 'BSh','BSk', 'Csa',
                            'Csb','Csc', 'Cwa', 'Cwb',
                            'Cwc', 'Cfa', 'Cfb', 'Cfc',
                            'Dsa', 'Dsb', 'Dsc', 'Dsd',
                            'Dwa', 'Dwb', 'Dwc', 'Dwd',
                            'Dfa', 'Dfb', 'Dfc', 'Dfd',
                            'ET', 'EF'])

    cbar.ax.tick_params(labelsize=6, labelrotation=45)

    # Adding lat and lon
    lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_met_path)
    for site_name in site_names:
        ax.plot(lon_dict[site_name], lat_dict[site_name], color='red', marker='s', markersize=0.3, transform=ccrs.PlateCarree())

    plt.savefig('./plots/climate_classification.png',dpi=300)
    
    return 

if __name__ == '__main__':

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"

    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"


    # Köppen-Geiger_climate_classification
    clim_class_path    = "/g/data/w97/mm3972/data/Köppen-Geiger_climate_classification/Beck_KG_V1/Beck_KG_V1_present_0p0083.tif"
    clim_class_path_low_res = "/g/data/w97/mm3972/data/Köppen-Geiger_climate_classification/Beck_KG_V1/Beck_KG_V1_present_0p5.tif"
    clim_class_out     = "/g/data/w97/mm3972/data/Köppen-Geiger_climate_classification/Beck_KG_V1/Beck_KG_V1_present_0p0083.nc"

    # The name of models
    model_names   = [ "CABLE","3km27","CABLE-POP-CN","CHTESSEL_Ref_exp1",
                      "GFDL","MATSIRO","NASAEnt","ORCHIDEE_tag2.1",
                      "QUINCY","ACASA","CHTESSEL_ERA5_3","CLM5a",
                      "JULES_GL9","LSTM_raw","MuSICA","NoahMPv401","ORCHIDEE_tag3_2",
                      "RF","STEMMUS-SCOPE","LPJ-GUESS","SDGVM", "BEPS",]

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    # print(site_names)
    # lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_met_path)
    # IGBP_dict = read_IGBP_veg_type(site_names, PLUMBER2_met_path)

    read_clim_class(clim_class_path, clim_class_out)
    plot_clim_class(site_names, PLUMBER2_met_path, clim_class_path_low_res)