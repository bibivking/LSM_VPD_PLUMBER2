__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from PLUMBER2_VPD_common_utils import load_default_list
import multiprocessing as mp
import logging

def latent_heat_vapourisation(tair):
    """
    Latent heat of vapourisation is approximated by a linear func of air
    temp (J kg-1)

    Reference:
    ----------
    * Stull, B., 1988: An Introduction to Boundary Layer Meteorology
      Boundary Conditions, pg 279.
    """
    return (2.501 - 0.00237 * tair) * 1E06

def site_annual_PET(site_name, site_tair, P_qc_tmp):

    # Read PET data
    PM_output_path = '/g/data/w97/mm3972/data/PLUMBER2/PenmanMonteith/'
    file_path      = glob.glob(PM_output_path + "*" + site_name + "*.nc")

    if not file_path:
        raise FileNotFoundError(f"No file found for site {site_name}")
    print(file_path[0])
    f    = nc.Dataset(file_path[0])
    time = nc.num2date(f.variables['time'][:], f.variables['time'].units,
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True)

    years   = [t.year for t in time]
    PET_tmp = f.variables['Qle'][:,0,0]

    # Use the same time steps as rainfall
    PET_tmp = np.where(P_qc_tmp <= 2., PET_tmp, 0.)

    # Calculate ET, W m-2 to kg m-2 per 0.5 h (time step)
    lhv     = latent_heat_vapourisation(site_tair)
    pet_tmp = PET_tmp / lhv * 60 * 30

    var_output         = pd.DataFrame(pet_tmp, columns=['PET'])
    var_output['year'] = years

    PET_annual = var_output.groupby('year').sum().reset_index()
    print("PET_annual['PET'] 1st",PET_annual['PET'])

    PET_annual['PET'] = np.where(PET_annual['PET']>10., PET_annual['PET'], np.nan)
    print("PET_annual['PET'] 2nd",PET_annual['PET'])

    return PET_annual['PET'].values

def site_annual_P(site_name):

    # Read P data
    PLUMBER2_met_path = '/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/'
    file_path         = glob.glob(PLUMBER2_met_path + "*" + site_name + "*.nc")

    if not file_path:
        raise FileNotFoundError(f"No file found for site {site_name}")
    print(file_path[0])
    f    = nc.Dataset(file_path[0])
    time = nc.num2date(f.variables['time'][:], f.variables['time'].units,
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True)

    years    = [t.year for t in time]
    P_tmp    = f.variables['Precip'][:,0,0] *60. * 30.
    print(np.sum(P_tmp))
    try:
        P_qc_tmp = f.variables['Precip_qc'][:,0,0]
        P_tmp    = np.where(P_qc_tmp <= 2., P_tmp, 0.)
    except:
        P_qc_tmp = np.zeros(len(P_tmp))
        print(site_name, 'does not have Precip_qc')

    precip_output         = pd.DataFrame(P_tmp, columns=['P'])
    precip_output['year'] = years

    print('P_tmp[P_tmp>0]',P_tmp[P_tmp>0])

    P_annual              = precip_output.groupby('year').sum().reset_index()
    print("P_annual['P'] 1st",P_annual['P'])

    P_annual['P'] = np.where(P_annual['P'] > 10., P_annual['P'], np.nan)
    print("P_annual['P'] 2nd",P_annual['P'])

    return P_annual['P'].values, P_qc_tmp

def calc_aridity_index(site_names):

    """
    Read Qle_qc and Qh_qc for flux files and write to csv.

    For site AU-DaP, no precipiation data cannot pass quality control,
    ignoring quality control PET/P = 1.7490075805107508
    """

    var_output = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv', usecols=['time', 'site_name', 'obs_Precip', 'obs_Tair'])
    var_output['aridity_index'] = np.nan

    for site_name in site_names:
        site_mask = (var_output['site_name'] == site_name)
        print('data_points',np.sum(site_mask))
        # calculate mean annual P
        P_annual, P_qc_tmp = site_annual_P(site_name)#, site_precip, years.values)

        # calculate mean annual PET
        site_tair          = var_output.loc[site_mask, 'obs_Tair'].values
        PET_annual         = site_annual_PET(site_name, site_tair, P_qc_tmp)

        var_output.loc[site_mask, 'aridity_index'] = np.nanmean(PET_annual[:-1] / P_annual)
        print(site_name, np.nanmean(PET_annual[:-1] / P_annual))
        gc.collect()

    var_output.to_csv(f'./txt/process1_output/Aridity_index_all_sites.csv')

    return

if __name__ == "__main__":

    # The site names
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    calc_aridity_index(site_names)
