'''
Including
    def time_step_2_daily
    def update_EF_in_Qle
    def time_step_2_daily_LAI
'''

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
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *
from copy import deepcopy

def read_RH(site_name):

    PLUMBER2_met_path = '/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/'
    file_path = glob.glob(PLUMBER2_met_path + "*" + site_name + "*.nc")

    if not file_path:
        raise FileNotFoundError(f"No file found for site {site_name}")

    f  = nc.Dataset(file_path[0])
    RH = f.variables['RH'][:,0,0]
    RH = np.where(RH < 0, np.nan, RH)

    return RH

def calc_72h_Tmin(site_tair):
    Tair = pd.DataFrame(site_tair, columns=['Tair'])
    Tmin = Tair['Tair'].rolling(window=72*2, min_periods=1).min().values  # 72 hours at half-hourly data
    print(Tair.values[:72*4], Tmin[:72*4])

    return Tmin


def calc_72h_Tmean(site_tair):
    Tair = pd.DataFrame(site_tair, columns=['Tair'])
    Tmean = Tair['Tair'].rolling(window=72*2, min_periods=1).mean().values  # 72 hours at half-hourly data
    print(Tair.values[:72*4], Tmean[:72*4])

    return Tmean

def calc_time_steps_after_precip(site_precip, valid_daily_precip=0.1):

    s2h   = 60*60
    ntime = len(site_precip)

    # calculate hourly precipitation
    site_precip = site_precip * s2h

    # check whether hourly precipitation passes the threshold
    valid_prec = np.where(site_precip > valid_daily_precip, 1, 0)

    # calculate hours without precipitation
    accul_ts = 0
    ts_after_precip = np.zeros(ntime)

    for t in np.arange(ntime):
        accul_ts = np.where(valid_prec[t] == 1, 0, accul_ts + 1)
        ts_after_precip[t] = accul_ts

    return ts_after_precip

def data_selection(site_names, clarify_site={'opt': False, 'remove_site': None}, middle_day=False, criteria=1):

    # ========== Adding selection data ===========
    # Read in data
    var_output = pd.read_csv(f'./txt/process1_output/Qle_all_sites.csv',
                             usecols=['site_name', 'obs', 'VPD', 'obs_Tair', 'obs_SWdown', 'obs_Precip'])

    Qh_output  = pd.read_csv(f'./txt/process1_output/Qh_all_sites.csv',
                             usecols=["obs"])
    var_output['obs_Qh']   = Qh_output.values

    Wind_output = pd.read_csv(f'./txt/process1_output/Gs_ref_all_sites_filtered.csv',
                              usecols=["obs_Wind"])
    var_output['obs_Wind'] = Wind_output.values

    # data quality control
    qc_input = pd.read_csv(f'./txt/process1_output/Qle_Qh_quality_control_all_sites.csv', na_values=[''])
    var_output['Qle_Qh_qc'] = qc_input['Qle_Qh_qc']

    var_output['obs_RH']          = np.nan
    var_output['pre_72h_T']       = np.nan
    var_output['ts_after_precip'] = np.nan

    for site_name in site_names:
        site_mask = (var_output['site_name'] == site_name)

        # read relative humidity
        var_output.loc[site_mask, 'obs_RH']       = read_RH(site_name)

        # read minima Tair
        if criteria == 1:
            var_output.loc[site_mask, 'pre_72h_T'] = calc_72h_Tmin(var_output.loc[site_mask, 'obs_Tair'].values)
        elif criteria == 2:
            var_output.loc[site_mask, 'pre_72h_T'] = calc_72h_Tmean(var_output.loc[site_mask, 'obs_Tair'].values)

        # calculate time steps after rainfall
        valid_daily_precip = 0.1
        var_output.loc[site_mask, 'ts_after_precip'] = calc_time_steps_after_precip(var_output.loc[site_mask, 'obs_Precip'].values,
                                                                                    valid_daily_precip=valid_daily_precip)

    # =========== Selecting data =============

    # To exclude the sites that have rainfall input problems and sites some models doesn't have output
    if clarify_site['opt']:
        # print('clarifying sites')
        length    = len(var_output)
        site_mask = np.full(length, True)

        for site_remove in clarify_site['remove_site']:
            site_mask = np.where(var_output['site_name'] == site_remove, False, site_mask)

    # quality control
    qc_mask = ~np.isnan(var_output['Qle_Qh_qc'])
    print('qc_mask is true',np.sum(qc_mask))


    # previous 24 hours no rainfall
    if criteria == 1:
        rain_mask = (var_output['ts_after_precip'] >= 48)
    elif criteria == 2:
        rain_mask = (var_output['ts_after_precip'] >= 12)

    print('rain_mask is true',np.sum(rain_mask))

    # To avoid stable boundary layer conditions

    if criteria == 1:
        sbl_mask = (var_output['obs_SWdown'] >= 50.) & (var_output['obs_Wind'] >= 1.) # (var_output['obs_Qh'] >= 5.) &
    elif criteria == 2:
        sbl_mask = (var_output['obs_SWdown'] >= 10.) & (var_output['obs_Wind'] >= 1.) # (var_output['obs_Qh'] >= 5.) &
    print('sbl_mask is true',np.sum(sbl_mask))

    # To avoid frozen soil and snow
    frsn_mask = (var_output['pre_72h_T'] > 273.15)
    print('frsn_mask is true',np.sum(frsn_mask))

    # To avoid dew
    dew_mask = (var_output['obs_RH'] <= 99.) & (var_output['obs'] >= 0.) & (var_output['obs_Qh'] >= 0.)
    print('dew_mask is true',np.sum(dew_mask))

    var_output['qc_mask']   = qc_mask
    var_output['site_mask'] = site_mask
    var_output['rain_mask'] = rain_mask
    var_output['sbl_mask']  = sbl_mask
    var_output['frsn_mask'] = frsn_mask
    var_output['dew_mask']  = dew_mask

    if middle_day:
        midday_mask = (var_output['obs_SWdown'] >= 600)
        var_output['select_data'] = qc_mask & site_mask & rain_mask & sbl_mask & frsn_mask & dew_mask & midday_mask
    else:
        var_output['select_data'] = qc_mask & site_mask & rain_mask & sbl_mask & frsn_mask & dew_mask

    print('select_data is true',np.sum(var_output['select_data'].values))

    # Save the processed data to a new CSV file
    var_output.to_csv(f'./txt/process2_output/data_selection_all_sites.csv', index=False)

    return

if __name__ == "__main__":

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    clarify_site      = {'opt': True,
                         'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6', # rainfall problems
                                         'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1',
                                         'AU-Wrr','CN-Din','US-WCr','ZM-Mon' # models miss the simulations of them
                                         ]}
    middle_day=False
    criteria  = 2
    data_selection(site_names, clarify_site=clarify_site, middle_day=middle_day, criteria=criteria)
