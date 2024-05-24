#!/usr/bin/env python
"""
Estimate the ecosystem conductance (Gs) from inverting the penman-monteith
against eddy covariance flux data. Finally, make a 1:1 plot of VPD_leaf vs
VPD_atmospheric

That's all folks.
"""
__author__   = "Martin De Kauwe"
__modifier__ = "Mengyuan Mu"
__version__  = "1.0 (17.01.2024)"
__email__    = "mu.mengyuan815@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import datetime as dt
from scipy.stats import pearsonr
# from rmse import rmse
import netCDF4 as nc
import re
from datetime import datetime
from PLUMBER2_VPD_common_utils import *
import pdb
import Martin_constants as c
from Martin_penman_monteith import PenmanMonteith

def main(PLUMBER2_path, PLUMBER2_met_path, PLUMBER2_flux_path, site_names, model_names, calc_ref=False):

    for site_name in site_names:

        print('site_name',site_name)
        Qle_dict = check_variable_exists(PLUMBER2_path, 'Qle', site_name, model_names)
        Qh_dict  = check_variable_exists(PLUMBER2_path, 'Qh', site_name, model_names)
        Qg_dict  = check_variable_exists(PLUMBER2_path, 'Qg', site_name, model_names)
        GPP_dict = check_variable_exists(PLUMBER2_path, 'GPP', site_name, model_names)
        Rnet_dict= check_variable_exists(PLUMBER2_path, 'Rnet', site_name, model_names)

        # calc_gs_one_site(PLUMBER2_met_path, PLUMBER2_flux_path, site_name, model_names,
        #                  Qle_dict, Qh_dict, Qg_dict, Rnet_dict, GPP_dict, calc_ref)

        calc_Rnet_caused_LH_ratio(PLUMBER2_met_path, PLUMBER2_flux_path, site_name, model_names,
                              Qle_dict, Qh_dict, Qg_dict, Rnet_dict, GPP_dict)

        # pdb.set_trace()
    return

def calc_gs_one_site(PLUMBER2_met_path, PLUMBER2_flux_path, site_name, model_names,
                    Qle_dict, Qh_dict, Qg_dict, Rnet_dict, GPP_dict, calc_ref=False):

    # ================== Reading data ==================
    # Read obs met data

    message = 'Site '+site_name+'\n'
    canht, VPD_obs, Tair_obs, Precip_obs, Wind_obs, Psurf_obs = \
        read_met_variables(PLUMBER2_met_path, site_name)

    # Read obs flux data
    Qle_obs, Qh_obs, GPP_obs, Rnet_obs, Qg_obs, Ustar_obs, message = \
        read_obs_flux_variables(PLUMBER2_flux_path, site_name, message)

    for model_name in model_names:
        message = 'Model '+model_name+'\n'

        # Give values to var_input
        var_input           = pd.DataFrame(VPD_obs/10., columns=['VPD']) # hPa=>kPa
        var_input['Tair']   = Tair_obs # K
        var_input['Precip'] = Precip_obs # kg/m2/s
        var_input['Wind']   = Wind_obs # m/s
        var_input['Psurf']  = Psurf_obs # Pa
        var_input['Qle']    = Qle_obs # W/m2
        var_input['Qh']     = Qh_obs # W/m2
        var_input['Qg']     = Qg_obs # W/m2
        var_input['GPP']    = GPP_obs # umol/m2/s
        var_input['Rnet']   = Rnet_obs # W/m2
        var_input['Ustar']  = Ustar_obs # m/s

        # Read model flux data
        if model_name != 'obs':

            file_path = glob.glob(PLUMBER2_path+model_name +"/*"+site_name+"*.nc")

            if not file_path:
                # if the model doesn't simulate this site
                return
            else:
                # Get model fluxes
                Qle_model, Qh_model, GPP_model, Rnet_model, Qg_model = \
                        read_model_flux_variables(file_path, Qle_dict[model_name],
                                                Qh_dict[model_name], Qg_dict[model_name],
                                                Rnet_dict[model_name], GPP_dict[model_name])

                # Check the time interval
                ntime_obs   = len(Qle_obs)
                ntime_model = len(Qle_model)

                # Give values to var_input
                if ntime_model == ntime_obs:
                    # if half-hourly as observation
                    if np.any(~np.isnan(Qle_model)):
                        var_input['Qle'] = Qle_model
                    else:
                        message = message+'         '+model_name+' Qle not exist.\n'

                    if np.any(~np.isnan(Qh_model)):
                        var_input['Qh']  = Qh_model
                    else:
                        message = message+'         '+model_name+' Qh not exist.\n'

                    if np.any(~np.isnan(Rnet_model)):
                        var_input['Rnet']= Rnet_model
                        if np.any(~np.isnan(Qg_model)):
                            var_input['Qg']  = Qg_model
                        else:
                            var_input['Qg']  = np.nan
                            message = message+'         '+model_name+' Rnet exists, but Qg not. Set Qg=nan\n'
                    else:
                        message = message+'         '+model_name+' Rnet not exist, use obs Rnet and Qh.\n'

                    if np.any(~np.isnan(GPP_model)):
                        var_input['GPP'] = GPP_model
                    else:
                        message = message+'         '+model_name+' GPP not exist.\n'

                elif ntime_model == int(ntime_obs/2):
                    # if it is hourly
                    print('model ', model_name, ' is hourly, model_ntime is', ntime_model, ' ntime_obs is', ntime_obs)

                    # put the value of hourly data to the first half hour
                    if np.any(~np.isnan(Qle_model)):
                        var_input['Qle'][::2]  = Qle_model
                        var_input['Qle'][1::2] = Qle_model
                    else:
                        message = message+'         '+model_name+' Qle not exist.\n'

                    if np.any(~np.isnan(Qh_model)):
                        var_input['Qh'][::2]  = Qh_model
                        var_input['Qh'][1::2] = Qh_model
                    else:
                        message = message+'         '+model_name+' Qh not exist.\n'

                    if np.any(~np.isnan(Rnet_model)):
                        var_input['Rnet'][::2]  = Rnet_model
                        var_input['Rnet'][1::2] = Rnet_model
                        if np.any(~np.isnan(Qg_model)):
                            var_input['Qg'][::2]  = Qg_model
                            var_input['Qg'][1::2] = Qg_model
                        else:
                            var_input['Qg']  = np.nan
                            message = message+'         '+model_name+' Rnet exists, but Qg not. Set Qg=nan\n'
                    else:
                        message = message+'         '+model_name+' Rnet not exist, use obs Rnet and Qh.\n'

                    if np.any(~np.isnan(GPP_model)):
                        var_input['GPP'][::2]  = GPP_model
                        var_input['GPP'][1::2] = GPP_model
                    else:
                        message = message+'         '+model_name+' GPP not exist.\n'

                else:
                    # if it is hourly
                    print('Error occur! ntime_model is ', ntime_model, 'ntime_obs is', ntime_obs)

        # ================== Prepare data ==================

        # Convert units ...
        var_input['Tair'] -= 273.15

        # screen for dew
        var_input['Qle'] = np.where( var_input['Qle'] > 0.0, var_input['Qle'], np.nan)

        # Calculate ET, W m-2 to kg m-2 s-1
        lhv              = latent_heat_vapourisation(var_input['Tair'])
        var_input['ET']  = var_input['Qle'] / lhv

        # kg m-2 s-1 to mol m-2 s-1
        conv            = c.KG_TO_G * c.G_WATER_TO_MOL_WATER
        var_input['ET']*= conv

        # screen highly saturated conditions
        # var_input['VPD']= np.where(var_input['VPD']> 0.05, var_input['VPD'], np.nan)
        # MMY: VPD units is kPa
        #      When VPD == 0, the calcuated Gs can be very weird values, I don't screen
        #      0 values here but in making plots VPD=0 need to be excluded, e.g. set the 
        #      VPD bins start from 0.02 rather than 0.

        if calc_ref:
            # set VPD for reference Gs as 1000 Pa
            var_input['VPD']= 1000.
        else:
            # kPa to Pa
            var_input['VPD']= var_input['VPD']*1000.

        # (var_input, no_G) = filter_dataframe(var_input, hour) # MMY: hour tells it is hourly or half-hour data
        # if no_G:
        #     G = None
        # print(var_input)

        # To avoid crash in Gs calculation, set the row with any missing value in a column as np.nan
        subset            = ['VPD', 'Rnet', 'Wind', 'Tair', 'Psurf', 'ET']
        var_input[subset] = var_input[subset].where(~var_input[subset].isna().any(axis=1), other=np.nan)

        # ==================
        """
        PM = PenmanMonteith(use_ustar=False)
        # Height from Wilkinson, M., Eaton, E. L., Broadmeadow, M. S. J., and
        # Morison, J. I. L.: Inter-annual variation of carbon uptake by a
        # plantation oak woodland in south-eastern England, Biogeosciences, 9,
        # 5373–5389, https://doi.org/10.5194/bg-9-5373-2012, 2012.
        (var_input['Gs'],
        var_input['VPDl'])  = PM.invert_penman(var_input['VPD'].values, var_input['Wind'].values,
                                        var_input['Rnet'].values, var_input['Tair'].values,
                                        var_input['Psurf'].values,
                                        var_input['ET'].values, canht=28., G=G)
        """
        if np.all(~np.isnan(var_input['Qg'])):
            G = var_input['Qg']
            message = '      Using Qg'
        else:
            G = None

        if np.all(~np.isnan(var_input['Ustar'])):
            PM = PenmanMonteith(use_ustar=True)
            (var_input['Gs'],
             var_input['VPDl'])  = PM.invert_penman(var_input['VPD'].values, var_input['Wind'].values,
                                             var_input['Rnet'].values, var_input['Tair'].values,
                                             var_input['Psurf'].values,
                                             var_input['ET'].values,
                                             ustar=var_input["Ustar"], G=G)
        else:
            PM = PenmanMonteith(use_ustar=False)
            (var_input['Gs'],
            var_input['VPDl'])  = PM.invert_penman(var_input['VPD'].values, var_input['Wind'].values,
                                        var_input['Rnet'].values, var_input['Tair'].values,
                                        var_input['Psurf'].values,
                                        var_input['ET'].values, canht=canht, G=G)

        # screen for bad inverted data
        # MMY, I will filter these data in process_step1 so comment out them here
        # mask_Gs               = (var_input['Gs'] <= 0.0) | (var_input['Gs'] > 4.5) | (np.isnan(var_input['Gs']))
        # var_input[mask_Gs]    = np.nan
        # mask_VPDl             = (var_input['VPDl'] <= 0.05 * 1000.) | (var_input['VPDl'] > 7.* 1000) | (np.isnan(var_input['VPDl']))
        # var_input[mask_VPDl]  = np.nan

        # VPDa = var_input['VPD'] * c.PA_TO_KPA
        # VPDl = var_input['VPDl'] * c.PA_TO_KPA

        # print(var_input)
        if calc_ref:
            with open(f'./txt/process1_output/Gs/Message_gs_ref_{site_name}_{model_name}.txt', 'w') as f:
                f.write(message)
            var_input.to_csv(f'./txt/process1_output/Gs/Gs_ref_{site_name}_{model_name}.csv') # , mode='a', index=False
        else:
            with open(f'./txt/process1_output/Gs/Message_gs_{site_name}_{model_name}.txt', 'w') as f:
                f.write(message)
            var_input.to_csv(f'./txt/process1_output/Gs/Gs_{site_name}_{model_name}.csv') # , mode='a', index=False

        var_input = None
        message   = None
    return

def calc_Rnet_caused_LH_ratio(PLUMBER2_met_path, PLUMBER2_flux_path, site_name, model_names,
                              Qle_dict, Qh_dict, Qg_dict, Rnet_dict, GPP_dict, calc_ref=False):

    # ================== Reading data ==================
    # Read obs met data

    message = 'Site '+site_name+'\n'
    canht, VPD_obs, Tair_obs, Precip_obs, Wind_obs, Psurf_obs = \
        read_met_variables(PLUMBER2_met_path, site_name)

    # Read obs flux data
    Qle_obs, Qh_obs, GPP_obs, Rnet_obs, Qg_obs, Ustar_obs, message = \
        read_obs_flux_variables(PLUMBER2_flux_path, site_name, message)

    for model_name in model_names:
        message = 'Model '+model_name+'\n'

        # Give values to var_input
        var_input           = pd.DataFrame(VPD_obs/10., columns=['VPD']) # hPa=>kPa
        var_input['Tair']   = Tair_obs # K
        var_input['Precip'] = Precip_obs # kg/m2/s
        var_input['Wind']   = Wind_obs # m/s
        var_input['Psurf']  = Psurf_obs # Pa
        var_input['Qle']    = Qle_obs # W/m2
        var_input['Qh']     = Qh_obs # W/m2
        var_input['Qg']     = Qg_obs # W/m2
        var_input['GPP']    = GPP_obs # umol/m2/s
        var_input['Rnet']   = Rnet_obs # W/m2
        var_input['Ustar']  = Ustar_obs # m/s

        # Read model flux data
        if model_name != 'obs':

            file_path = glob.glob(PLUMBER2_path+model_name +"/*"+site_name+"*.nc")

            if not file_path:
                # if the model doesn't simulate this site
                return
            else:
                # Get model fluxes
                Qle_model, Qh_model, GPP_model, Rnet_model, Qg_model = \
                        read_model_flux_variables(file_path, Qle_dict[model_name],
                                                Qh_dict[model_name], Qg_dict[model_name],
                                                Rnet_dict[model_name], GPP_dict[model_name])

                # Check the time interval
                ntime_obs   = len(Qle_obs)
                ntime_model = len(Qle_model)

                # Give values to var_input
                if ntime_model == ntime_obs:
                    # if half-hourly as observation
                    if np.any(~np.isnan(Qle_model)):
                        var_input['Qle'] = Qle_model
                    else:
                        message = message+'         '+model_name+' Qle not exist.\n'

                    if np.any(~np.isnan(Qh_model)):
                        var_input['Qh']  = Qh_model
                    else:
                        message = message+'         '+model_name+' Qh not exist.\n'

                    if np.any(~np.isnan(Rnet_model)):
                        var_input['Rnet']= Rnet_model
                        if np.any(~np.isnan(Qg_model)):
                            var_input['Qg']  = Qg_model
                        else:
                            var_input['Qg']  = np.nan
                            message = message+'         '+model_name+' Rnet exists, but Qg not. Set Qg=nan\n'
                    else:
                        message = message+'         '+model_name+' Rnet not exist, use obs Rnet and Qh.\n'

                    if np.any(~np.isnan(GPP_model)):
                        var_input['GPP'] = GPP_model
                    else:
                        message = message+'         '+model_name+' GPP not exist.\n'

                elif ntime_model == int(ntime_obs/2):
                    # if it is hourly
                    print('model ', model_name, ' is hourly, model_ntime is', ntime_model, ' ntime_obs is', ntime_obs)

                    # put the value of hourly data to the first half hour
                    if np.any(~np.isnan(Qle_model)):
                        var_input['Qle'][::2]  = Qle_model
                        var_input['Qle'][1::2] = Qle_model
                    else:
                        message = message+'         '+model_name+' Qle not exist.\n'

                    if np.any(~np.isnan(Qh_model)):
                        var_input['Qh'][::2]  = Qh_model
                        var_input['Qh'][1::2] = Qh_model
                    else:
                        message = message+'         '+model_name+' Qh not exist.\n'

                    if np.any(~np.isnan(Rnet_model)):
                        var_input['Rnet'][::2]  = Rnet_model
                        var_input['Rnet'][1::2] = Rnet_model
                        if np.any(~np.isnan(Qg_model)):
                            var_input['Qg'][::2]  = Qg_model
                            var_input['Qg'][1::2] = Qg_model
                        else:
                            var_input['Qg']  = np.nan
                            message = message+'         '+model_name+' Rnet exists, but Qg not. Set Qg=nan\n'
                    else:
                        message = message+'         '+model_name+' Rnet not exist, use obs Rnet and Qh.\n'

                    if np.any(~np.isnan(GPP_model)):
                        var_input['GPP'][::2]  = GPP_model
                        var_input['GPP'][1::2] = GPP_model
                    else:
                        message = message+'         '+model_name+' GPP not exist.\n'

                else:
                    # if it is hourly
                    print('Error occur! ntime_model is ', ntime_model, 'ntime_obs is', ntime_obs)

        # ================== Prepare data ==================

        # Convert units ...
        var_input['Tair'] -= 273.15

        # screen for dew
        var_input['Qle'] = np.where( var_input['Qle'] > 0.0, var_input['Qle'], np.nan)

        # Calculate ET, W m-2 to kg m-2 s-1
        lhv              = latent_heat_vapourisation(var_input['Tair'])
        var_input['ET']  = var_input['Qle'] / lhv

        # kg m-2 s-1 to mol m-2 s-1
        conv            = c.KG_TO_G * c.G_WATER_TO_MOL_WATER
        var_input['ET']*= conv

        # screen highly saturated conditions
        # var_input['VPD']= np.where(var_input['VPD']> 0.05, var_input['VPD'], np.nan)
        # MMY: VPD units is kPa
        #      When VPD == 0, the calcuated Gs can be very weird values, I don't screen
        #      0 values here but in making plots VPD=0 need to be excluded, e.g. set the 
        #      VPD bins start from 0.02 rather than 0.

        if calc_ref:
            # set VPD for reference Gs as 1000 Pa
            var_input['VPD']= 1000.
        else:
            # kPa to Pa
            var_input['VPD']= var_input['VPD']*1000.

        # (var_input, no_G) = filter_dataframe(var_input, hour) # MMY: hour tells it is hourly or half-hour data
        # if no_G:
        #     G = None
        # print(var_input)

        # To avoid crash in Gs calculation, set the row with any missing value in a column as np.nan
        subset            = ['VPD', 'Rnet', 'Wind', 'Tair', 'Psurf', 'ET']
        var_input[subset] = var_input[subset].where(~var_input[subset].isna().any(axis=1), other=np.nan)

        # ==================
        """
        PM = PenmanMonteith(use_ustar=False)
        # Height from Wilkinson, M., Eaton, E. L., Broadmeadow, M. S. J., and
        # Morison, J. I. L.: Inter-annual variation of carbon uptake by a
        # plantation oak woodland in south-eastern England, Biogeosciences, 9,
        # 5373–5389, https://doi.org/10.5194/bg-9-5373-2012, 2012.
        (var_input['Gs'],
        var_input['VPDl'])  = PM.invert_penman(var_input['VPD'].values, var_input['Wind'].values,
                                        var_input['Rnet'].values, var_input['Tair'].values,
                                        var_input['Psurf'].values,
                                        var_input['ET'].values, canht=28., G=G)
        """
        if np.all(~np.isnan(var_input['Qg'])):
            G = var_input['Qg']
            message = '      Using Qg'
        else:
            G = None

        if np.all(~np.isnan(var_input['Ustar'])):
            PM = PenmanMonteith(use_ustar=True)
            var_input['Rnet_caused_ratio'] = PM.calc_rnet_caused_to_LH_ratio(var_input['VPD'].values, var_input['Wind'].values,
                                             var_input['Rnet'].values, var_input['Tair'].values,
                                             var_input['Psurf'].values, ustar=var_input["Ustar"], G=G)

        # screen for bad inverted data
        # MMY, I will filter these data in process_step1 so comment out them here
        # mask_Gs               = (var_input['Gs'] <= 0.0) | (var_input['Gs'] > 4.5) | (np.isnan(var_input['Gs']))
        # var_input[mask_Gs]    = np.nan
        # mask_VPDl             = (var_input['VPDl'] <= 0.05 * 1000.) | (var_input['VPDl'] > 7.* 1000) | (np.isnan(var_input['VPDl']))
        # var_input[mask_VPDl]  = np.nan

        # VPDa = var_input['VPD'] * c.PA_TO_KPA
        # VPDl = var_input['VPDl'] * c.PA_TO_KPA

        # print(var_input)

        with open(f'./txt/process1_output/Rnet_caused_LH_ratio/Message_Rnet_caused_LH_ratio_{site_name}_{model_name}.txt', 'w') as f:
            f.write(message)
        var_input.to_csv(f'./txt/process1_output/Rnet_caused_LH_ratio/Rnet_caused_LH_ratio_{site_name}_{model_name}.csv') # , mode='a', index=False

        var_input = None
        message   = None
    return

def read_met_variables(PLUMBER2_met_path, site_name):

    '''
    Read met variable from PLUMBER2 met files: 'VPD', 'Tair', 'Precip', 'Wind', 'Psurf',
    Other variables may need: 'CO2air', 'CO2air_qc'
    '''

    file_path = glob.glob(PLUMBER2_met_path+site_name+"*.nc")
    # print('file_path1',file_path)
    f         = nc.Dataset(file_path[0])
    VPD       = np.squeeze(f.variables['VPD'][:])
    Tair      = np.squeeze(f.variables['Tair'][:])
    Precip    = np.squeeze(f.variables['Precip'][:])
    Wind      = np.squeeze(f.variables['Wind'][:])
    Psurf     = np.squeeze(f.variables['Psurf'][:])
    canht     = np.squeeze(f.variables['canopy_height'][:])

    return canht, VPD, Tair, Precip, Wind, Psurf

def read_obs_flux_variables(PLUMBER2_flux_path, site_name, message):

    '''
    Read obs from PLUMBER2 flux files: 'Qle', 'Qh', 'GPP', 'Rnet', 'Qg'
    Other variables may need: 'Qle_cor', 'Qh_cor', 'Qg_qc', 'Qle_qc', 'Qh_qc',
                              'Qle_cor_uc', 'Qh_cor_uc',
    '''

    # Read model flux variables
    file_path = glob.glob(PLUMBER2_flux_path+site_name+"*.nc")
    # print('file_path2',file_path)
    f         = nc.Dataset(file_path[0])

    try:
        Qle   = np.squeeze(f.variables['Qle'][:])
    except:
        Qle   = np.nan
        message = '         Obs Qle not exist.\n'

    try:
        Qh    = np.squeeze(f.variables['Qh'][:])
    except:
        Qh    = np.nan
        message = '         Obs Qh not exist.\n'

    try:
        GPP   = np.squeeze(f.variables['GPP'][:])
    except:
        GPP   = np.nan
        message = '         Obs GPP not exist.\n'

    try:
        Rnet  = np.squeeze(f.variables['Rnet'][:])
    except:
        Rnet  = np.nan
        message = '         Obs Rnet not exist.\n'

    try:
        Qg    = np.squeeze(f.variables['Qg'][:])
    except:
        Qg    = np.nan
        # message = '         Obs Qg not exist.\n'

    try:
        Ustar = np.squeeze(f.variables['Ustar'][:])
    except:
        Ustar = np.nan
        message = '         Obs Ustar not exist.\n'

    return Qle, Qh, GPP, Rnet, Qg, Ustar, message

def read_model_flux_variables(file_path, Qle_name=None, Qh_name=None, Qg_name=None, Rnet_name=None, GPP_name=None):

    '''
    Read obs from model files: 'Qle', 'Qh', 'GPP', 'Rnet', Qg'
    models with Qg: ACASA, CABLE, CABLE-POP-CN, MuSICA, ORC2_r6593, ORC2_r6593_CO2, STEMMUS-SCOPE
    '''

    try:
        f         = nc.Dataset(file_path[0])
        if not ('None' in Qle_name):
            Qle_model = np.squeeze(f.variables[Qle_name][:])
        else:
            Qle_model = np.nan

        if not ('None' in Qh_name):
            Qh_model  = np.squeeze(f.variables[Qh_name][:])
        else:
            Qh_model  = np.nan

        if not ('None' in Qg_name):
            Qg_model  = np.squeeze(f.variables[Qg_name][:])
        else:
            Qg_model  = np.nan

        if not ('None' in GPP_name):
            GPP_model = np.squeeze(f.variables[GPP_name][:])
        else:
            GPP_model = np.nan

        print('Rnet_name',Rnet_name)
        if ('None' in Rnet_name):
            Rnet_model = np.nan
        elif len(Rnet_name) == 2:
            Rnet_model = np.squeeze(f.variables[Rnet_name[0]][:]+f.variables[Rnet_name[1]][:])
        else:
            Rnet_model = np.squeeze(f.variables[Rnet_name][:])

    except:
        Qle_model, Qh_model, GPP_model, Rnet_model, Qg_model = np.nan, np.nan, np.nan, np.nan, np.nan

    return Qle_model, Qh_model, GPP_model, Rnet_model, Qg_model

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

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"
    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

    site_names, IGBP_types, clim_types, model_names_list = load_default_list()
    # The site names
    model_names        = model_names_list['model_select_new']
    calc_ref           = True
    main(PLUMBER2_path, PLUMBER2_met_path, PLUMBER2_flux_path, site_names, model_names, calc_ref)
