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
# from lmfit import minimize, Parameters
from PLUMBER2_VPD_common_utils import *
import pdb

sys.path.append('src')
import Martin_constants as c
from Martin_penman_monteith import PenmanMonteith

def main(PLUMBER2_path, PLUMBER2_met_path, PLUMBER2_flux_path, site_names, model_names):

    for site_name in site_names:
        print('site_name',site_name)
        Qle_dict = check_variable_exists(PLUMBER2_path, 'Qle', site_name, model_names)
        # print('Qle_dict',Qle_dict)
        Qh_dict  = check_variable_exists(PLUMBER2_path, 'Qh', site_name, model_names)
        # print('Qh_dict',Qh_dict)
        Qg_dict  = check_variable_exists(PLUMBER2_path, 'Qg', site_name, model_names)
        # print('Qg_dict',Qg_dict)
        GPP_dict = check_variable_exists(PLUMBER2_path, 'GPP', site_name, model_names)
        # print('GPP_dict',GPP_dict)
        Rnet_dict= check_variable_exists(PLUMBER2_path, 'Rnet', site_name, model_names)
        # print('Rnet_dict',Rnet_dict)

        calc_gs_one_site(PLUMBER2_met_path, PLUMBER2_flux_path, site_name, model_names,
                         Qle_dict, Qh_dict, Qg_dict, Rnet_dict, GPP_dict)
        # pdb.set_trace()
    return

def calc_gs_one_site(PLUMBER2_met_path, PLUMBER2_flux_path, site_name, model_names,
                    Qle_dict, Qh_dict, Qg_dict, Rnet_dict, GPP_dict):

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
        var_input           = pd.DataFrame(VPD_obs, columns=['VPD'])
        var_input['Tair']   = Tair_obs
        var_input['Precip'] = Precip_obs
        var_input['Wind']   = Wind_obs
        var_input['Psurf']  = Psurf_obs
        var_input['Qle']    = Qle_obs
        var_input['Qh']     = Qh_obs
        var_input['Qg']     = Qg_obs
        var_input['GPP']    = GPP_obs
        var_input['Rnet']   = Rnet_obs
        var_input['Ustar']  = Ustar_obs

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

                    if np.any(~np.isnan(Qg_model)):
                        var_input['Qg']  = Qg_model
                    # else:
                    #     message = message+'         '+model_name+' Qg not exist.\n'

                    if np.any(~np.isnan(Rnet_model)):
                        var_input['Rnet']= Rnet_model
                    else:
                        message = message+'         '+model_name+' Rnet not exist.\n'

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

                    if np.any(~np.isnan(Qg_model)):
                        var_input['Qg'][::2]  = Qg_model
                        var_input['Qg'][1::2] = Qg_model
                    # else:
                    #     message = message+'         '+model_name+' Qg not exist.\n'

                    if np.any(~np.isnan(Rnet_model)):
                        var_input['Rnet'][::2]  = Rnet_model
                        var_input['Rnet'][1::2] = Rnet_model
                    else:
                        message = message+'         '+model_name+' Rnet not exist.\n'

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
        var_input['VPD']= np.where(var_input['VPD']> 0.05, var_input['VPD'], np.nan)
        # MMY: VPD units is kPa

        # kPa to Pa
        var_input['VPD']= var_input['VPD']*1000.

        # (var_input, no_G) = filter_dataframe(var_input, hour) # MMY: hour tells it is hourly or half-hour data
        # if no_G:
        #     G = None
        # print(var_input)

        # print("np.sum(var_input['GPP'])",np.sum(var_input['GPP']))
        # print('Before len(var_input)',len(var_input))
        # To avoid crash in Gs calculation, set the row with any missing value in a column as np.nan
        subset            = ['VPD', 'Rnet', 'Wind', 'Tair', 'Psurf', 'ET']
        var_input[subset] = var_input[subset].where(~var_input[subset].isna().any(axis=1), other=np.nan)
        # print('After len(var_input)',len(var_input))

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

        # plot_VPD(VPDa, VPDl, site_name)
        # plot_gs_vs_D(var_input['Gs'], VPDa, var_input['ET'], site_name)
        # plot_LE_vs_Tair(var_input['LE'], var_input['Tair'], site_name)

        # When parameterizing the model of Eq. 1 in the main text, we
        # limited the data to those collected when VPD > 1.0 kPa. From Novick sup
        # remove stable conditions
        #var_input = var_input[(var_input.VPD/1000. >= 1.) & (var_input.Wind >= 1.)]

        # gs_ref = np.mean(var_input[(var_input['VPD'] * c.PA_TO_KPA > 0.9) & \
        #                     (var_input['VPD'] * c.PA_TO_KPA < 1.1)]['Gs'])

        # params = Parameters()
        # params.add('m', value=0.5)
        # params.add('gs_ref', value=gs_ref, vary=False)

        # result = minimize(residual, params, args=(var_input['VPD']*c.PA_TO_KPA, var_input['Gs']))
        # for name, par in result.params.items():
        #     print('%s = %.4f +/- %.4f ' % (name, par.value, par.stderr))

        # m_pred = result.params['m'].value

        #plt.plot(np.log(var_input['VPD']*c.PA_TO_KPA), var_input['Gs'], "ro")
        #plt.plot(np.log(var_input['VPD']*c.PA_TO_KPA),
        #         np.log(var_input['VPD']*c.PA_TO_KPA) * m_pred + gs_ref, "k-")
        #plt.show()

        # print(var_input)
        with open(f'./txt/process1_output/Gs/Message_gs_{site_name}_{model_name}.txt', 'w') as f:
            f.write(message)
        var_input.to_csv(f'./txt/process1_output/Gs/Gs_{site_name}_{model_name}.csv') # , mode='a', index=False
        var_input = None
        message   = None
    return

# def gs_model_lohammar(VPD, m, gs_ref):
#     return -m * np.log(VPD) + gs_ref

# def residual(params, VPD, obs):
#     m = params['m'].value
#     gs_ref = params['gs_ref'].value
#     model = gs_model_lohammar(VPD, m, gs_ref)

#     return (obs - model)

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


# def plot_gs_vs_D(Gs, VPDa, ET, site_name):

#     fig = plt.figure(figsize=(9,6))
#     fig.subplots_adjust(hspace=0.1)
#     fig.subplots_adjust(wspace=0.05)
#     plt.rcParams['text.usetex'] = False
#     plt.rcParams['font.family'] = "sans-serif"
#     plt.rcParams['font.sans-serif'] = "Helvetica"
#     plt.rcParams['axes.labelsize'] = 14
#     plt.rcParams['font.size'] = 14
#     plt.rcParams['legend.fontsize'] = 14
#     plt.rcParams['xtick.labelsize'] = 14
#     plt.rcParams['ytick.labelsize'] = 14

#     ax = fig.add_subplot(111)
#     #ax.set_aspect('equal', adjustable='box')

#     ln1 = ax.plot(VPDa, Gs, marker="o", ls=" ", color="royalblue", alpha=0.3,
#                   label="G$_{s}$", markersize=7,
#                   markeredgecolor="None")

#     ax2 = ax.twinx()
#     ax2.set_ylabel("Evapotranspiration (mmol m$^{-2}$ s$^{-1}$)")
#     ln2 = ax2.plot(VPDa, ET*1000, "go", alpha=0.3, label="ET",
#                    markersize=7, markeredgecolor="None")

#     # added these three lines
#     lns = ln1+ln2
#     labs = [l.get_label() for l in lns]
#     ax.legend(lns, labs, numpoints=1, loc="best")


#     ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
#     ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

#     #ax.set_ylim(0, 6)
#     #ax.set_xlim(0, 6)
#     ax.set_xlabel("VPD (kPa)")
#     ax.set_ylabel("Ecosystem conductance (mol m$^{-2}$ s$^{-1}$)")


#     odir = "plots"
#     ofname = "%s_gs_vs_VPD_2022.pdf" % (site_name)
#     fig.savefig(os.path.join(odir, ofname),
#                 bbox_inches='tight', pad_inches=0.1)

# def plot_VPD(VPDa, VPDl, site_name):

#     fig = plt.figure(figsize=(9,6))
#     fig.subplots_adjust(hspace=0.1)
#     fig.subplots_adjust(wspace=0.05)
#     plt.rcParams['text.usetex'] = False
#     plt.rcParams['font.family'] = "sans-serif"
#     plt.rcParams['font.sans-serif'] = "Helvetica"
#     plt.rcParams['axes.labelsize'] = 14
#     plt.rcParams['font.size'] = 14
#     plt.rcParams['legend.fontsize'] = 14
#     plt.rcParams['xtick.labelsize'] = 14
#     plt.rcParams['ytick.labelsize'] = 14

#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal', adjustable='box')

#     ax.plot(VPDa, VPDl, "k.", alpha=0.5)

#     one2one = np.array([0, 350])
#     ax.plot(one2one, one2one, ls='--', color="grey", label="1:1 line")

#     r, pval = pearsonr(VPDa, VPDl)
#     print(r, pval)
#     #if pval <= 0.05:
#     m,c = np.polyfit(VPDa, VPDl, 1)
#     ax.plot(VPDa, VPDa*m+c, ls="-", c="red")


#     ax.set_ylim(0, 6)
#     ax.set_xlim(0, 6)
#     ax.set_xlabel("VPD$_a$ (kPa)")
#     ax.set_ylabel("VPD$_l$ (kPa)")

#     ax.text(3.5, 0.5, 'R$^{2}$ = %0.2f' % r**2)
#     ax.text(3.5, 0.25, 'm = %0.2f; c = %0.2f' % (m, c))

#     odir = "plots"

#     ofname = "%s_2022.pdf" % (site_name)
#     fig.savefig(os.path.join(odir, ofname),
#                 bbox_inches='tight', pad_inches=0.1)


# def plot_LE_vs_Tair(LE, Tair, site_name):

#     fig = plt.figure(figsize=(9,6))
#     fig.subplots_adjust(hspace=0.1)
#     fig.subplots_adjust(wspace=0.05)
#     plt.rcParams['text.usetex'] = False
#     plt.rcParams['font.family'] = "sans-serif"
#     plt.rcParams['font.sans-serif'] = "Helvetica"
#     plt.rcParams['axes.labelsize'] = 14
#     plt.rcParams['font.size'] = 14
#     plt.rcParams['legend.fontsize'] = 14
#     plt.rcParams['xtick.labelsize'] = 14
#     plt.rcParams['ytick.labelsize'] = 14

#     ax = fig.add_subplot(111)
#     #ax.set_aspect('equal', adjustable='box')

#     ln1 = ax.plot(Tair, LE, marker="o", ls=" ", alpha=0.5, color="royalblue")


#     #ax.set_ylim(0, 6)
#     #ax.set_xlim(0, 6)
#     ax.set_ylabel("LE (W m$^{-2}$)")
#     ax.set_xlabel('Temperature ($\degree$C)')

#     odir = "plots"
#     ofname = "%s_LE_vs_Tair_2022.pdf" % (site_name)
#     fig.savefig(os.path.join(odir, ofname),
#                 bbox_inches='tight', pad_inches=0.1)



# def get_site_info(df_site, fname):

#     d = {}
#     s = os.path.basename(fname).split(".")[0].split("_")[1].strip()

#     d['site'] = s
#     d['yrs'] = os.path.basename(fname).split(".")[0].split("_")[5]
#     d['lat'] = df_site.loc[df_site.SiteCode == s,'SiteLatitude'].values[0]
#     d['lon'] = df_site.loc[df_site.SiteCode == s,'SiteLongitude'].values[0]
#     d['pft'] = df_site.loc[df_site.SiteCode == s,\
#                            'IGBP_vegetation_short'].values[0]
#     d['pft_long'] = df_site.loc[df_site.SiteCode == s,\
#                                 'IGBP_vegetation_long'].values[0]

#     # remove commas from country tag as it messes out csv output
#     name = df_site.loc[df_site.SiteCode == s,'Fullname'].values[0]
#     d['name'] = name.replace("," ,"")
#     d['country'] = df_site.loc[df_site.SiteCode == s,'Country'].values[0]
#     d['elev'] = df_site.loc[df_site.SiteCode == s,'SiteElevation'].values[0]
#     d['Vegetation_description'] = df_site.loc[df_site.SiteCode == s,\
#                                         'VegetationDescription'].values[0]
#     d['soil_type'] = df_site.loc[df_site.SiteCode == s,\
#                                         'SoilType'].values[0]
#     d['disturbance'] = df_site.loc[df_site.SiteCode == s,\
#                                         'Disturbance'].values[0]
#     d['crop_description'] = df_site.loc[df_site.SiteCode == s,\
#                                         'CropDescription'].values[0]
#     d['irrigation'] = df_site.loc[df_site.SiteCode == s,\
#                                         'Irrigation'].values[0]
#     d['measurement_ht'] = -999.9
#     try:
#         ht = float(df_site.loc[df_site.SiteCode == s, \
#                    'MeasurementHeight'].values[0])
#         if ~np.isnan(ht):
#             d['measurement_ht'] = ht
#     except IndexError:
#         pass

#     d['tower_ht'] = -999.9
#     try:
#         ht = float(df_site.loc[df_site.SiteCode == s, \
#                    'TowerHeight'].values[0])
#         if ~np.isnan(ht):
#             d['tower_ht'] = ht
#     except IndexError:
#         pass

#     d['canopy_ht'] = -999.9
#     try:
#         ht = float(df_site.loc[df_site.SiteCode == s, \
#                    'CanopyHeight'].values[0])
#         if ~np.isnan(ht):
#             d['canopy_ht'] = ht
#     except IndexError:
#         pass

#     return (d)

# MMY, calculate every time step's gs, so no need to filter gs here
#     but "(var_input['ET'] > 0.01 / 1000.) &  # check in mmol, but units are mol
#         (var_input['VPD']/1000 > 0.05)" might be useful

# def filter_dataframe(var_input, hour):
#     """
#     Filter data only using QA=0 (obs) and QA=1 (good)
#     """
#     no_G = False

#     # filter daylight hours
#     #
#     # If we have no ground heat flux, just use Rn
#     no_G = True


#     var_input = var_input[(var_input.index.hour >= 7) &
#             (var_input.index.hour <= 18) &
#             (var_input['ET'] > 0.01 / 1000.) &  # check in mmol, but units are mol
#             (var_input['VPD']/1000 > 0.05)]

#     #"""
#     # Filter events after rain ...
#     idx = var_input[var_input.Rainf > 0.0].index.tolist()

#     if hour:
#         # hour gap i.e. Tumba
#         bad_dates = []
#         for rain_idx in idx:
#             bad_dates.append(rain_idx)
#             for i in range(24):
#                 new_idx = rain_idx + dt.timedelta(minutes=60)
#                 bad_dates.append(new_idx)
#                 rain_idx = new_idx
#     else:


#         # 30 min gap
#         bad_dates = []
#         for rain_idx in idx:
#             bad_dates.append(rain_idx)
#             for i in range(48):
#                 new_idx = rain_idx + dt.timedelta(minutes=30)
#                 bad_dates.append(new_idx)
#                 rain_idx = new_idx

#     # There will be duplicate dates most likely so remove these.
#     bad_dates = np.unique(bad_dates)

#     # remove rain days...
#     var_input = var_input[~var_input.index.isin(bad_dates)]
#     #"""

#     return (var_input, no_G)

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"
    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

    site_names, IGBP_types, clim_types, model_names_list = load_default_list()
    # The site names
    model_names        = model_names_list['model_select']

    main(PLUMBER2_path, PLUMBER2_met_path, PLUMBER2_flux_path, site_names, model_names)
