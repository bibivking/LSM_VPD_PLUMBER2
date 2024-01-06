#!/usr/bin/env python
"""
Estimate the ecosystem conductance (Gs) from inverting the penman-monteith
against eddy covariance flux data. Finally, make a 1:1 plot of VPD_leaf vs
VPD_atmospheric

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (20.06.2023)"
__email__ = "mdekauwe@gmail.com"

#import matplotlib
#matplotlib.use('agg') # stop windows popping up

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import datetime as dt
from scipy.stats import pearsonr
from rmse import rmse
import re
from datetime import datetime
from lmfit import minimize, Parameters

sys.path.append('src')
import constants as c
from penman_monteith import PenmanMonteith
from estimate_pressure import estimate_pressure


def main(fname, hour=False):

    site_name = "Alice_Holt"
    model_out_list
    for model_name in model_out_list:
        if model_name != 'obs':
            varname = 'model_'+model_name
        else:
            varname = model_name

    # Check here
    date_parse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(fname, index_col='DateTime',
                     parse_dates=['DateTime'],
                     date_parser=date_parse)
    df.index.names = ['date']
    ################ 
    
,Unnamed: 0,year,month,day,site_name,IGBP_type,climate_type,Unnamed: 0.1,model_3km27,3km27_EF,model_6km729,6km729_EF,model_6km729lag,6km729lag_EF,model_ACASA,ACASA_EF,model_CABLE,CABLE_EF,model_CABLE-POP-CN,CABLE-POP-CN_EF,model_CHTESSEL_ERA5_3,CHTESSEL_ERA5_3_EF,model_CHTESSEL_Ref_exp1,CHTESSEL_Ref_exp1_EF,model_CLM5a,CLM5a_EF,model_GFDL,GFDL_EF,model_JULES_GL9_withLAI,JULES_GL9_withLAI_EF,model_JULES_test,JULES_test_EF,model_LSTM_eb,LSTM_eb_EF,model_LSTM_raw,LSTM_raw_EF,model_Manabe,Manabe_EF,model_ManabeV2,ManabeV2_EF,model_MATSIRO,MATSIRO_EF,model_MuSICA,MuSICA_EF,model_NASAEnt,NASAEnt_EF,model_NoahMPv401,NoahMPv401_EF,model_ORC2_r6593,ORC2_r6593_EF,model_ORC2_r6593_CO2,ORC2_r6593_CO2_EF,model_ORC3_r7245_NEE,ORC3_r7245_NEE_EF,model_ORC3_r8120,ORC3_r8120_EF,model_PenmanMonteith,PenmanMonteith_EF,model_QUINCY,QUINCY_EF,model_RF_eb,RF_eb_EF,model_RF_raw,RF_raw_EF,model_STEMMUS-SCOPE,STEMMUS-SCOPE_EF,obs,obs_cor,obs_EF,VPD,
obs_Tair,obs_Qair,obs_Precip,obs_SWdown,NoahMPv401_greenness,hour,half_hrs_after_precip

    # Convert units ...
    df.loc[:, 'obs_Tair'] -= 273.15

    # screen for dew
    #df = df[df['LE'] > 0.0]

    # W m-2 to kg m-2 s-1
    lhv = latent_heat_vapourisation(df['obs_Tair'])
    df.loc[:, varname] = df[varname] / lhv


    # kg m-2 s-1 to mol m-2 s-1
    conv = c.KG_TO_G * c.G_WATER_TO_MOL_WATER
    df.loc[:, varname] *= conv

    (df, no_G) = filter_dataframe(df, hour)
    if no_G:
        G = None

    df = df.dropna(subset = ['VPD', 'Rnet', 'Wind', 'Tair', 'Psurf', 'ET'])
    df = df[df.VPD > 1000* 0.05]


    """
    PM = PenmanMonteith(use_ustar=False)
    # Height from Wilkinson, M., Eaton, E. L., Broadmeadow, M. S. J., and
    # Morison, J. I. L.: Inter-annual variation of carbon uptake by a
    # plantation oak woodland in south-eastern England, Biogeosciences, 9,
    # 5373–5389, https://doi.org/10.5194/bg-9-5373-2012, 2012.
    (df['Gs'],
     df['VPDl'])  = PM.invert_penman(df['VPD'].values, df['Wind'].values,
                                     df['Rnet'].values, df['Tair'].values,
                                     df['Psurf'].values,
                                     df['ET'].values, canht=28., G=G)
    """
    PM = PenmanMonteith(use_ustar=True)
    (df['Gs'],
     df['VPDl'])  = PM.invert_penman(df['VPD'].values, df['Wind'].values,
                                     df['Rnet'].values, df['Tair'].values,
                                     df['Psurf'].values,
                                     df['ET'].values,
                                     ustar=df["Ustar"], G=G)



    # screen for bad inverted data
    df = df[(df['Gs'] > 0.0) & (df['Gs'] < 4.5) & (np.isnan(df['Gs']) == False)]
    df = df[(df['VPDl'] > 0.05 * c.PA_TO_KPA) & \
            (df['VPDl'] < 7.* 1000)  & \
            (np.isnan(df['VPDl']) == False)]

    VPDa = df['VPD'] * c.PA_TO_KPA
    VPDl = df['VPDl'] * c.PA_TO_KPA

    plot_VPD(VPDa, VPDl, site_name)
    plot_gs_vs_D(df['Gs'], VPDa, df['ET'], site_name)
    plot_LE_vs_Tair(df['LE'], df['Tair'], site_name)

    # When parameterizing the model of Eq. 1 in the main text, we
    # limited the data to those collected when VPD > 1.0 kPa. From Novick sup
    # remove stable conditions
    #df = df[(df.VPD/1000. >= 1.) & (df.Wind >= 1.)]

    gs_ref = np.mean(df[(df.VPD * c.PA_TO_KPA > 0.9) & \
                        (df.VPD * c.PA_TO_KPA < 1.1)].Gs)

    params = Parameters()
    params.add('m', value=0.5)
    params.add('gs_ref', value=gs_ref, vary=False)


    result = minimize(residual, params, args=(df['VPD']*c.PA_TO_KPA, df['Gs']))
    for name, par in result.params.items():
        print('%s = %.4f +/- %.4f ' % (name, par.value, par.stderr))

    m_pred = result.params['m'].value

    #plt.plot(np.log(df['VPD']*c.PA_TO_KPA), df['Gs'], "ro")
    #plt.plot(np.log(df['VPD']*c.PA_TO_KPA),
    #         np.log(df['VPD']*c.PA_TO_KPA) * m_pred + gs_ref, "k-")
    #plt.show()

def gs_model_lohammar(VPD, m, gs_ref):
    return -m * np.log(VPD) + gs_ref

def residual(params, VPD, obs):
    m = params['m'].value
    gs_ref = params['gs_ref'].value
    model = gs_model_lohammar(VPD, m, gs_ref)

    return (obs - model)

def plot_gs_vs_D(Gs, VPDa, ET, site_name):

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)
    #ax.set_aspect('equal', adjustable='box')

    ln1 = ax.plot(VPDa, Gs, marker="o", ls=" ", color="royalblue", alpha=0.3,
                  label="G$_{s}$", markersize=7,
                  markeredgecolor="None")

    ax2 = ax.twinx()
    ax2.set_ylabel("Evapotranspiration (mmol m$^{-2}$ s$^{-1}$)")
    ln2 = ax2.plot(VPDa, ET*1000, "go", alpha=0.3, label="ET",
                   markersize=7, markeredgecolor="None")

    # added these three lines
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, numpoints=1, loc="best")


    ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
    ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

    #ax.set_ylim(0, 6)
    #ax.set_xlim(0, 6)
    ax.set_xlabel("VPD (kPa)")
    ax.set_ylabel("Ecosystem conductance (mol m$^{-2}$ s$^{-1}$)")


    odir = "plots"
    ofname = "%s_gs_vs_VPD_2022.pdf" % (site_name)
    fig.savefig(os.path.join(odir, ofname),
                bbox_inches='tight', pad_inches=0.1)

def plot_VPD(VPDa, VPDl, site_name):

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    ax.plot(VPDa, VPDl, "k.", alpha=0.5)

    one2one = np.array([0, 350])
    ax.plot(one2one, one2one, ls='--', color="grey", label="1:1 line")

    r, pval = pearsonr(VPDa, VPDl)
    print(r, pval)
    #if pval <= 0.05:
    m,c = np.polyfit(VPDa, VPDl, 1)
    ax.plot(VPDa, VPDa*m+c, ls="-", c="red")


    ax.set_ylim(0, 6)
    ax.set_xlim(0, 6)
    ax.set_xlabel("VPD$_a$ (kPa)")
    ax.set_ylabel("VPD$_l$ (kPa)")

    ax.text(3.5, 0.5, 'R$^{2}$ = %0.2f' % r**2)
    ax.text(3.5, 0.25, 'm = %0.2f; c = %0.2f' % (m, c))

    odir = "plots"

    ofname = "%s_2022.pdf" % (site_name)
    fig.savefig(os.path.join(odir, ofname),
                bbox_inches='tight', pad_inches=0.1)


def plot_LE_vs_Tair(LE, Tair, site_name):

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)
    #ax.set_aspect('equal', adjustable='box')

    ln1 = ax.plot(Tair, LE, marker="o", ls=" ", alpha=0.5, color="royalblue")


    #ax.set_ylim(0, 6)
    #ax.set_xlim(0, 6)
    ax.set_ylabel("LE (W m$^{-2}$)")
    ax.set_xlabel('Temperature ($\degree$C)')

    odir = "plots"
    ofname = "%s_LE_vs_Tair_2022.pdf" % (site_name)
    fig.savefig(os.path.join(odir, ofname),
                bbox_inches='tight', pad_inches=0.1)



def get_site_info(df_site, fname):

    d = {}
    s = os.path.basename(fname).split(".")[0].split("_")[1].strip()

    d['site'] = s
    d['yrs'] = os.path.basename(fname).split(".")[0].split("_")[5]
    d['lat'] = df_site.loc[df_site.SiteCode == s,'SiteLatitude'].values[0]
    d['lon'] = df_site.loc[df_site.SiteCode == s,'SiteLongitude'].values[0]
    d['pft'] = df_site.loc[df_site.SiteCode == s,\
                           'IGBP_vegetation_short'].values[0]
    d['pft_long'] = df_site.loc[df_site.SiteCode == s,\
                                'IGBP_vegetation_long'].values[0]

    # remove commas from country tag as it messes out csv output
    name = df_site.loc[df_site.SiteCode == s,'Fullname'].values[0]
    d['name'] = name.replace("," ,"")
    d['country'] = df_site.loc[df_site.SiteCode == s,'Country'].values[0]
    d['elev'] = df_site.loc[df_site.SiteCode == s,'SiteElevation'].values[0]
    d['Vegetation_description'] = df_site.loc[df_site.SiteCode == s,\
                                        'VegetationDescription'].values[0]
    d['soil_type'] = df_site.loc[df_site.SiteCode == s,\
                                        'SoilType'].values[0]
    d['disturbance'] = df_site.loc[df_site.SiteCode == s,\
                                        'Disturbance'].values[0]
    d['crop_description'] = df_site.loc[df_site.SiteCode == s,\
                                        'CropDescription'].values[0]
    d['irrigation'] = df_site.loc[df_site.SiteCode == s,\
                                        'Irrigation'].values[0]
    d['measurement_ht'] = -999.9
    try:
        ht = float(df_site.loc[df_site.SiteCode == s, \
                   'MeasurementHeight'].values[0])
        if ~np.isnan(ht):
            d['measurement_ht'] = ht
    except IndexError:
        pass

    d['tower_ht'] = -999.9
    try:
        ht = float(df_site.loc[df_site.SiteCode == s, \
                   'TowerHeight'].values[0])
        if ~np.isnan(ht):
            d['tower_ht'] = ht
    except IndexError:
        pass

    d['canopy_ht'] = -999.9
    try:
        ht = float(df_site.loc[df_site.SiteCode == s, \
                   'CanopyHeight'].values[0])
        if ~np.isnan(ht):
            d['canopy_ht'] = ht
    except IndexError:
        pass

    return (d)


def read_file(fname):

    date_parse = lambda x: datetime.strptime(x, '%Y%m%d%H%M%S')

    df = pd.read_csv(fname, index_col='TIMESTAMP_START',
                     parse_dates=['TIMESTAMP_START'],
                     date_parser=date_parse)
    df.index.names = ['date']

    # Using ERA interim filled met vars ... _F
    df = df.rename(columns={'LE_F_MDS': 'Qle', 'H_F_MDS': 'Qh',
                            'VPD_F_MDS': 'VPD', 'TA_F': 'Tair',
                            'NETRAD': 'Rnet',
                            'G_F_MDS': 'Qg',
                            'WS_F': 'Wind', 'P_F': 'Precip',
                            'USTAR': 'ustar', 'LE_CORR': 'Qle_cor',
                            'H_CORR': 'Qh_cor', 'CO2_F_MDS': 'CO2air',
                            'CO2_F_MDS_QC': 'CO2air_qc', 'PA_F': 'Psurf',
                            'G_F_MDS_QC': 'Qg_qc',
                            'LE_F_MDS_QC': 'Qle_qc', 'H_F_MDS_QC': 'Qh_qc',
                            'LE_CORR_JOINTUNC': 'Qle_cor_uc',
                            'H_CORR_JOINTUNC': 'Qh_cor_uc',
                            'GPP_NT_VUT_REF': 'GPP'})


    df = df[['Qle', 'Qh', 'VPD', 'Tair', 'Rnet', 'Qg', 'Wind', \
             'Precip', 'ustar', 'Qle_cor', 'Qh_cor', 'Psurf',\
             'CO2air', 'CO2air_qc', 'Qg_qc', 'Qle_qc', 'Qh_qc', \
             'Qle_cor_uc', 'Qh_cor_uc', 'GPP']]

    # Convert units ...


    # hPa -> Pa
    df.loc[:, 'VPD'] *= c.HPA_TO_KPA * c.KPA_TO_PA

    # kPa -> Pa
    df.loc[:, 'Psurf'] *= c.KPA_TO_PA

    # W m-2 to kg m-2 s-1
    lhv = latent_heat_vapourisation(df['Tair'])
    df.loc[:, 'ET'] = df['Qle'] / lhv


    # Use EBR value instead - uncomment to output this correction
    #df.loc[:, 'ET'] = df['Qle_cor'] / lhv


    # kg m-2 s-1 to mol m-2 s-1
    conv = c.KG_TO_G * c.G_WATER_TO_MOL_WATER
    df.loc[:, 'ET'] *= conv

    # screen by low u*, i.e. conditions which are often indicative of
    # poorly developed turbulence, after Sanchez et al. 2010, HESS, 14,
    # 1487-1497. Some authors use 0.3 m s-1 (Oliphant et al. 2004) or
    # 0.35 m s-1 (Barr et al. 2006) as a threshold for u*
    df = df[df.ustar >= 0.25]

    # screen for bad data
    df = df[df['Rnet'] > -900.0]

    return (df)

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

def filter_dataframe(df, hour):
    """
    Filter data only using QA=0 (obs) and QA=1 (good)
    """
    no_G = False

    # filter daylight hours
    #
    # If we have no ground heat flux, just use Rn
    no_G = True



    df = df[(df.index.hour >= 7) &
            (df.index.hour <= 18) &
            (df['ET'] > 0.01 / 1000.) &  # check in mmol, but units are mol
            (df['VPD']/1000 > 0.05)]

    #"""
    # Filter events after rain ...
    idx = df[df.Rainf > 0.0].index.tolist()

    if hour:
        # hour gap i.e. Tumba
        bad_dates = []
        for rain_idx in idx:
            bad_dates.append(rain_idx)
            for i in range(24):
                new_idx = rain_idx + dt.timedelta(minutes=60)
                bad_dates.append(new_idx)
                rain_idx = new_idx
    else:


        # 30 min gap
        bad_dates = []
        for rain_idx in idx:
            bad_dates.append(rain_idx)
            for i in range(48):
                new_idx = rain_idx + dt.timedelta(minutes=30)
                bad_dates.append(new_idx)
                rain_idx = new_idx

    # There will be duplicate dates most likely so remove these.
    bad_dates = np.unique(bad_dates)

    # remove rain days...
    df = df[~df.index.isin(bad_dates)]
    #"""

    return (df, no_G)

if __name__ == "__main__":

    fname = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process3_output/curves/raw_data_Qle_VPD_daily_RM16_SLCT_EF_model_0-0.2.csv"
    main(fname, hour=False)
