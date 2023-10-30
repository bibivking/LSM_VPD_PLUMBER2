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
from calc_turning_points import *
from PLUMBER2_VPD_common_utils import *

def plot_var_VPD(var_name, bin_by=None, low_bound=None, high_bound=None,
                 day_time=False, summer_time=False, window_size=11, order=3,
                 smooth_type='S-G_filter', method='bin_by_vpd',
                 IGBP_type=None, clim_type=None, message=None,model_names=None,
                 turning_point={'calc':False,'method':'kneed'}):

    # ============== read data ==============
    file_name = ''

    if day_time:
        file_name = file_name + '_daytime'

    if IGBP_type !=None:
        file_name = file_name + '_IGBP='+IGBP_type

    if clim_type !=None:
        file_name = file_name + '_clim='+clim_type

    if len(low_bound) >1 and len(high_bound) >1:
        if low_bound[1] > 1:
            dry_file = f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'th_'+method+'_coarse.csv'
            wet_file = f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(high_bound[0])+'-'+str(high_bound[1])+'th_'+method+'_coarse.csv'
            var_dry = pd.read_csv(dry_file)
            var_wet = pd.read_csv(wet_file)
        else:
            dry_file = f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'_'+method+'_coarse.csv'
            wet_file = f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(high_bound[0])+'-'+str(high_bound[1])+'_'+method+'_coarse.csv'
            var_dry = pd.read_csv(dry_file)
            var_wet = pd.read_csv(wet_file)
    elif len(low_bound) == 1 and len(high_bound) == 1:
        if low_bound > 1:
            dry_file= f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(low_bound)+'th_'+method+'_coarse.csv'
            wet_file= f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(high_bound)+'th_'+method+'_coarse.csv'
            var_dry = pd.read_csv(dry_file)
            var_wet = pd.read_csv(wet_file)
        else:
            dry_file= f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(low_bound)+'_'+method+'_coarse.csv'
            wet_file= f'./txt/{var_name}_VPD'+file_name+'_'+bin_by+'_'+str(high_bound)+'_'+method+'_coarse.csv'
            var_dry = pd.read_csv(dry_file)
            var_wet = pd.read_csv(wet_file)

    print('Read ',dry_file,' and ', wet_file)

    # how to get the model out list from the column names???
    model_out_list = []
    for column_name in var_dry.columns:
        if "_vals" in column_name:
            model_out_list.append(column_name.split("_vals")[0])
    print('Checking model_out_list',model_out_list)

    # Calculate turning points
    if turning_point['calc']:
        # print('turning_point["calc"]',turning_point['calc'])
        nmodel   = len(model_out_list)
        nvpd     = len(var_dry['vpd_series'])

        # Smoothing the curve and remove vpd_num < 100.
        vals_dry = np.zeros((nmodel,nvpd))
        vals_wet = np.zeros((nmodel,nvpd))

        for i, model_out_name in enumerate(model_out_list):
            print('model_out_name',model_out_name)
            vals_vpd_num_dry = var_dry[model_out_name+'_vpd_num']
            vals_vpd_num_wet = var_wet[model_out_name+'_vpd_num']
            vals_dry[i,:]    = np.where(vals_vpd_num_dry>200, var_dry[model_out_name+'_vals'], np.nan)
            vals_wet[i,:]    = np.where(vals_vpd_num_wet>200, var_wet[model_out_name+'_vals'], np.nan)

        # Find the turning points
        if turning_point['method']=='kneed' :
            print('calculate turining points by find_turning_points_by_kneed')
            turning_points_dry = find_turning_points_by_kneed(model_out_list, var_dry['vpd_series'], vals_dry)
            turning_points_wet = find_turning_points_by_kneed(model_out_list, var_wet['vpd_series'], vals_wet)
        elif turning_point['method']=='cdf' :
            print('calculate turining points by find_turning_points_by_cdf')
            turning_points_dry = find_turning_points_by_cdf(model_out_list, var_dry['vpd_series'], vals_dry)
            turning_points_wet = find_turning_points_by_cdf(model_out_list, var_wet['vpd_series'], vals_wet)
        elif turning_point['method']=='piecewise':
            print('calculate turining points by find_turning_points_by_piecewise_regression')
            turning_points_dry,slope_dry = find_turning_points_by_piecewise_regression(model_out_list, var_dry['vpd_series'], vals_dry, var_name)
            turning_points_wet,slope_wet = find_turning_points_by_piecewise_regression(model_out_list, var_wet['vpd_series'], vals_wet, var_name)

    # remove two simulations

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=2, ncols=2, figsize=[15,10],sharex=True, sharey=False, squeeze=True) #
    # fig, ax = plt.subplots(figsize=[10, 7])
    # plt.subplots_adjust(wspace=0.0, hspace=0.0)

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

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    model_order = []
    model_names_all = model_names['all_model']
    for model_name in model_names_all:
        if (model_name in model_out_list) and (model_name not in ['obs_cor','RF_eb']):
            model_order.append(model_name)

    if var_name in ['Qle','NEE']:
        model_order.append('obs')

    print('model_order',model_order)

    for i, model_out_name in enumerate(model_order):
        line_color = model_colors[model_out_name] #plt.cm.tab20(i / len(model_out_list))

        if smooth_type != 'no_soomth':
            dry_vals = smooth_vpd_series(var_dry[model_out_name+'_vals'], window_size, order, smooth_type)
            wet_vals = smooth_vpd_series(var_wet[model_out_name+'_vals'], window_size, order, smooth_type)
        else:
            dry_vals = var_dry[model_out_name+'_vals']
            wet_vals = var_wet[model_out_name+'_vals']

        dry_above_200 = (var_dry[model_out_name+'_vpd_num']>200)
        wet_above_200 = (var_wet[model_out_name+'_vpd_num']>200)

        var_dry_vpd_series = var_dry['vpd_series'][dry_above_200]
        dry_vals           = dry_vals[dry_above_200]
        dry_vals_bot       = var_dry[model_out_name+'_bot'][dry_above_200]
        dry_vals_top       = var_dry[model_out_name+'_top'][dry_above_200]

        var_wet_vpd_series = var_wet['vpd_series'][wet_above_200]
        wet_vals           = wet_vals[wet_above_200]
        wet_vals_bot       = var_wet[model_out_name+'_bot'][wet_above_200]
        wet_vals_top       = var_wet[model_out_name+'_top'][wet_above_200]


        if (var_name=='NEE'):
            # Unify NEE units : upwards CO2 movement is positive values
            if  model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA']:
                print("(var_name=='NEE') and (model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA'])")
                dry_vals       = dry_vals*(-1)
                dry_vals_bot   = dry_vals_bot*(-1)
                dry_vals_top   = dry_vals_top*(-1)
                wet_vals       = wet_vals*(-1)
                wet_vals_bot   = wet_vals_bot*(-1)
                wet_vals_top   = wet_vals_top*(-1)
            # Changes to downwards CO2 movement is positive values (now simaler to NEP, rather than NEE)
            dry_vals       = dry_vals*(-1)
            dry_vals_bot   = dry_vals_bot*(-1)
            dry_vals_top   = dry_vals_top*(-1)
            wet_vals       = wet_vals*(-1)
            wet_vals_bot   = wet_vals_bot*(-1)
            wet_vals_top   = wet_vals_top*(-1)

        # start plotting
        if np.sum(var_dry[model_out_name+'_vpd_num']-200) > 0:
            ax[0,0].plot(var_dry['vpd_series'], var_dry[model_out_name+'_vpd_num'], lw=2.0, color=line_color, alpha=0.7,label=model_out_name)
            ax[0,0].axhline(y=200, color='black', linestyle='-.', linewidth=1)
            plot = ax[1,0].plot(var_dry_vpd_series, dry_vals, lw=2.0,
                                color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()

            fill = ax[1,0].fill_between(var_dry_vpd_series,
                                        dry_vals_bot,
                                        dry_vals_top,
                                        color=line_color, edgecolor="none",
                                        alpha=0.05) #  .rolling(window=10).mean()

        if np.sum(var_wet[model_out_name+'_vpd_num']-200) > 0:
            ax[0,1].plot(var_wet['vpd_series'], var_wet[model_out_name+'_vpd_num'], lw=2.0, color=line_color, alpha=0.7,label=model_out_name)
            ax[0,1].axhline(y=200, color='black', linestyle='-.', linewidth=1)

            plot = ax[1,1].plot(var_wet_vpd_series, wet_vals, lw=2.0,
                                color=line_color, alpha=0.9, label=model_out_name) #edgecolor='none', c='red' .rolling(window=10).mean()
            fill = ax[1,1].fill_between(var_wet_vpd_series,
                                        wet_vals_bot,
                                        wet_vals_top,
                                        color=line_color, edgecolor="none",
                                        alpha=0.05) #  .rolling(window=10).mean()

        if turning_point['calc']:
            # print("turning_point['calc']",turning_point['calc'])
            ax[1,0].scatter(turning_points_dry[model_out_name][0], turning_points_dry[model_out_name][1], marker='o', color=line_color, s=10)
            ax[1,1].scatter(turning_points_wet[model_out_name][0], turning_points_wet[model_out_name][1], marker='o', color=line_color, s=10)

    ax[0,0].legend(fontsize=6, frameon=False, ncol=3)
    ax[0,1].legend(fontsize=6, frameon=False, ncol=3)

    if IGBP_type !=None:
        ax[1,0].text(0.12, 0.92, 'IGBP='+IGBP_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)
    if clim_type !=None:
        ax[1,0].text(0.12, 0.92, 'Clim_type='+clim_type, va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[1,0].text(0.12, 0.87, 'site_num='+str(var_dry['site_num'][0]), va='bottom', ha='center', rotation_mode='anchor',transform=ax[1,0].transAxes, fontsize=12)

    ax[0,0].set_xlim(0, 7.)
    ax[0,1].set_xlim(0, 7.)

    # ax[0,0].set_ylim(0, 10000)
    # ax[0,1].set_ylim(0, 10000)

    if var_name == 'TVeg':
        ax[1,0].set_ylim(-0.01,low_bound[1]/2)
        ax[1,1].set_ylim(-0.01,high_bound[1]/2)
    if var_name == 'Qle':
        ax[1,0].set_ylim(-10, low_bound[1]*500)
        ax[1,1].set_ylim(-10, high_bound[1]*500)
    if var_name == 'NEE':
        ax[1,0].set_ylim(-0.5, low_bound[1]*2)
        ax[1,1].set_ylim(-0.5, high_bound[1]*2)

    # ax[1].set_xlabel('VPD (kPa)', loc='center',size=14)# rotation=270,
    fig.savefig("./plots/"+var_name+'_VPD_all_sites'+file_name+'_'+message+'_'+smooth_type+'_coarse.png',bbox_inches='tight',dpi=300) # '_30percent'


def plot_model_bias_VPD(var_name, bin_by=None, day_time=False, summer_time=False,
                        window_size=11, order=3, smooth_type='S-G_filter', method='bin_by_vpd',
                        IGBP_type=None, clim_type=None, message=None, model_names=None,
                        turning_point={'calc':False,'method':'kneed'}):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=3, ncols=2, figsize=[15,10],sharex=False, sharey=False, squeeze=True) #
    # fig, ax = plt.subplots(figsize=[10, 7])
    # plt.subplots_adjust(wspace=0.0, hspace=0.0)

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

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # ============== read data ==============
    file_message = ''

    if day_time:
        file_message = file_message + '_daytime'

    if IGBP_type !=None:
        file_message = file_message + '_IGBP='+IGBP_type

    if clim_type !=None:
        file_message = file_message + '_clim='+clim_type

    file_names  = [f'./txt/{var_name}_VPD'+file_message+'_'+bin_by+'_0-0.2_'  +method+'_coarse.csv',
                   f'./txt/{var_name}_VPD'+file_message+'_'+bin_by+'_0.2-0.4_'+method+'_coarse.csv',
                   f'./txt/{var_name}_VPD'+file_message+'_'+bin_by+'_0.4-0.6_'+method+'_coarse.csv',
                   f'./txt/{var_name}_VPD'+file_message+'_'+bin_by+'_0.6-0.8_'+method+'_coarse.csv',
                   f'./txt/{var_name}_VPD'+file_message+'_'+bin_by+'_0.8-1.0_'+method+'_coarse.csv']

    # Read model classification
    empirical_model    = model_names['empirical_model']
    hydrological_model = model_names['hydrological_model']
    land_surface_model = model_names['land_surface_model']
    ecosystem_model    = model_names['ecosystem_model']

    empirical_num     = len(empirical_model)
    hydrological_num  = len(hydrological_model)
    land_surface_num  = len(land_surface_model)
    ecosystem_num     = len(ecosystem_model)
    print('empirical_num',empirical_num)

    for i, file_name in enumerate(file_names):
        print('Reading',file_name)
        # read data
        var               = pd.read_csv(file_name)

        # Initilization
        nvpd              = len(var['vpd_series'])
        var_empirical     = np.full((empirical_num, nvpd),np.nan)
        var_hydrological  = np.full((hydrological_num, nvpd),np.nan)
        var_land_surface  = np.full((land_surface_num, nvpd),np.nan)
        var_ecosystem     = np.full((ecosystem_num, nvpd),np.nan)

        empirical_model_list    = []
        hydrological_model_list = []
        land_surface_model_list = []
        ecosystem_model_list    = []

        emp_tot   = 0
        hydro_tot = 0
        lsm_tot   = 0
        eco_tot   = 0


        # get all model list in the file
        model_out_list = []
        for column_name in var.columns:
            if "_vals" in column_name:
                model_out_list.append(column_name.split("_vals")[0])

        print('model_out_list',model_out_list)

        # dividing
        for model_out_name in model_out_list:
            if np.sum(var[model_out_name+'_vpd_num']-200) > 0:
                print('Check point 1')
                # Change NEE to NEP
                if (var_name=='NEE') and (model_out_name not in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA']):
                    values = var[model_out_name+'_vals']*(-1)
                else:
                    values = var[model_out_name+'_vals']
                print('model_out_name',model_out_name)
                print('empirical_model',empirical_model)
                if model_out_name in empirical_model:
                    print('Check point 2')
                    empirical_model_list.append(model_out_name)
                    var_empirical[emp_tot,:] = values
                    emp_tot                  = emp_tot + 1
                elif model_out_name in hydrological_model:
                    print('Check point 3')
                    hydrological_model_list.append(model_out_name)
                    var_hydrological[hydro_tot,:] = values
                    hydro_tot                     = hydro_tot + 1
                elif (model_out_name in land_surface_model):
                    print('Check point 4')
                    land_surface_model_list.append(model_out_name)
                    var_land_surface[lsm_tot,:] = values
                    lsm_tot                     = lsm_tot + 1
                elif (model_out_name in ecosystem_model):
                    print('Check point 5')
                    ecosystem_model_list.append(model_out_name)
                    var_ecosystem[eco_tot,:] = values
                    eco_tot                  = eco_tot + 1

        print('emp_tot',emp_tot)
        print('hydro_tot',hydro_tot)
        print('lsm_tot',lsm_tot)
        print('eco_tot',eco_tot)

        # Calculate range
        var_empirical_top     = np.nanmax(var_empirical,axis=0)
        var_hydrological_top  = np.nanmax(var_hydrological,axis=0)
        var_land_surface_top  = np.nanmax(var_land_surface,axis=0)
        var_ecosystem_top     = np.nanmax(var_ecosystem,axis=0)

        var_empirical_bot     = np.nanmin(var_empirical,axis=0)
        var_hydrological_bot  = np.nanmin(var_hydrological,axis=0)
        var_land_surface_bot  = np.nanmin(var_land_surface,axis=0)
        var_ecosystem_bot     = np.nanmin(var_ecosystem,axis=0)

        # var_empirical_mean     = np.nanmean(var_empirical,axis=0)
        # var_hydrological_mean  = np.nanmean(var_hydrological,axis=0)
        # var_land_surface_mean  = np.nanmean(var_land_surface,axis=0)
        # var_ecosystem_mean     = np.nanmean(var_ecosystem,axis=0)

        var_empirical_mean     = np.nanmedian(var_empirical,axis=0)
        var_hydrological_mean  = np.nanmedian(var_hydrological,axis=0)
        var_land_surface_mean  = np.nanmedian(var_land_surface,axis=0)
        var_ecosystem_mean     = np.nanmedian(var_ecosystem,axis=0)

        # print("var['vpd_series']",var['vpd_series'])
        # print('var_empirical_mean',var_empirical_mean)
        row = int(i/2)
        col = i%2

        print('row',row,'col',col)

        plot = ax[row,col].plot(var['vpd_series'], var_empirical_mean, lw=2.0, color='red', alpha=0.9, label='empirical model')
        fill = ax[row,col].fill_between(var['vpd_series'],
                                        var_empirical_bot,
                                        var_empirical_top,
                                        color='red', edgecolor="none",
                                        alpha=0.1) #  .rolling(window=10).mean()

        plot = ax[row,col].plot(var['vpd_series'], var_hydrological_mean, lw=2.0, color='purple', alpha=0.9, label='hydrological model')
        fill = ax[row,col].fill_between(var['vpd_series'],
                                        var_hydrological_bot,
                                        var_hydrological_top,
                                        color='purple', edgecolor="none",
                                        alpha=0.1) #  .rolling(window=10).mean()

        plot = ax[row,col].plot(var['vpd_series'], var_land_surface_mean, lw=2.0, color='green', alpha=0.9, label='land surface model')
        fill = ax[row,col].fill_between(var['vpd_series'],
                                        var_land_surface_bot,
                                        var_land_surface_top,
                                        color='green', edgecolor="none",
                                        alpha=0.15) #  .rolling(window=10).mean()

        plot = ax[row,col].plot(var['vpd_series'], var_ecosystem_mean, lw=2.0, color='orange', alpha=0.9, label='ecosystem model')
        fill = ax[row,col].fill_between(var['vpd_series'],
                                        var_ecosystem_bot,
                                        var_ecosystem_top,
                                        color='yellow', edgecolor="none",
                                        alpha=0.3) #  .rolling(window=10).mean()
        if var_name == 'NEE':
            plot = ax[row,col].plot(var['vpd_series'], var['obs_vals']*(-1), lw=2.0, color='black', alpha=0.9, label='observation')
        elif var_name == 'Qle':
            plot = ax[row,col].plot(var['vpd_series'], var['obs_vals'], lw=2.0, color='black', alpha=0.9, label='observation')

        ax[row,col].text(0.2, 0.75, f'EF 0.{i*2}-0.{i*2+2}\nemp_tot={emp_tot} hydro_tot={hydro_tot}\nlsm_tot={lsm_tot} eco_tot={eco_tot}',
                        va='bottom', ha='center', rotation_mode='anchor',
                        transform=ax[row,col].transAxes, fontsize=9)

        ax[row,col].set_xlim(0, 7.)

        if var_name == 'TVeg':
            ax[row,col].set_ylim(-0.01, 0.35)
        if var_name == 'Qle':
            ax[row,col].set_ylim(-10, (i+1)*100)
        if var_name == 'NEE':
            ax[row,col].set_ylim(-0.2, 0.6)

        ax[0,0].legend(fontsize=6, frameon=False, ncol=3)
        fig.savefig("./plots/model_bias_VPD_"+var_name+file_message+"_coarse.png",bbox_inches='tight',dpi=300) # '_30percent'


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # Read site names, IGBP and clim
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    print(model_names)
    var_names      = ['NEE','Qle','TVeg']#
    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#

    day_time       = True
    method         = 'bin_by_vpd' #'GAM'
    # Smoothing setting

    window_size    = 11
    order          = 3
    smooth_type    = 'no_soomth' #'S-G_filter' #
    turning_point  =  {'calc':True, 'method':'piecewise'}
                      #{'calc':True, 'method':'cdf'}#{'calc':True, 'method':'kneed'}
    #
    # # for IGBP_type in IGBP_types:
    # # for clim_type in clim_types:
    message        = '0-0.4'
    low_bound      = [0,0.2]
    high_bound     = [0.2,0.4]
    for var_name in var_names:
        plot_var_VPD(var_name, bin_by=bin_by, low_bound=low_bound, high_bound=high_bound,
             day_time=day_time, window_size=window_size, order=order,
             smooth_type=smooth_type,message=message,model_names=model_names,turning_point=turning_point)#, IGBP_type=IGBP_type) #, clim_type=clim_type)
    
        gc.collect()
    
    message        = '0.4-0.8'
    low_bound      = [0.4,0.6]
    high_bound     = [0.6,0.8]
    for var_name in var_names:    
        plot_var_VPD(var_name, bin_by=bin_by, low_bound=low_bound, high_bound=high_bound,
             day_time=day_time, window_size=window_size, order=order,
             smooth_type=smooth_type,message=message,model_names=model_names,turning_point=turning_point)#, IGBP_type=IGBP_type) #, clim_type=clim_type)
    
        gc.collect()

    message        = 'dry-wet'
    low_bound      = [0,0.2]
    high_bound     = [0.8,1.0]
    for var_name in var_names:
        plot_var_VPD(var_name, bin_by=bin_by, low_bound=low_bound, high_bound=high_bound,
             day_time=day_time, window_size=window_size, order=order,
             smooth_type=smooth_type,message=message,model_names=model_names,turning_point=turning_point)#, IGBP_type=IGBP_type) #, clim_type=clim_type)

        gc.collect()

    # for var_name in var_names:
    #     plot_model_bias_VPD(var_name, bin_by=bin_by,
    #          day_time=day_time, window_size=window_size, order=order,
    #          smooth_type=smooth_type,message=message,model_names=model_names,)
    #          # turning_point=turning_point)#, IGBP_type=IGBP_type) #, clim_type=clim_type)
    #     gc.collect()
