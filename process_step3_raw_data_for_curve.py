'''
Select the raw data by
    var_name
    selected_by
    standardize
    day_time
    output_2d_grids_only
    time_scale
    country_code
    low_bound
    high_bound

Including:
    def write_raw_data_var_VPD

'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import gc
import sys
import glob
import copy
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

def write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=None, low_bound=30,
                            high_bound=70, day_time=False, summer_time=False, IGBP_type=None,
                            clim_type=None, energy_cor=False, VPD_num_threshold=None, select_site=None,
                            models_calc_LAI=None, veg_fraction=None, time_scale=None,
                            add_Rnet_caused_ratio=False, quality_ctrl=False,add_qc=False,
                            clarify_site={'opt':False,'remove_site':None}, standardize=None, add_LAI=False,
                            remove_strange_values=True, country_code=None, LAI_range=None, add_SMtopXm=None,
                            add_normalized_SMtopXm=None, hours_precip_free=None, output_2d_grids_only=True,
                            regional_sites=None, middle_day=False, VPD_sensitive=False,
                            Tair_constrain=None, add_Xday_mean_EF=None, data_selection=True,
                            add_aridity_index=True):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # ========== read the data ==========
    message = ''
    if time_scale!= None:
        message = message + "_"+time_scale
    if country_code != None:
        message = message+'_'+country_code

    # Read in data
    if 'TVeg' in var_name:
        add_units = '_Wm2'
    else:
        add_units = ''

    if time_scale == 'daily':
        var_output = pd.read_csv(f'./txt/process2_output/daily/{var_name}_all_sites{message}{add_units}.csv',na_values=[''])
    else:
        var_output = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites{add_units}.csv',na_values=[''])

    # Get model names
    if var_name == 'Gs':
        site_names, IGBP_types, clim_types, model_names = load_default_list()
        model_out_list = model_names['model_select_new']
    else:
        model_out_list = get_model_out_list(var_name)
    print('model_out_list',model_out_list)

    # Read LAI if needed
    if add_aridity_index:
        AI_input = pd.read_csv(f'./txt/process1_output/Aridity_index_all_sites.csv', na_values=[''])
        var_output['aridity_index'] = AI_input['aridity_index']

    # Read LAI if needed
    if LAI_range !=None or add_LAI:
        if time_scale == 'daily':
            LAI_input = pd.read_csv(f'./txt/process2_output/daily/LAI_all_sites_daily.csv', na_values=[''])
        else:
            LAI_input = pd.read_csv(f'./txt/process1_output/LAI_all_sites_parallel.csv', na_values=[''])

        for model_out_name in model_out_list:
            try:
                var_output[model_out_name+'_LAI'] = LAI_input[model_out_name+'_LAI']
            except:
                print(model_out_name,'use obs_LAI')
                var_output[model_out_name+'_LAI'] = LAI_input['obs_LAI']

    if selected_by == 'SM_per_all_models':

        site_names, IGBP_types, clim_types, model_names = load_default_list()
        model_names = model_names['model_select_new']

        SM_percentile_input = pd.read_csv(f'./txt/process2_output/SMtop{add_SMtopXm}m_percentile_all_sites.csv', na_values=[''])

        for model_out_name in model_names:
            if model_out_name == 'obs':
                head = ''
            else:
                head = 'model_'
            try:
                var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m_percentile'] = \
                    SM_percentile_input[head+model_out_name].values
            except:
                # Since obs and models without simulated SM are given model mean SM so code should not go here
                print(f'{model_out_name}_SMtop{add_SMtopXm}m_percentile, does not exist')
                var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m_percentile'] = np.nan

    if add_SMtopXm != None:

        SM_input = pd.read_csv('./txt/process1_output/SMtop'+str(add_SMtopXm)+'m_all_sites.csv', na_values=[''])

        for model_out_name in model_out_list:
            try:
                var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m'] = SM_input[model_out_name+'_SMtop'+str(add_SMtopXm)+'m']
            except:
                var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m'] = np.nan

        var_output['model_mean_SMtop'+str(add_SMtopXm)+'m'] = SM_input['model_mean_SMtop'+str(add_SMtopXm)+'m']

    if add_Xday_mean_EF != None:

        # Replace the hourly/half-hour EF by the Xday mean EF, which is calcuated based on daily Qle and Qh
        Xday_mean_EF_input = pd.read_csv(f'./txt/process2_output/EF_all_sites_{add_Xday_mean_EF}_day_mean.csv')

        for model_out_name in model_out_list:
            if model_out_name == 'obs':
                head = ''
            else:
                head = 'model_'

            try:
                var_output.loc[:,model_out_name+'_EF'] = Xday_mean_EF_input[head+model_out_name]
            except:
                var_output.loc[:,model_out_name+'_EF'] = np.nan

    if add_normalized_SMtopXm != None:

        normalized_SM_input = pd.read_csv('./txt/process1_output/normalized_SMtop'+str(add_normalized_SMtopXm)+'m_all_sites.csv', na_values=[''])

        for model_out_name in model_out_list:
            try:
                var_output[model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m'] = \
                    normalized_SM_input[model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m']
            except:
                var_output[model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m'] = np.nan

        var_output['model_mean_normalized_SMtop'+str(add_normalized_SMtopXm)+'m'] = \
            normalized_SM_input['model_mean_normalized_SMtop'+str(add_normalized_SMtopXm)+'m']

    if add_Rnet_caused_ratio:
        LH_ratio_input = pd.read_csv(f'./txt/process1_output/Rnet_caused_ratio_all_sites.csv', na_values=[''])

        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            try:
                # Calculate VPD account ratio
                VPD_caused_ratio_tmp  = 1. - LH_ratio_input[head+model_out_name]
                VPD_caused_ratio_tmp  = np.where(VPD_caused_ratio_tmp>=0,VPD_caused_ratio_tmp,np.nan)
                VPD_caused_ratio_tmp  = np.where(VPD_caused_ratio_tmp<=1,VPD_caused_ratio_tmp,np.nan)
                var_output[model_out_name+'_VPD_caused_ratio'] = VPD_caused_ratio_tmp
            except:
                var_output[model_out_name+'_VPD_caused_ratio'] = np.nan

    # !!!!!!!!!!!!!!!! edit here !!!!!!!!!!!!!!!!!!
    if data_selection:
        ds_input            = pd.read_csv(f'./txt/process2_output/data_selection_all_sites.csv', na_values=[''])
        data_selection_mask = ds_input['select_data'].values
        print('np.sum(data_selection_mask)',np.sum(data_selection_mask))
        var_output          = var_output[data_selection_mask]
        print('len(var_output)',len(var_output))

    if standardize=='STD_Gs_ref':
        Gs_ref_input = pd.read_csv(f'./txt/process1_output/Gs_ref_all_sites_filtered.csv', na_values=[''])
        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            try:
                var_output[model_out_name+'_Gs_ref'] = Gs_ref_input[head+model_out_name]
            except:
                var_output[model_out_name+'_Gs_ref'] = np.nan

    # total site number
    site_num    = len(np.unique(var_output["site_name"]))
    # # print('Point 1, site_num=',site_num)
    # # print('Finish reading csv file')/

    # ========== select data ==========

    # whether only considers the sites with energy budget corrected fluxs
    if var_name in ['Qle','Qh'] and energy_cor:
        check_obs_cor = var_output['obs_cor']
        check_obs_cor.to_csv(f'./txt/check_obs_cor.csv')

        cor_notNan_mask = ~ np.isnan(var_output['obs_cor'])
        var_output      = var_output[cor_notNan_mask]

    if select_site!=None:
        # Use radiation as threshold
        site_mask   = (var_output['site_name'] == select_site)
        var_output  = var_output[site_mask]
        site_num    = len(np.unique(var_output["site_name"]))

    # To select the sites for the region
    if regional_sites != None:

        # print('clarifying sites')
        length         = len(var_output)
        site_keep_mask = np.full(length,False)

        for site_keep in regional_sites['sites']:
            site_keep_mask = np.where(var_output['site_name'] == site_keep, True, site_keep_mask)

        var_output  = var_output[site_keep_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print(regional_sites['name'],":",np.unique(var_output["site_name"]))

    # whether only considers one type of IGBP
    if IGBP_type!=None:

        # if more than one IGBP_type
        if isinstance(IGBP_type, list):
            IGBP_type_array = np.array(IGBP_type)
            IGBP_mask   = np.isin(var_output['IGBP_type'], IGBP_type_array)
        elif isinstance(IGBP_type, str):
            IGBP_mask   = (var_output['IGBP_type'] == IGBP_type)
        else:
            print("IGBP_type is of an unknown type")

        # # print('np.any(IGBP_mask)', np.any(IGBP_mask))
        var_output  = var_output[IGBP_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        # # print('Point 4, site_num=',site_num)

    # whether only considers one type of climate type
    if clim_type!=None:
        clim_mask   = (var_output['climate_type'] == clim_type)
        # # print('np.any(clim_mask)', np.any(clim_mask))
        var_output  = var_output[clim_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        # # print('Point 5, site_num=',site_num)

    # select data with required vegetation fraction
    if veg_fraction !=None:
        veg_frac_mask= (var_output['NoahMPv401_greenness']>=veg_fraction[0]) & (var_output['NoahMPv401_greenness']<=veg_fraction[1])
        var_output   = var_output[veg_frac_mask]
        site_num     = len(np.unique(var_output["site_name"]))
        # # print('Point 6, site_num=',site_num)

    # select data with required LAI values : Check the code!!!
    if LAI_range !=None:

        for i, model_out_name in enumerate(model_out_list):
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            try:
                LAI_mask  = (var_output[model_out_name+'_LAI'] >= LAI_range[0]) & (var_output[model_out_name+'_LAI'] < LAI_range[1])
            except:
                LAI_mask  = (var_output['obs_LAI'] >= LAI_range[0]) & (var_output['obs_LAI'] < LAI_range[1])
            var_output[head+model_out_name]  = np.where(LAI_mask, var_output[head+model_out_name], np.nan)
        site_num     = len(np.unique(var_output["site_name"]))

    if var_name == 'NEE':
        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            if (model_out_name == 'GFDL') | (model_out_name == 'NoahMPv401') | (model_out_name =='STEMMUS-SCOPE') | (model_out_name =='ACASA'):
                # print('model_out_name=',model_out_name,'in GFDL, NoahMPv401, STEMMUS-SCOPE,ACASA')
                values = var_output[head+model_out_name]
            else:
                values = var_output[head+model_out_name]*(-1)
            var_output[head+model_out_name] = values

    # remove outlier data
    if remove_strange_values:
        for model_out_name in model_out_list:
            # print('Checking strange values in', model_out_name)
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            print('before remove_strange_values', model_out_name, np.sum(~np.isnan(var_output[head+model_out_name])))
            var_output[head+model_out_name] = np.where(np.any([var_output[head+model_out_name]>2000.,
                                                       var_output[head+model_out_name]<0.],axis=0),
                                                       np.nan, var_output[head+model_out_name])
            print('after remove_strange_values', model_out_name, np.sum(~np.isnan(var_output[head+model_out_name])))

    # ========== Standardize data ==========
    if standardize == 'STD_annual_obs':

        # print('standardized by annual obs mean')

        # Get all sites left
        sites_left    = np.unique(var_output["site_name"])

        # Initialize the variable of the mean of the left observation for each left site
        site_obs_mean = {}

        # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
        for site in sites_left:

            # Get the mask of this site
            site_mask_tmp       = (var_output['site_name'] == site)

            # Mask the dataframe to get slide of the dataframe for this site
            var_tmp             = var_output[site_mask_tmp]

            # Calculate site obs mean
            site_obs_mean[site] = np.nanmean(var_tmp['obs'])

            # Standardize the different model's values by the obs mean for this site
            for i, model_out_name in enumerate(model_out_list):
                if 'obs' in model_out_name:
                    head = ''
                else:
                    head = 'model_'
                var_output.loc[site_mask_tmp, head+model_out_name] = var_tmp[head+model_out_name]/site_obs_mean[site]

        # print('site_obs_mean',site_obs_mean)
    # elif standardize == 'STD_annual_model':
    #
    #     # print('standardized by annual model mean')
    #
    #     # Get all sites left
    #     sites_left    = np.unique(var_output["site_name"])
    #
    #     for site in sites_left:
    #
    #         # Get the mask of this site
    #         site_mask_tmp   = (var_output['site_name'] == site)
    #
    #         # Mask the dataframe to get slide of the dataframe for this site
    #         var_tmp         = var_output[site_mask_tmp]
    #
    #         # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
    #         for i, model_out_name in enumerate(model_out_list):
    #             if 'obs' in model_out_name:
    #                 head = ''
    #             else:
    #                 head = 'model_'
    #
    #             # Calculate site obs mean
    #             site_model_mean = np.nanmean(var_tmp[head+model_out_name])
    #
    #             # Standardize the different model's values by the obs mean for this site
    #             var_output.loc[site_mask_tmp, head+model_out_name] = var_tmp[head+model_out_name]/site_model_mean

    elif standardize == 'STD_SMtopXm' and add_SMtopXm:

        print("in standardize == 'STD_SMtopXm' and add_SMtopXm")

        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            print('before',var_output[head+model_out_name])

            if model_out_name+'_SMtop'+str(add_SMtopXm)+'m' in var_output.columns:
                if np.all(np.isnan(var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m'])):
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']
                else:
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m']
            else:
                var_output[head+model_out_name][:] = \
                    var_output[head+model_out_name]/var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']

            ### for testing whether excludes LAI < 0.01 time step matters for the curves
            ### Testing result: exclude time steps that LAI < 0.01, will change the shape
            ### of curves in a few models. So, don't exclude these values
            # if add_LAI:
            #     # check whether the model has calculated LAI
            #     use_model_LAI = False
            #     for model_calc_LAI in models_calc_LAI:
            #         if model_out_name == model_calc_LAI:
            #             use_model_LAI = True
            #             break
            #
            #     if use_model_LAI:
            #         modeled_LAI = var_output[model_out_name+'_LAI']
            #         var_output[head+model_out_name][:] = np.where( np.all([ modeled_LAI >= 0.01, np.isnan(modeled_LAI) == False], axis=0),
            #                                                     var_output[head+model_out_name], np.nan )
            #     else:
            #         var_output[head+model_out_name][:] = np.where( var_output['obs_LAI'] >= 0.01,
            #                                                     var_output[head+model_out_name],
            #                                                     np.nan )
            print('after',var_output[head+model_out_name])

            if np.any(var_output[head+model_out_name] == np.inf):
                print(model_out_name)

                site_with_inf        = np.where(var_output[head+model_out_name] == np.inf, var_output['site_name'], 'NaN')
                site_with_inf_unique = np.unique(site_with_inf)
                for site in site_with_inf_unique:
                    print(site,'has', np.sum(site_with_inf == site), 'data points with inf')

                raise ValueError("Inf exists")
    elif standardize == 'STD_normalized_SMtopXm' and add_normalized_SMtopXm:

        print("in standardize == 'STD_normalized_SMtopXm' and add_normalized_SMtopXm")

        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            print('before',var_output[head+model_out_name])

            if model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m' in var_output.columns:
                if np.all(np.isnan(var_output[model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m'])):
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output['model_mean_normalized_SMtop'+str(add_normalized_SMtopXm)+'m']
                else:
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output[model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m']
            else:
                var_output[head+model_out_name][:] = \
                    var_output[head+model_out_name]/var_output['model_mean_normalized_SMtop'+str(add_normalized_SMtopXm)+'m']

            print('after',var_output[head+model_out_name])

            if np.any(var_output[head+model_out_name] == np.inf):
                print(model_out_name)

                site_with_inf        = np.where(var_output[head+model_out_name] == np.inf, var_output['site_name'], 'NaN')
                site_with_inf_unique = np.unique(site_with_inf)
                for site in site_with_inf_unique:
                    print(site,'has', np.sum(site_with_inf == site), 'data points with inf')

                raise ValueError("Inf exists")
    elif standardize == 'STD_SWdown':

        print("in standardize == 'STD_SWdown'")

        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            # standardize by SWdown
            var_output[head+model_out_name][:] = var_output[head+model_out_name]/var_output['obs_SWdown']
    elif standardize == 'STD_SWdown_SMtopXm' and add_SMtopXm:

        print("in standardize == 'STD_SWdown_SMtopXm' and add_SMtopXm")

        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            print('before',var_output[head+model_out_name])

            if model_out_name+'_SMtop'+str(add_SMtopXm)+'m' in var_output.columns:
                if np.all(np.isnan(var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m'])):
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']
                else:
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m']
            else:
                var_output[head+model_out_name][:] = \
                    var_output[head+model_out_name]/var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']

            print('after',var_output[head+model_out_name])

            if np.any(var_output[head+model_out_name] == np.inf):
                print(model_out_name)

                site_with_inf        = np.where(var_output[head+model_out_name] == np.inf, var_output['site_name'], 'NaN')
                site_with_inf_unique = np.unique(site_with_inf)
                for site in site_with_inf_unique:
                    print(site,'has', np.sum(site_with_inf == site), 'data points with inf')

                raise ValueError("Inf exists")

            # standardize by SWdown
            var_output[head+model_out_name][:] = var_output[head+model_out_name]/var_output['obs_SWdown']
    elif standardize == 'STD_LAI_SMtopXm':

        '''
        standardlize by LAI should only use on TVeg
        '''

        # print('standardized by SMtop1m and then LAI')
        for model_out_name in model_out_list:

            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            if model_out_name+'_SMtop'+str(add_SMtopXm)+'m' in var_output.columns:
                if np.all(np.isnan(var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m'])):
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']
                else:
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m']
            else:
                var_output[head+model_out_name][:] = \
                    var_output[head+model_out_name]/var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']

            if np.any(var_output[head+model_out_name] == np.inf):
                print(model_out_name)
                print(var_output[head+model_out_name])
                print(var_output['model_mean_SMtop'+str(add_SMtopXm)+'m'])
                raise ValueError("Inf exists")

            # check whether the model has calculated LAI
            # use_model_LAI = False
            # for model_calc_LAI in models_calc_LAI:
            #     if model_out_name == model_calc_LAI:
            #         use_model_LAI = True
            #         break

            # if use_model_LAI:
            modeled_LAI = var_output[model_out_name+'_LAI']
            var_output[head+model_out_name][:] = np.where( np.all([ modeled_LAI >= 0.01, np.isnan(modeled_LAI) == False], axis=0) ,
                                                        var_output[head+model_out_name]/modeled_LAI,
                                                        np.nan ) # since removing LAI <0.01 will change curve shape, change the
                                                                    # restriction to LAI should be >=0.001
            # else:
            #     var_output[head+model_out_name][:] = np.where( var_output['obs_LAI'] >= 0.01 ,
            #                                                 var_output[head+model_out_name]/var_output['obs_LAI'],
            #                                                 np.nan )
    elif standardize == 'STD_LAI_normalized_SMtopXm':

        '''
        standardlize by LAI should only use on TVeg
        '''

        # print('standardized by SMtop1m and then LAI')
        for model_out_name in model_out_list:

            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            if model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m' in var_output.columns:
                if np.all(np.isnan(var_output[model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m'])):
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output['model_mean_normalized_SMtop'+str(add_normalized_SMtopXm)+'m']
                else:
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output[model_out_name+'_normalized_SMtop'+str(add_normalized_SMtopXm)+'m']
            else:
                var_output[head+model_out_name][:] = \
                    var_output[head+model_out_name]/var_output['model_mean_normalized_SMtop'+str(add_normalized_SMtopXm)+'m']

            if np.any(var_output[head+model_out_name] == np.inf):
                print(model_out_name)
                print(var_output[head+model_out_name])
                print(var_output['model_mean_normalized_SMtop'+str(add_normalized_SMtopXm)+'m'])
                raise ValueError("Inf exists")

            # if use_model_LAI:
            modeled_LAI = var_output[model_out_name+'_LAI']
            var_output[head+model_out_name][:] = np.where( np.all([ modeled_LAI >= 0.01, np.isnan(modeled_LAI) == False], axis=0) ,
                                                        var_output[head+model_out_name]/modeled_LAI,
                                                        np.nan ) # since removing LAI <0.01 will change curve shape, change the
                                                                    # restriction to LAI should be >=0.001
    elif standardize == 'STD_SWdown_LAI_SMtopXm':

        '''
        standardlize by LAI should only use on TVeg
        '''

        # print('standardized by SMtop1m and then LAI')
        for model_out_name in model_out_list:

            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            # standardize by SMtop1m
            if model_out_name+'_SMtop'+str(add_SMtopXm)+'m' in var_output.columns:
                if np.all(np.isnan(var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m'])):
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']
                else:
                    var_output[head+model_out_name][:] = \
                        var_output[head+model_out_name]/var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m']
            else:
                var_output[head+model_out_name][:] = \
                    var_output[head+model_out_name]/var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']

            if np.any(var_output[head+model_out_name] == np.inf):
                print(model_out_name)
                print(var_output[head+model_out_name])
                print(var_output['model_mean_SMtop'+str(add_SMtopXm)+'m'])
                raise ValueError("Inf exists")

            # standardize by LAI
            # check whether the model has calculated LAI
            # use_model_LAI = False
            # for model_calc_LAI in models_calc_LAI:
            #     if model_out_name == model_calc_LAI:
            #         use_model_LAI = True
            #         break

            # if use_model_LAI:
            modeled_LAI = var_output[model_out_name+'_LAI']
            var_output[head+model_out_name][:] = np.where( np.all([ modeled_LAI >= 0.01, np.isnan(modeled_LAI) == False], axis=0) ,
                                                        var_output[head+model_out_name]/modeled_LAI,
                                                        np.nan ) # since removing LAI <0.01 will change curve shape, change the
                                                                    # restriction to LAI should be >=0.001
            # else:
            #     var_output[head+model_out_name][:] = np.where( var_output['obs_LAI'] >= 0.01,
            #                                                 var_output[head+model_out_name]/var_output['obs_LAI'],
            #                                                 np.nan )

            # standardize by SWdown
            var_output[head+model_out_name][:] = var_output[head+model_out_name]/var_output['obs_SWdown']
    elif standardize == 'STD_LAI':

        # print('standardized by LAI')

        # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
        for i, model_out_name in enumerate(model_out_list):

            # check whether the model has calculated LAI
            use_model_LAI = False
            for model_calc_LAI in models_calc_LAI:
                if model_out_name == model_calc_LAI:
                    use_model_LAI = True
                    break

            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            if use_model_LAI:
                modeled_LAI = var_output[model_out_name+'_LAI']
                var_output[head+model_out_name] = np.where( np.all([ modeled_LAI != 0, np.isnan(modeled_LAI) == False], axis=0) ,
                                                            var_output[head+model_out_name]/modeled_LAI,
                                                            np.nan )
            else:
                var_output[head+model_out_name] = np.where( var_output['obs_LAI'] != 0 ,
                                                            var_output[head+model_out_name]/var_output['obs_LAI'],
                                                            np.nan )
    elif standardize == 'STD_montly_obs':

        # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
        for site in sites_left:
            for mth in np.arange(1,13,1):

                # print('site=',site,'mth=',mth)

                # Get the mask of this site
                site_mask_tmp       = (var_output['site_name'] == site) & (var_output['month'] == mth)
                site_mask_month     = (var_monthly_input['month'] == mth) & (var_monthly_input['site_name'] == site)

                # print('Point 8, np.any(site_mask_tmp)',np.any(site_mask_tmp))
                # print('!!! Point 9, np.any(site_mask_month)',np.any(site_mask_month))

                # Mask the dataframe to get slide of the dataframe for this site
                var_tmp             = var_output[site_mask_tmp]
                # print('var_tmp', var_tmp)

                # get site monthly mean
                site_obs_month      = var_monthly_input.loc[site_mask_month]['obs'].values
                # print('site_obs_month', site_obs_month)

                # Standardize the different model's values by the obs mean for this site
                for i, model_out_name in enumerate(model_out_list):
                    if 'obs' in model_out_name:
                        head = ''
                    else:
                        head = 'model_'
                    var_output.loc[site_mask_tmp, head+model_out_name] = var_tmp[head+model_out_name]/site_obs_month
        # print('var_output',var_output)
    elif standardize == 'STD_month_model':

        # print('standardized by monthly model mean')

        # read monthly mean
        var_monthly_input = pd.read_csv(f'./txt/process2_output/monthly/{var_name}_all_sites_monthly.csv',na_values=[''])

        # Get all sites left
        sites_left        = np.unique(var_output["site_name"])

        # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
        for site in sites_left:
            for mth in np.arange(1,13,1):

                # Get the mask of this site
                site_mask_tmp       = (var_output['site_name'] == site) & (var_output['month'] == mth)
                site_mask_month     = (var_monthly_input['month'] == mth) & (var_monthly_input['site_name'] == site)

                # Mask the dataframe to get slide of the dataframe for this site
                var_tmp             = var_output[site_mask_tmp]

                # Standardize the different model's values by the obs mean for this site
                for i, model_out_name in enumerate(model_out_list):
                    if 'obs' in model_out_name:
                        head = ''
                    else:
                        head = 'model_'

                    # get site monthly mean
                    site_model_month      = var_monthly_input.loc[site_mask_month][head+model_out_name].values
                    var_output.loc[site_mask_tmp, head+model_out_name] = var_tmp[head+model_out_name]/site_model_month
    elif standardize == 'STD_daily_obs':

        # print('standardized by daily obs mean')
        obs_daily = var_output['obs']

        # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
        for i, model_out_name in enumerate(model_out_list):

            # check whether the model has calculated LAI
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            var_output[head+model_out_name] = np.where( obs_daily != 0 ,var_output[head+model_out_name]/obs_daily,
                                                        np.nan )
    elif standardize == 'Gs_ref_input':

        print("in standardize == 'Gs_ref_input'")

        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            # standardize by SWdown
            var_output[head+model_out_name][:] = var_output[head+model_out_name]/var_output[head+model_out_name+'_Gs_ref']

    # if add_Rnet_caused_ratio:
    #
    #     # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
    #     for i, model_out_name in enumerate(model_out_list):
    #
    #         # check whether the model has calculated LAI
    #         if 'obs' in model_out_name:
    #             head = ''
    #         else:
    #             head = 'model_'
    #
    #         var_output[model_out_name+'_VPD_caused'] = var_output[head+model_out_name]*var_output[model_out_name+'_VPD_caused_ratio']
    #         print(model_out_name,'no nan LH is',np.sum(~np.isnan(var_output[head+model_out_name])),
    #               'no nan LH_VPD_caused is',np.sum(~np.isnan(var_output[model_out_name+'_VPD_caused_ratio'])))

    if output_2d_grids_only:
        mask_VPD_tmp       = (var_output['VPD'] != np.nan)
        var_output_raw_data= var_output[mask_VPD_tmp]

        # save data
        if var_name == 'NEE':
            var_name = 'NEP'

        folder_name, file_message = decide_filename(day_time=day_time, summer_time=summer_time, energy_cor=energy_cor,
                                                    IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale,
                                                    standardize=standardize, country_code=country_code, LAI_range=LAI_range,
                                                    veg_fraction=veg_fraction, clarify_site=clarify_site , data_selection=data_selection)
        if select_site != None:
            var_output_raw_data.to_csv(f'./txt/process3_output/2d_grid/raw_data_{var_name}_VPD{file_message}_{select_site}.csv')
        else:
            var_output_raw_data.to_csv(f'./txt/process3_output/2d_grid/raw_data_{var_name}_VPD'+file_message+'.csv')
        return

    # ========== Divide dry and wet periods ==========

    # Calculate EF thresholds
    if selected_by == 'SM_per_all_models':

        '''
        Be carefull: this script select the time steps that all 13 models with SM simulations are within the
        required ranges, rather than the models with the variable (such as TVeg, non TVeg have fewer models).
        That's the reason use model_names rather than model_out_list
        '''

        var_output_dry = copy.deepcopy(var_output)
        var_output_wet = copy.deepcopy(var_output)

        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        models_with_SM_per = ['CABLE', 'CABLE-POP-CN',
                              'CHTESSEL_Ref_exp1', 'CLM5a',
                              'GFDL', 'JULES_GL9', 'JULES_GL9_withLAI',
                              'MATSIRO', 'MuSICA', 'NoahMPv401',
                              'ORC2_r6593', 'ORC3_r8120', 'STEMMUS-SCOPE']

        SM_per_model_names = ['CABLE_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'CABLE-POP-CN_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'CHTESSEL_Ref_exp1_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'CLM5a_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'GFDL_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'JULES_GL9_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'JULES_GL9_withLAI_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'MATSIRO_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'MuSICA_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'NoahMPv401_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'ORC2_r6593_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'ORC3_r8120_SMtop'+str(add_SMtopXm)+'m_percentile',
                              'STEMMUS-SCOPE_SMtop'+str(add_SMtopXm)+'m_percentile']

        for i, model_out_name in enumerate(models_with_SM_per):
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            SM_per_name = model_out_name+'_SMtop'+str(add_SMtopXm)+'m_percentile'

            try:
                if len(low_bound)>1 and len(high_bound)>1:
                    dry_mask  = (var_output[SM_per_name] > low_bound[0]) & (var_output[SM_per_name] < low_bound[1])
                    wet_mask  = (var_output[SM_per_name] > high_bound[0]) & (var_output[SM_per_name] < high_bound[1])
                elif len(low_bound)==1 and len(high_bound)==1:
                    dry_mask  = (var_output[SM_per_name] < low_bound)
                    wet_mask  = (var_output[SM_per_name] > high_bound)
                else:
                    sys.exit('len(low_bound)=',len(low_bound),'len(high_bound)=',len(high_bound))

                var_output_dry[SM_per_name] = np.where(dry_mask, var_output[SM_per_name], np.nan)
                var_output_wet[SM_per_name] = np.where(wet_mask, var_output[SM_per_name], np.nan)
            except:
                print(SM_per_name,'does not exist')

        # Print the expected column names
        print("Expected columns (SM_per_model_names):")
        print(SM_per_model_names)

        # Print the actual columns in the DataFrame
        print("Columns in var_output_dry:")
        print(var_output_dry.columns)
        # return

        # Mask out SM is inconsistent
        var_output_dry[SM_per_model_names] = var_output_dry[SM_per_model_names].where(~var_output_dry[SM_per_model_names].isna().any(axis=1), other=np.nan)
        var_output_wet[SM_per_model_names] = var_output_wet[SM_per_model_names].where(~var_output_wet[SM_per_model_names].isna().any(axis=1), other=np.nan)
        print('CABLE SM per non nan:', np.sum(~np.isnan(var_output_dry['CABLE_SMtop'+str(add_SMtopXm)+'m_percentile'].values)),
              'GFDL SM per non nan:', np.sum(~np.isnan(var_output_dry['GFDL_SMtop'+str(add_SMtopXm)+'m_percentile'].values)))

        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            var_output_dry.loc[:,head+model_out_name] = np.where(~np.isnan(var_output_dry['CABLE_SMtop'+str(add_SMtopXm)+'m_percentile'].values),
                                                           var_output_dry[head+model_out_name], np.nan)
            var_output_wet.loc[:,head+model_out_name] = np.where(~np.isnan(var_output_wet['CABLE_SMtop'+str(add_SMtopXm)+'m_percentile'].values),
                                                           var_output_wet[head+model_out_name], np.nan)

        # ======== add 2 July 2024: keep time steps that only all models have data ========
        model_unify_names = []
        for i, model_out_name in enumerate(model_out_list):
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            model_unify_names.append(head+model_out_name)
        # print('model_unify_names', model_unify_names)

        # Mask out SM is inconsistent
        print('before dry',np.sum(~np.isnan(var_output_dry['model_CABLE'])))
        print('before wet',np.sum(~np.isnan(var_output_wet['model_CABLE'])))
        var_output_dry[model_unify_names] = var_output_dry[model_unify_names].where(~var_output_dry[model_unify_names].isna().any(axis=1), other=np.nan)
        var_output_wet[model_unify_names] = var_output_wet[model_unify_names].where(~var_output_wet[model_unify_names].isna().any(axis=1), other=np.nan)
        print('after dry',np.sum(~np.isnan(var_output_dry['model_CABLE'])))
        print('after wet',np.sum(~np.isnan(var_output_wet['model_CABLE'])))

    elif selected_by == 'EF_obs':

        var_output_dry = copy.deepcopy(var_output)
        var_output_wet = copy.deepcopy(var_output)

        # print( 'Check point 6, np.any(~np.isnan(var_output_dry["model_CABLE"]))=',
        #       np.any(~np.isnan(var_output_dry["model_CABLE"])) )

        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        for i, model_out_name in enumerate(model_out_list):

            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            if len(low_bound)>1 and len(high_bound)>1:
                dry_mask  = (var_output['obs_EF'] > low_bound[0]) & (var_output['obs_EF'] < low_bound[1])
                wet_mask  = (var_output['obs_EF'] > high_bound[0]) & (var_output['obs_EF'] < high_bound[1])
            elif len(low_bound)==1 and len(high_bound)==1:
                dry_mask  = (var_output['obs_EF'] < low_bound)
                wet_mask  = (var_output['obs_EF'] > high_bound)
            else:
                sys.exit('len(low_bound)=',len(low_bound),'len(high_bound)=',len(high_bound))

            var_output_dry[head+model_out_name] = np.where(dry_mask, var_output[head+model_out_name], np.nan)
            var_output_wet[head+model_out_name] = np.where(wet_mask, var_output[head+model_out_name], np.nan)

    elif selected_by == 'EF_model':

        var_output_dry = copy.deepcopy(var_output)
        var_output_wet = copy.deepcopy(var_output)

        # print( 'Check point 6, np.any(~np.isnan(var_output_dry["model_CABLE"]))=',
        #       np.any(~np.isnan(var_output_dry["model_CABLE"])) )

        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        for i, model_out_name in enumerate(model_out_list):
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            if model_out_name == 'obs_cor':
                # Use Qle_obs and Qh_obs calculated EF to bin obs_cor, this method may introduce
                # some bias, but keep it for now
                EF_var_name = 'obs_EF'
            else:
                EF_var_name = model_out_name+'_EF'

            if len(low_bound)>1 and len(high_bound)>1:
                dry_mask  = (var_output[EF_var_name] > low_bound[0]) & (var_output[EF_var_name] < low_bound[1])
                wet_mask  = (var_output[EF_var_name] > high_bound[0]) & (var_output[EF_var_name] < high_bound[1])
            elif len(low_bound)==1 and len(high_bound)==1:
                dry_mask  = (var_output[EF_var_name] < low_bound)
                wet_mask  = (var_output[EF_var_name] > high_bound)
            else:
                sys.exit('len(low_bound)=',len(low_bound),'len(high_bound)=',len(high_bound))

            var_output_dry[head+model_out_name] = np.where(dry_mask, var_output[head+model_out_name], np.nan)
            var_output_wet[head+model_out_name] = np.where(wet_mask, var_output[head+model_out_name], np.nan)

        # print( 'Check point 7, np.any(~np.isnan(var_output_dry["model_CABLE"]))=',
        #       np.any(~np.isnan(var_output_dry["model_CABLE"])) )

    elif 'EF_' in selected_by:

        bin_by_model = selected_by[3:]
        print('bin by model', bin_by_model)

        # bin by any model
        var_output_dry = copy.deepcopy(var_output)
        var_output_wet = copy.deepcopy(var_output)
        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        for i, model_out_name in enumerate(model_out_list):

            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            if len(low_bound)>1 and len(high_bound)>1:
                dry_mask  = (var_output[bin_by_model+'_EF'] > low_bound[0]) & (var_output[bin_by_model+'_EF'] < low_bound[1])
                wet_mask  = (var_output[bin_by_model+'_EF'] > high_bound[0]) & (var_output[bin_by_model+'_EF'] < high_bound[1])
            elif len(low_bound)==1 and len(high_bound)==1:
                dry_mask  = (var_output[bin_by_model+'_EF'] < low_bound)
                wet_mask  = (var_output[bin_by_model+'_EF'] > high_bound)
            else:
                sys.exit('len(low_bound)=',len(low_bound),'len(high_bound)=',len(high_bound))

            var_output_dry[head+model_out_name] = np.where(dry_mask, var_output[head+model_out_name], np.nan)
            var_output_wet[head+model_out_name] = np.where(wet_mask, var_output[head+model_out_name], np.nan)

    elif selected_by == 'SMtopXm':

        var_output_dry = copy.deepcopy(var_output)
        var_output_wet = copy.deepcopy(var_output)

        for i, model_out_name in enumerate(model_out_list):

            # get SMtop1m
            if model_out_name+'_SMtop'+str(add_SMtopXm)+'m' in var_output.columns:
                if np.all(np.isnan(var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m'])):
                    SMtopXm_tmp = var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']
                else:
                    SMtopXm_tmp = var_output[model_out_name+'_SMtop'+str(add_SMtopXm)+'m']
            else:
                SMtopXm_tmp = var_output['model_mean_SMtop'+str(add_SMtopXm)+'m']

            # get dry and wet thresholds
            mask_temp   = (~ np.isnan(SMtopXm_tmp))
            bot_dry_tmp = np.percentile(SMtopXm_tmp[mask_temp], low_bound[0])
            top_dry_tmp = np.percentile(SMtopXm_tmp[mask_temp], high_bound[0])

            bot_wet_tmp = np.percentile(SMtopXm_tmp[mask_temp], low_bound[1])
            top_wet_tmp = np.percentile(SMtopXm_tmp[mask_temp], high_bound[1])

            SM_dry_mask  = (SMtopXm_tmp >= bot_dry_tmp) & (SMtopXm_tmp < top_dry_tmp)
            SM_wet_mask  = (SMtopXm_tmp >= bot_wet_tmp) & (SMtopXm_tmp < top_wet_tmp)

            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            var_output_dry[head+model_out_name] = np.where(SM_dry_mask, var_output[head+model_out_name], np.nan)
            var_output_wet[head+model_out_name] = np.where(SM_wet_mask, var_output[head+model_out_name], np.nan)

    # print('Finish dividing dry and wet periods')

    # ========== Test standardize by only by the data used in the curves rather than all available time steps
    if add_Rnet_caused_ratio:

        # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
        for i, model_out_name in enumerate(model_out_list):

            # check whether the model has calculated LAI
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            var_output_dry[model_out_name+'_VPD_caused'] = var_output_dry[head+model_out_name]*var_output_dry[model_out_name+'_VPD_caused_ratio']
            var_output_wet[model_out_name+'_VPD_caused'] = var_output_wet[head+model_out_name]*var_output_wet[model_out_name+'_VPD_caused_ratio']

            print(model_out_name,'in dry file, no nan LH is',np.sum(~np.isnan(var_output_dry[head+model_out_name])),
                  'in dry file, no nan LH_VPD_caused is',np.sum(~np.isnan(var_output_dry[model_out_name+'_VPD_caused'])))
            print(model_out_name,'in wet file, no nan LH is',np.sum(~np.isnan(var_output_wet[head+model_out_name])),
                  'in wet file, no nan LH_VPD_caused is',np.sum(~np.isnan(var_output_wet[model_out_name+'_VPD_caused'])))

    if standardize == 'STD_annual_model':

        # print('standardized by annual model mean')

        # Get all sites left
        sites_left_dry    = np.unique(var_output_dry["site_name"])
        sites_left_wet    = np.unique(var_output_wet["site_name"])

        for site in sites_left_dry:

            # Get the mask of this site
            site_mask_tmp   = (var_output_dry['site_name'] == site)

            # Mask the dataframe to get slide of the dataframe for this site
            var_tmp         = var_output_dry[site_mask_tmp]

            # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
            for i, model_out_name in enumerate(model_out_list):
                if 'obs' in model_out_name:
                    head = ''
                else:
                    head = 'model_'

                # Calculate site obs mean
                site_model_mean = np.nanmean(var_tmp[head+model_out_name])

                # Standardize the different model's values by the obs mean for this site
                var_output_dry.loc[site_mask_tmp, head+model_out_name] = var_tmp[head+model_out_name]/site_model_mean

        for site in sites_left_wet:

            # Get the mask of this site
            site_mask_tmp   = (var_output_wet['site_name'] == site)

            # Mask the dataframe to get slide of the dataframe for this site
            var_tmp         = var_output_wet[site_mask_tmp]

            # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
            for i, model_out_name in enumerate(model_out_list):
                if 'obs' in model_out_name:
                    head = ''
                else:
                    head = 'model_'

                # Calculate site obs mean
                site_model_mean = np.nanmean(var_tmp[head+model_out_name])

                # Standardize the different model's values by the obs mean for this site
                var_output_wet.loc[site_mask_tmp, head+model_out_name] = var_tmp[head+model_out_name]/site_model_mean

        if add_Rnet_caused_ratio:

            # Get all sites left
            sites_left_dry    = np.unique(var_output_dry["site_name"])
            sites_left_wet    = np.unique(var_output_wet["site_name"])

            for site in sites_left_dry:

                # Get the mask of this site
                site_mask_tmp   = (var_output_dry['site_name'] == site)

                # Mask the dataframe to get slide of the dataframe for this site
                var_tmp         = var_output_dry[site_mask_tmp]

                # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
                for i, model_out_name in enumerate(model_out_list):

                    # Calculate site obs mean
                    site_model_mean = np.nanmean(var_tmp[model_out_name+'_VPD_caused'])

                    # Standardize the different model's values by the obs mean for this site
                    var_output_dry.loc[site_mask_tmp, model_out_name+'_VPD_caused'] = var_tmp[model_out_name+'_VPD_caused']/site_model_mean

            for site in sites_left_wet:

                # Get the mask of this site
                site_mask_tmp   = (var_output_wet['site_name'] == site)

                # Mask the dataframe to get slide of the dataframe for this site
                var_tmp         = var_output_wet[site_mask_tmp]

                # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
                for i, model_out_name in enumerate(model_out_list):

                    # Calculate site obs mean
                    site_model_mean = np.nanmean(var_tmp[model_out_name+'_VPD_caused'])

                    # Standardize the different model's values by the obs mean for this site
                    var_output_wet.loc[site_mask_tmp, model_out_name+'_VPD_caused'] = var_tmp[model_out_name+'_VPD_caused']/site_model_mean

    # ========== Save curves ==========
    if var_name == 'NEE':
        var_name = 'NEP'

    mask_VPD_dry       = (var_output_dry['VPD'] != np.nan)
    mask_VPD_wet       = (var_output_wet['VPD'] != np.nan)

    # var_select_dry_tmp = var_output_dry[['VPD','obs','model_CABLE','model_STEMMUS-SCOPE']]
    var_select_dry     = var_output_dry[mask_VPD_dry]
    var_select_wet     = var_output_wet[mask_VPD_wet]

    # file name
    folder_name, file_message1 = decide_filename(day_time=day_time, summer_time=summer_time, energy_cor=energy_cor,
                                        IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale,
                                        standardize=standardize, country_code=country_code, selected_by=selected_by,
                                        bounds=low_bound, veg_fraction=veg_fraction, LAI_range=LAI_range,
                                        clarify_site=clarify_site, regional_sites=regional_sites, add_Xday_mean_EF=add_Xday_mean_EF, data_selection=data_selection)

    folder_name, file_message2 = decide_filename(day_time=day_time, summer_time=summer_time, energy_cor=energy_cor,
                                        IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale,
                                        standardize=standardize, country_code=country_code, selected_by=selected_by,
                                        bounds=high_bound, veg_fraction=veg_fraction, LAI_range=LAI_range,
                                        clarify_site=clarify_site, regional_sites=regional_sites, add_Xday_mean_EF=add_Xday_mean_EF, data_selection=data_selection)

    if Tair_constrain != None:
        message_midday = '_Tair_'+str(Tair_constrain)
    elif VPD_sensitive:
        message_midday = '_VPD_sensitive_periods'
    elif middle_day:
        message_midday = '_midday'
    else:
        message_midday = ''

    if select_site != None:
        var_select_dry.to_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD{file_message1}_{select_site}{message_midday}.csv')
        var_select_wet.to_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD{file_message2}_{select_site}{message_midday}.csv')
    else:
        var_select_dry.to_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD{file_message1}{message_midday}.csv')
        var_select_wet.to_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD{file_message2}{message_midday}.csv')

    return

# if __name__ == "__main__":

#     # Path of PLUMBER 2 dataset
#     PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
#     site_names, IGBP_types, clim_types, model_names = load_default_list()

#     var_name       = 'Qle'      #'nonTVeg' #'TVeg'
#     selected_by    = 'EF_model' #'EF_model' #'EF_obs'
#     standardize    = None          # None
#                                    # 'STD_LAI'
#                                    # 'STD_annual_obs'
#                                    # 'STD_monthly_obs'
#                                    # 'STD_monthly_model'
#                                    # 'STD_daily_obs'
#     add_SMtopXm            = None
#     add_normalized_SMtopXm = None
#     time_scale             = 'hourly' # 'daily'
#                                       # 'hourly'

#     day_time          = False
#     if time_scale == 'hourly':
#         day_time      = True

#     clarify_site      = {'opt': True,
#                          'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6', # rainfall problems
#                                          'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1',
#                                          'AU-Wrr','CN-Din','US-WCr','ZM-Mon' # models miss the simulations of them
#                                          ]}

#     models_calc_LAI   = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

#     energy_cor        = False
#     if var_name == 'NEE':
#         energy_cor = False

#     country_code   = None #'AU'
#     if country_code != None:
#         site_names = load_sites_in_country_list(country_code)

#     # IGBP_types     = ['GRA', 'DBF', 'ENF', 'EBF']

#     # whether only provide 2d_grid csv data and stop the script
#     output_2d_grids_only = False
#     region_name          = 'global'
#     if region_name == 'global':
#         region = {'name':'global', 'lat':None, 'lon':None}
#         regional_sites   = None
#     elif region_name == 'east_AU':
#         region = {'name':'east_AU', 'lat':[-44.5,-10], 'lon':[129,155]}
#         regional_sites   = get_regional_site_list(region)
#     elif region_name == 'west_EU':
#         region = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
#         regional_sites   = get_regional_site_list(region)
#     elif region_name == 'north_Am':
#         region = {'name':'north_Am', 'lat':[25,52], 'lon':[-125,-65]}
#         regional_sites   = get_regional_site_list(region)

#     # # ========= clim_types ==========
#     # low_bound      = [0,0.2] #30
#     # high_bound     = [0.8,1.] #70
#     # LAI_range      = None   # [0,1.]
#     #                         # [1.,2.]
#     #                         # [2.,4.]
#     #                         # [4.,10.]
#     # veg_fraction   = None
#     # for clim_type in clim_types:
#     #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale,
#     #                     country_code=country_code, LAI_range=LAI_range,  clim_type=clim_type,
#     #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
#     #     gc.collect()

#     # # ========= veg type ==========
#     # low_bound      = [0,0.2] #30
#     # high_bound     = [0.8,1.] #70
#     # LAI_range      = None   # [0,1.]
#     #                         # [1.,2.]
#     #                         # [2.,4.]
#     #                         # [4.,10.]
#     # veg_fraction   = None
#     # for IGBP_type in IGBP_types:
#     #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale,
#     #                     country_code=country_code, LAI_range=LAI_range,  IGBP_type=IGBP_type,
#     #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
#     #     gc.collect()

#     # ========= wet & dry ==========
#     if 1:
#         var_name       = 'Qle'
#         veg_fraction   = None # [0,0.3] # low veg fraction
#         LAI_range      = None # [0.,1.]
#         IGBP_type      = None
#         middle_day     = False #True

#         selected_by    = 'EF_model' #'EF_obs' #'EF_CABLE' #
#         add_LAI        = True
#         add_qc         = True
#         add_Xday_mean_EF= None
#         quality_ctrl   = True
#         add_Rnet_caused_ratio = True
#         Tair_constrain = None
#         VPD_sensitive  = False
#         add_normalized_SMtopXm = None

#         # # SMtop percentile
#         # low_bounds      = [[0,15], [30,50], [70,90],]#,  [0.2,0.4],[0.6,0.8]]
#         # high_bounds     = [[15,30],[50,70],[90,100],]#,[0.4,0.6],[0.8,1.0]]

#         # EF
#         # low_bounds      = [[0.0,0.2],[0.2,0.4],[0.6,0.8]]
#         # high_bounds     = [[0.8,1.0],[0.4,0.6],[0.8,1.0]]
#         # selected_by     = 'EF_CABLE'
#         # LAI_ranges      = [[0, 1.], [1.,2.], [2.,4.], [4.,10.]]
#         IGBP_types      = ['GRA', 'OSH', 'SAV', 'WSA', 'CSH', 'DBF', 'ENF', 'EBF', 'MF', 'WET', 'CRO']

#         low_bounds      = [[0.0,0.3]]
#         high_bounds     = [[0.7,1.0]]

#         # Tair_constrain = 15
#         add_Xday_mean_EF= '1'

#         for i in np.arange(1):

#             add_SMtopXm = '0.3'
#             low_bound   = low_bounds[i]
#             high_bound  = high_bounds[i]

#             # var_name    = 'Qle'
#             selected_by = 'EF_model'
#             # standardize = 'STD_normalized_SMtopXm'
#             # add_normalized_SMtopXm = '0.3'

#             var_name    = 'Qle'
#             # for IGBP_type in IGBP_types:
#             # for LAI_range in LAI_ranges:
#             write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                             high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                             models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#                             add_SMtopXm=add_SMtopXm, add_normalized_SMtopXm=add_normalized_SMtopXm,
#                             add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,  IGBP_type=IGBP_type,
#                             country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#                             energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#                             middle_day=middle_day, VPD_sensitive=VPD_sensitive, Tair_constrain=Tair_constrain,
#                             add_Xday_mean_EF=add_Xday_mean_EF)

#                 # var_name    = 'TVeg'
#                 # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                 #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                 #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#                 #                 add_SMtopXm=add_SMtopXm, add_normalized_SMtopXm=add_normalized_SMtopXm,
#                 #                 add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,  IGBP_type=IGBP_type,
#                 #                 country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#                 #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#                 #                 middle_day=middle_day, VPD_sensitive=VPD_sensitive, Tair_constrain=Tair_constrain)

#                 # var_name    = 'nonTVeg'
#                 # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                 #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                 #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#                 #                 add_SMtopXm=add_SMtopXm, add_normalized_SMtopXm=add_normalized_SMtopXm,
#                 #                 add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,  IGBP_type=IGBP_type,
#                 #                 country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, #
#                 #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#                 #                 middle_day=middle_day, VPD_sensitive=VPD_sensitive, Tair_constrain=Tair_constrain)

#             # var_name    = 'Qle'
#             # selected_by = 'EF_obs'
#             # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#             #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#             #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#             #                 add_SMtopXm=add_SMtopXm,add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,
#             #                 country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#             #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#             #                 middle_day=middle_day)


#             # LAI_ranges      = [3.,15.]
#             # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#             #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#             #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#             #                 add_SMtopXm=add_SMtopXm,add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,
#             #                 country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#             #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#             #                 middle_day=middle_day)
#             #
#             # selected_by    = 'EF_model' #'EF_obs'
#             # # LAI_ranges      = [0.,0.1]
#             # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#             #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#             #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#             #                 add_SMtopXm=add_SMtopXm,add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,
#             #                 country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#             #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#             #                 middle_day=middle_day)

#             # LAI_ranges      = [3.,15.]
#             # standardize    = 'STD_LAI'
#             # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#             #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#             #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#             #                 add_SMtopXm=add_SMtopXm,add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,
#             #                 country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#             #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#             #                 middle_day=middle_day)

#         # low_bounds      = [[0,0.8]]#,  [0.2,0.4],[0.6,0.8]]
#         # high_bounds     = [[0.2,1.0]]#,[0.4,0.6],[0.8,1.0]]

#         # for i in np.arange(1):

#         #     # selected_by = 'SMtop1m'
#         #     low_bound  = low_bounds[i]
#         #     high_bound = high_bounds[i]

#         #     # standardize    = 'STD_SMtop1m'
#         #     # selected_by    = 'EF_obs' #'EF_model' #'EF_obs'
#         #     LAI_ranges      = [0.,0.1]
#         #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#         #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#         #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#         #                     add_SMtopXm=add_SMtopXm,add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,
#         #                     country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#         #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#         #                     middle_day=middle_day)

#         #     standardize    = 'STD_LAI'
#         #     LAI_ranges      = [3.,15.]
#         #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#         #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#         #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#         #                     add_SMtopXm=add_SMtopXm,add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,
#         #                     country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#         #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#         #                     middle_day=middle_day)
#         #     #
#         #     selected_by    = 'EF_model' #'EF_obs'
#         #     LAI_ranges      = [0.,0.1]
#         #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#         #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#         #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#         #                     add_SMtopXm=add_SMtopXm,add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,
#         #                     country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#         #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#         #                     middle_day=middle_day)

#         #     standardize    = 'STD_LAI'
#         #     LAI_ranges      = [3.,15.]
#         #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#         #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#         #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#         #                     add_SMtopXm=add_SMtopXm,add_LAI=add_LAI, add_Rnet_caused_ratio=add_Rnet_caused_ratio,
#         #                     country_code=country_code, LAI_range=LAI_range, add_qc=add_qc, quality_ctrl=quality_ctrl, # IGBP_type=IGBP_type,
#         #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites,
#         #                     middle_day=middle_day)
#     # ========= standardize ==========
#     if 0:

#         var_name       = 'TVeg'
#         standardize    = 'STD_SWdown_SMtopXm'
#         low_bound      = [0,0.2]  #30
#         high_bound     = [0.8,1.] #70
#         veg_fraction   = None # [0,0.3] # low veg fraction
#         LAI_range      = None # [0.,1.]
#         add_SMtopXm    = True
#         add_LAI        = True
#         write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                         high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                         models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#                         country_code=country_code, LAI_range=LAI_range, add_SMtopXm=add_SMtopXm, add_LAI=add_LAI,# IGBP_type=IGBP_type,
#                         energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#         low_bound      = [0.2,0.4] #30
#         high_bound     = [0.6,0.8] #70
#         veg_fraction   = None # [0,0.3] # low veg fraction
#         LAI_range      = None # [0.,1.]
#         add_SMtopXm    = True
#         add_LAI        = True
#         write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                         high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                         models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#                         country_code=country_code, LAI_range=LAI_range, add_SMtopXm=add_SMtopXm, add_LAI=add_LAI,# IGBP_type=IGBP_type,
#                         energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#         low_bound      = [0.4,0.6] #30
#         high_bound     = [0.6,0.8] #70
#         veg_fraction   = None # [0,0.3] # low veg fraction
#         LAI_range      = None # [0.,1.]
#         add_SMtopXm    = True
#         add_LAI        = True
#         write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                         high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                         models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#                         country_code=country_code, LAI_range=LAI_range, add_SMtopXm=add_SMtopXm, add_LAI=add_LAI,# IGBP_type=IGBP_type,
#                         energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#     # ========= add_LAI ==========
#     if 0:
#         var_name       = 'Qle'
#         low_bound      = [0,0.2] #30
#         high_bound     = [0.8,1.] #70
#         veg_fraction   = None # [0,0.3] # low veg fraction
#         LAI_range      = None # [0.,1.]
#         add_SMtopXm    = True
#         add_LAI        = True
#         write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                         high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                         models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#                         country_code=country_code, LAI_range=LAI_range, add_SMtopXm=add_SMtopXm, add_LAI=add_LAI,
#                         energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#     # # # ================ Select sites ====================
#     # # low_bound      = [0,0.2] #30
#     # # high_bound     = [0.8,1.] #70
#     # # veg_fraction   = None # [0,0.3] # low veg fraction
#     # # LAI_range      = None # [0.,1.]
#     #
#     # # for select_site in site_names:
#     # #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     # #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     # #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#     # #                 country_code=country_code, LAI_range=LAI_range, select_site=select_site,
#     # #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
#     #
#     #
#     # # ========= 0.2 < EF < 0.8 ==========
#     if 0:
#         veg_fraction   = None
#         IGBP_type      = None
#         add_SMtopXm    = True
#         add_LAI        = True
#         LAI_range      = None   # [0,1.]
#                                 # [1.,2.]
#                                 # [2.,4.]
#                                 # [4.,10.]
#         low_bound      = [0.2,0.4] #30
#         high_bound     = [0.6,0.8] #70
#         # standardize    = 'STD_Gs_ref'
#         write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                         high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                         models_calc_LAI=models_calc_LAI, time_scale=time_scale,add_SMtopXm=add_SMtopXm,add_LAI=add_LAI,
#                         country_code=country_code, LAI_range=LAI_range,  # IGBP_type=IGBP_type,
#                         energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#         low_bound      = [0.2,0.4] #30
#         high_bound     = [0.4,0.6] #70

#         write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#                         high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#                         models_calc_LAI=models_calc_LAI, time_scale=time_scale,add_SMtopXm=add_SMtopXm,add_LAI=add_LAI,
#                         country_code=country_code, LAI_range=LAI_range,  # IGBP_type=IGBP_type,
#                         energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#         gc.collect()

#     # # ========= LAI ==========
#     # low_bound      = [0,0.2] #30
#     # high_bound     = [0.8,1.] #70
#     # IGBP_type      = None
#     # veg_fraction   = None

#     # LAI_range      = [0.,1.]
#     # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#     #                 country_code=country_code, LAI_range=LAI_range, # IGBP_type=IGBP_type,
#     #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#     # LAI_range      = [1.,2.]
#     # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#     #                 country_code=country_code, LAI_range=LAI_range, # IGBP_type=IGBP_type,
#     #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#     # LAI_range      = [2.,4.]
#     # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#     #                 country_code=country_code, LAI_range=LAI_range, # IGBP_type=IGBP_type,
#     #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#     # LAI_range      = [4.,10.]
#     # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#     #                 country_code=country_code, LAI_range=LAI_range, # IGBP_type=IGBP_type,
#     #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#     # # ========= veg type ==========
#     # low_bound      = [0,0.2] #30
#     # high_bound     = [0.8,1.] #70
#     # LAI_range      = None   # [0,1.]
#     #                         # [1.,2.]
#     #                         # [2.,4.]
#     #                         # [4.,10.]
#     # veg_fraction   = None
#     # for IGBP_type in IGBP_types:
#     #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale,
#     #                     country_code=country_code, LAI_range=LAI_range,  IGBP_type=IGBP_type,
#     #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
#     #     gc.collect()

#     # ========= veg fraction ==========
#     # low_bound      = [0,0.2] #30
#     # high_bound     = [0.8,1.] #70
#     # veg_fraction   = [0,0.3] # low veg fraction
#     # LAI_range      = None
#     # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#     #                 country_code=country_code,  # IGBP_type=IGBP_type,
#     #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

#     # low_bound      = [0,0.2] #30
#     # high_bound     = [0.8,1.] #70
#     # veg_fraction   = [0.7,1.] # high veg fraction
#     # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
#     #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
#     #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
#     #                 country_code=country_code,  # IGBP_type=IGBP_type,
#     #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
