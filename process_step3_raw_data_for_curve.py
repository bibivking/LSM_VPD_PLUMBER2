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
                            clarify_site={'opt':False,'remove_site':None}, standardize=None,
                            remove_strange_values=True, country_code=None, LAI_range=None, add_SMtop1m=False,
                            hours_precip_free=None, output_2d_grids_only=True, regional_sites=None):

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
        # var_output = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites'+message+'.csv',na_values=[''])
        var_output = pd.read_csv(f'./txt/process1_output/{var_name}_all_sites{add_units}.csv',na_values=[''])

    print('Reading ',f'./txt/process1_output/{var_name}_all_sites{add_units}.csv')

    # print( 'Check point 1, np.any(~np.isnan(var_output["model_CABLE"]))=',
    #       np.any(~np.isnan(var_output["model_CABLE"])) )

    # Get model names
    if var_name == 'Gs':
        site_names, IGBP_types, clim_types, model_names = load_default_list()
        model_out_list = model_names['model_select_new']
    else:
        model_out_list = get_model_out_list(var_name)

    # Read LAI if needed
    if LAI_range !=None:
        if time_scale == 'daily':
            LAI_input = pd.read_csv(f'./txt/process2_output/daily/LAI_all_sites_daily.csv', na_values=[''])
        else:
            LAI_input = pd.read_csv(f'./txt/process1_output/LAI_all_sites_parallel.csv', na_values=[''])

        for model_out_name in model_out_list:
            try:
                var_output[model_out_name+'_LAI'] = LAI_input[model_out_name+'_LAI']
            except:
                var_output[model_out_name+'_LAI'] = LAI_input['obs_LAI']

    if add_SMtop1m:
        if time_scale == 'daily':
            SM_input = pd.read_csv(f'./txt/process1_output/SMtop1m_all_sites.csv', na_values=[''])

        for model_out_name in model_out_list:
            try:
                var_output[model_out_name+'_SMtop1m'] = LAI_input[model_out_name+'_SMtop1m']
            except:
                var_output[model_out_name+'_SMtop1m'] = np.nan

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

    if remove_strange_values:
        for model_out_name in model_out_list:
            # print('Checking strange values in', model_out_name)
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            var_output[head+model_out_name] = np.where(np.any([var_output[head+model_out_name]>999.,
                                                       var_output[head+model_out_name]<-999.],axis=0),
                                                       np.nan, var_output[head+model_out_name])
            # print('np.any(np.isnan(var_output[head+model_out_name]))',np.any(np.isnan(var_output[head+model_out_name])))

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

    # whether only considers day time
    if day_time:
        # Use hours as threshold
        # day_mask    = (var_output['hour'] >= 9) & (var_output['hour'] <= 16)

        # Use radiation as threshold
        day_mask    = (var_output['obs_SWdown'] >= 5)

        var_output  = var_output[day_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        # # print('Point 2, site_num=',site_num)

        check_site = var_output[ var_output['site_name']=='CA-NS1']

    # whether only considers summers
    if summer_time:
        summer_mask = (var_output['month'] > 11) | (var_output['month']< 3)
        # # print('np.any(summer_mask)', np.any(summer_mask))
        var_output  = var_output[summer_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        # # print('Point 3, site_num=',site_num)

    # whether only considers one type of IGBP
    if IGBP_type!=None:
        IGBP_mask   = (var_output['IGBP_type'] == IGBP_type)
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

    # whether only considers observation without precipitation in hours_precip_free hours
    if hours_precip_free!=None:
        rain_mask   = (var_output['hrs_after_precip'] > hours_precip_free)
        var_output  = var_output[rain_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        # # print('Point 7, site_num=',site_num)

    # To exclude the sites have rainfall input problems
    if clarify_site['opt']:
        # print('clarifying sites')
        length    = len(var_output)
        site_mask = np.full(length,True)

        for site_remove in clarify_site['remove_site']:
            site_mask = np.where(var_output['site_name'] == site_remove, False, site_mask)
        # # print('np.all(site_mask)',np.all(site_mask))

        # site_mask = ~(var_output['site_name'] in clarify_site['remove_site'])
        var_output  = var_output[site_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        # # print('Point 8, site_num=',site_num)

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

    elif standardize == 'STD_SMtop1m' and add_SMtop1m:

        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'

            if model_out_name+'_SMtop1m' in var_tmp.columns:
                if np.all(np.isnan(var_tmp[model_out_name+'_SMtop1m'])):
                    var_output.loc[head+model_out_name] = \
                        var_tmp[head+model_out_name]/var_tmp['model_mean_SMtop1m']
                else:
                    var_output.loc[head+model_out_name] = \
                        var_tmp[head+model_out_name]/var_tmp[model_out_name+'_SMtop1m']
            else:
                var_output.loc[head+model_out_name] = \
                    var_tmp[head+model_out_name]/var_tmp['model_mean_SMtop1m']

    elif standardize == 'STD_LAI ':

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

    if output_2d_grids_only:
        mask_VPD_tmp       = (var_output['VPD'] != np.nan)
        var_output_raw_data= var_output[mask_VPD_tmp]

        # save data
        if var_name == 'NEE':
            var_name = 'NEP'

        folder_name, file_message = decide_filename(day_time=day_time, summer_time=summer_time, energy_cor=energy_cor,
                                                    IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale,
                                                    standardize=standardize, country_code=country_code, LAI_range=LAI_range,
                                                    veg_fraction=veg_fraction, clarify_site=clarify_site)
        if select_site != None:
            var_output_raw_data.to_csv(f'./txt/process3_output/2d_grid/raw_data_{var_name}_VPD{file_message}_{select_site}.csv')
        else:
            var_output_raw_data.to_csv(f'./txt/process3_output/2d_grid/raw_data_{var_name}_VPD'+file_message+'.csv')
        return

    # ========== Divide dry and wet periods ==========

    # Calculate EF thresholds
    if selected_by == 'EF_obs':

        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        EF_notNan_mask = ~ np.isnan(var_output['obs_EF'])
        var_output     = var_output[EF_notNan_mask]

        # # print('np.any(EF_notNan_mask)', np.any(EF_notNan_mask))

        # Select EF<low_bound and EF>high_bound for each site to make sure
        # that every site can contribute to the final VPD lines
        for site_name in site_names:

            # select data for this site
            site_mask       = (var_output['site_name'] == site_name)

            # print('In bin by EF, site_name=', site_name, 'np.any(site_mask)',np.any(site_mask))

            # calculate EF thresholds for this site
            if len(low_bound)>1 and len(high_bound)>1:
                try:
                    bin_dry_low  = np.percentile(var_output[site_mask]['obs_EF'], low_bound[0])
                    bin_dry_high = np.percentile(var_output[site_mask]['obs_EF'], low_bound[1])
                    bin_wet_low  = np.percentile(var_output[site_mask]['obs_EF'], high_bound[0])
                    bin_wet_high = np.percentile(var_output[site_mask]['obs_EF'], high_bound[1])
                except:
                    bin_dry_low  = np.nan
                    bin_dry_high = np.nan
                    bin_wet_low  = np.nan
                    bin_wet_high = np.nan
                # make the mask based on EF thresholds and append it to a full-site long logic array
                try:
                    dry_mask = dry_mask.append((var_output[site_mask]['obs_EF'] > bin_dry_low)
                                             & (var_output[site_mask]['obs_EF'] < bin_dry_high))
                    wet_mask = wet_mask.append((var_output[site_mask]['obs_EF'] > bin_wet_low)
                                             & (var_output[site_mask]['obs_EF'] < bin_wet_high))
                except:
                    dry_mask = (var_output[site_mask]['obs_EF'] > bin_dry_low) & (var_output[site_mask]['obs_EF'] < bin_dry_high)
                    wet_mask = (var_output[site_mask]['obs_EF'] > bin_wet_low) & (var_output[site_mask]['obs_EF'] < bin_wet_high)
            elif len(low_bound)==1 and len(high_bound)==1:
                try:
                    bin_dry     = np.percentile(var_output[site_mask]['obs_EF'], low_bound)
                    bin_wet     = np.percentile(var_output[site_mask]['obs_EF'], high_bound)
                except:
                    bin_dry     = np.nan
                    bin_wet     = np.nan

                # make the mask based on EF thresholds and append it to a full-site long logic array
                try:
                    dry_mask = dry_mask.append(var_output[site_mask]['obs_EF'] < bin_dry)
                    wet_mask = wet_mask.append(var_output[site_mask]['obs_EF'] > bin_wet)
                except:
                    dry_mask = (var_output[site_mask]['obs_EF'] < bin_dry)
                    wet_mask = (var_output[site_mask]['obs_EF'] > bin_wet)
            else:
                sys.exit('len(low_bound)=',len(low_bound),'len(high_bound)=',len(high_bound))

        # Mask out the time steps beyond the EF thresholds
        var_output_dry = var_output[dry_mask]
        var_output_wet = var_output[wet_mask]

        # free memory
        EF_notNan_mask=None

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

    # print('Finish dividing dry and wet periods')

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
                                        clarify_site=clarify_site, regional_sites=regional_sites)

    folder_name, file_message2 = decide_filename(day_time=day_time, summer_time=summer_time, energy_cor=energy_cor,
                                        IGBP_type=IGBP_type, clim_type=clim_type, time_scale=time_scale,
                                        standardize=standardize, country_code=country_code, selected_by=selected_by,
                                        bounds=high_bound, veg_fraction=veg_fraction, LAI_range=LAI_range,
                                        clarify_site=clarify_site, regional_sites=regional_sites)
    if select_site != None:
        var_select_dry.to_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD{file_message1}_{select_site}.csv')
        var_select_wet.to_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD{file_message2}_{select_site}.csv')
    else:
        var_select_dry.to_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD{file_message1}.csv')
        var_select_wet.to_csv(f'./txt/process3_output/curves/raw_data_{var_name}_VPD{file_message2}.csv')

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    var_name       = 'Qle'      #'nonTVeg' #'TVeg'
    selected_by    = 'EF_model' #'EF_model' #'EF_obs'
    standardize    = None          # None
                                   # 'STD_LAI'
                                   # 'STD_annual_obs'
                                   # 'STD_monthly_obs'
                                   # 'STD_monthly_model'
                                   # 'STD_daily_obs'

    time_scale        = 'hourly' # 'daily'
                                 # 'hourly'

    day_time          = False
    if time_scale == 'hourly':
        day_time      = True

    clarify_site      = {'opt': True,
                         'remove_site': ['AU-Rig','AU-Rob','AU-Whr','AU-Ync','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                         'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}

    models_calc_LAI   = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

    energy_cor        = False
    if var_name == 'NEE':
        energy_cor = False

    country_code   = None #'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # IGBP_types     = ['GRA', 'DBF', 'ENF', 'EBF']

    # whether only provide 2d_grid csv data and stop the script
    output_2d_grids_only = False
    region_name          = 'global'
    if region_name == 'global':
        region = {'name':'global', 'lat':None, 'lon':None}
        regional_sites   = None
    elif region_name == 'east_AU':
        region = {'name':'east_AU', 'lat':[-44.5,-10], 'lon':[129,155]}
        regional_sites   = get_regional_site_list(region)
    elif region_name == 'west_EU':
        region = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
        regional_sites   = get_regional_site_list(region)
    elif region_name == 'north_Am':
        region = {'name':'north_Am', 'lat':[25,52], 'lon':[-125,-65]}
        regional_sites   = get_regional_site_list(region)

    # # ========= veg type ==========
    # low_bound      = [0,0.2] #30
    # high_bound     = [0.8,1.] #70
    # LAI_range      = None   # [0,1.]
    #                         # [1.,2.]
    #                         # [2.,4.]
    #                         # [4.,10.]
    # veg_fraction   = None
    # for clim_type in clim_types:
    #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale,
    #                     country_code=country_code, LAI_range=LAI_range,  clim_type=clim_type,
    #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
    #     gc.collect()

    # # ========= veg type ==========
    # low_bound      = [0,0.2] #30
    # high_bound     = [0.8,1.] #70
    # LAI_range      = None   # [0,1.]
    #                         # [1.,2.]
    #                         # [2.,4.]
    #                         # [4.,10.]
    # veg_fraction   = None
    # for IGBP_type in IGBP_types:
    #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale,
    #                     country_code=country_code, LAI_range=LAI_range,  IGBP_type=IGBP_type,
    #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
    #     gc.collect()

    # ========= wet & dry ==========
    var_name       = 'Qle'
    standardize    = 'STD_SMtop1m'
    low_bound      = [0,0.2] #30
    high_bound     = [0.8,1.] #70
    veg_fraction   = None # [0,0.3] # low veg fraction
    LAI_range      = None # [0.,1.]
    add_SMtop1m    = True
    write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
                    high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
                    models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
                    country_code=country_code, LAI_range=LAI_range, add_SMtop1m=add_SMtop1m, # IGBP_type=IGBP_type,
                    energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

    # # # ================ Select sites ====================
    # # low_bound      = [0,0.2] #30
    # # high_bound     = [0.8,1.] #70
    # # veg_fraction   = None # [0,0.3] # low veg fraction
    # # LAI_range      = None # [0.,1.]
    #
    # # for select_site in site_names:
    # #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    # #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    # #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
    # #                 country_code=country_code, LAI_range=LAI_range, select_site=select_site,
    # #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
    #
    #
    # # ========= 0.2 < EF < 0.8 ==========
    # low_bound      = [0.2,0.4] #30
    # high_bound     = [0.6,0.8] #70
    # LAI_range      = None   # [0,1.]
    #                         # [1.,2.]
    #                         # [2.,4.]
    #                         # [4.,10.]
    # veg_fraction   = None
    # IGBP_type      = None
    # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale,
    #                 country_code=country_code, LAI_range=LAI_range,  # IGBP_type=IGBP_type,
    #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
    #
    # low_bound      = [0.2,0.4] #30
    # high_bound     = [0.4,0.6] #70
    # LAI_range      = None   # [0,1.]
    #                         # [1.,2.]
    #                         # [2.,4.]
    #                         # [4.,10.]
    #
    # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale,
    #                 country_code=country_code, LAI_range=LAI_range,  # IGBP_type=IGBP_type,
    #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
    #
    # gc.collect()

    # # ========= LAI ==========
    # low_bound      = [0,0.2] #30
    # high_bound     = [0.8,1.] #70
    # IGBP_type      = None
    # veg_fraction   = None

    # LAI_range      = [0.,1.]
    # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
    #                 country_code=country_code, LAI_range=LAI_range, # IGBP_type=IGBP_type,
    #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

    # LAI_range      = [1.,2.]
    # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
    #                 country_code=country_code, LAI_range=LAI_range, # IGBP_type=IGBP_type,
    #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

    # LAI_range      = [2.,4.]
    # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
    #                 country_code=country_code, LAI_range=LAI_range, # IGBP_type=IGBP_type,
    #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

    # LAI_range      = [4.,10.]
    # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
    #                 country_code=country_code, LAI_range=LAI_range, # IGBP_type=IGBP_type,
    #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

    # # ========= veg type ==========
    # low_bound      = [0,0.2] #30
    # high_bound     = [0.8,1.] #70
    # LAI_range      = None   # [0,1.]
    #                         # [1.,2.]
    #                         # [2.,4.]
    #                         # [4.,10.]
    # veg_fraction   = None
    # for IGBP_type in IGBP_types:
    #     write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                     high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                     models_calc_LAI=models_calc_LAI, time_scale=time_scale,
    #                     country_code=country_code, LAI_range=LAI_range,  IGBP_type=IGBP_type,
    #                     energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
    #     gc.collect()



    # ========= veg fraction ==========
    # low_bound      = [0,0.2] #30
    # high_bound     = [0.8,1.] #70
    # veg_fraction   = [0,0.3] # low veg fraction
    # LAI_range      = None
    # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
    #                 country_code=country_code,  # IGBP_type=IGBP_type,
    #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)

    # low_bound      = [0,0.2] #30
    # high_bound     = [0.8,1.] #70
    # veg_fraction   = [0.7,1.] # high veg fraction
    # write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, selected_by=selected_by, low_bound=low_bound,
    #                 high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
    #                 models_calc_LAI=models_calc_LAI, time_scale=time_scale, veg_fraction=veg_fraction,
    #                 country_code=country_code,  # IGBP_type=IGBP_type,
    #                 energy_cor=energy_cor, output_2d_grids_only=output_2d_grids_only,regional_sites=regional_sites)
