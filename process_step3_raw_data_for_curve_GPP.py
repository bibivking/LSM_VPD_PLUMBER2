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

def write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=None, low_bound=30,
                            high_bound=70, day_time=False, summer_time=False, IGBP_type=None,
                            clim_type=None, energy_cor=False,VPD_num_threshold=None,
                            models_calc_LAI=None, veg_fraction=None,
                            clarify_site={'opt':False,'remove_site':None}, standardize=None,
                            remove_strange_values=True, country_code=None,
                            hours_precip_free=None, method='GAM', selected_raw_data=True):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # ========== read the data ==========
    if country_code!=None:
        var_output    = pd.read_csv(f'./txt/all_sites/{var_name}_all_sites_with_LAI_'+country_code+'.csv',na_values=[''])
    else:
        var_output    = pd.read_csv(f'./txt/all_sites/{var_name}_all_sites.csv',na_values=[''])
    print( 'Check point 1, np.any(~np.isnan(var_output["model_CABLE"]))=',
           np.any(~np.isnan(var_output["model_CABLE"])) )

    # Using AR-SLu.nc file to get the model namelist
    f             = nc.Dataset(PLUMBER2_path+"/AR-SLu.nc", mode='r')
    model_in_list = f.variables[var_name + '_models']
    ntime         = len(f.variables['CABLE_time'])
    model_out_list= []

    # Compare each model's output time interval with CABLE hourly interval
    # If the model has hourly output then use the model simulation
    for model_in in model_in_list:
        if len(f.variables[f"{model_in}_time"]) == ntime:
            model_out_list.append(model_in)

    # add obs to draw-out namelist
    if var_name in ['Qle','Qh','NEE','GPP']:
        model_out_list.append('obs')
        # model_out_list.append('obs_cor')


    # model_out_list = np.array(model_out_list)
    # total site number
    site_num    = len(np.unique(var_output["site_name"]))
    print('Point 1, site_num=',site_num)
    print('Finish reading csv file')

    # ========== select data ==========

    # whether only considers the sites with energy budget corrected fluxs
    if var_name in ['Qle','Qh'] and energy_cor:
        check_obs_cor = var_output['obs_cor']
        check_obs_cor.to_csv(f'./txt/check_obs_cor.csv')

        cor_notNan_mask = ~ np.isnan(var_output['obs_cor'])
        var_output      = var_output[cor_notNan_mask]

    if remove_strange_values:
        for model_out_name in model_out_list:
            print('Checking strange values in', model_out_name)
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            var_output[head+model_out_name] = np.where(np.any([var_output[head+model_out_name]>999.,
                                                       var_output[head+model_out_name]<-999.],axis=0),
                                                       np.nan, var_output[head+model_out_name])
            print('np.any(np.isnan(var_output[head+model_out_name]))',np.any(np.isnan(var_output[head+model_out_name])))

    # whether only considers day time
    if day_time:
        # print('Check 1 var_output["hour"]', var_output['hour'])

        # Use hours as threshold
        # day_mask    = (var_output['hour'] >= 9) & (var_output['hour'] <= 16)

        # Use radiation as threshold
        day_mask    = (var_output['obs_SWdown'] >= 5)

        var_output  = var_output[day_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 2, site_num=',site_num)

        check_site = var_output[ var_output['site_name']=='CA-NS1']

    # whether only considers summers
    if summer_time:
        summer_mask = (var_output['month'] > 11) | (var_output['month']< 3)
        # print('np.any(summer_mask)', np.any(summer_mask))
        var_output  = var_output[summer_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 3, site_num=',site_num)

    # whether only considers one type of IGBP
    if IGBP_type!=None:
        IGBP_mask   = (var_output['IGBP_type'] == IGBP_type)
        # print('np.any(IGBP_mask)', np.any(IGBP_mask))
        var_output  = var_output[IGBP_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 4, site_num=',site_num)

    # whether only considers one type of climate type
    if clim_type!=None:
        clim_mask   = (var_output['climate_type'] == clim_type)
        # print('np.any(clim_mask)', np.any(clim_mask))
        var_output  = var_output[clim_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 5, site_num=',site_num)

    # select data with required vegetation fraction
    if veg_fraction !=None:
        veg_frac_mask= (var_output['NoahMPv401_greenness']>=veg_fraction[0]) & (var_output['NoahMPv401_greenness']<=veg_fraction[1])
        var_output   = var_output[veg_frac_mask]
        site_num     = len(np.unique(var_output["site_name"]))
        print('Point 6, site_num=',site_num)

    # whether only considers observation without precipitation in hours_precip_free hours
    if hours_precip_free!=None:
        rain_mask   = (var_output['hrs_after_precip'] > hours_precip_free)
        var_output  = var_output[rain_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 7, site_num=',site_num)

    # To exclude the sites have rainfall input problems
    if clarify_site['opt']:
        print('clarifying sites')
        length    = len(var_output)
        site_mask = np.full(length,True)

        for site_remove in clarify_site['remove_site']:
            site_mask = np.where(var_output['site_name'] == site_remove, False, site_mask)
        print('np.all(site_mask)',np.all(site_mask))

        # site_mask = ~(var_output['site_name'] in clarify_site['remove_site'])
        var_output  = var_output[site_mask]
        site_num    = len(np.unique(var_output["site_name"]))
        print('Point 8, site_num=',site_num)

    print( 'Check point 4, np.any(~np.isnan(var_output["model_CABLE"]))=',
           np.any(~np.isnan(var_output["model_CABLE"])) )

    print('Finish selecting data')

    if var_name == 'NEE':
        for model_out_name in model_out_list:
            if 'obs' in model_out_name:
                head = ''
            else:
                head = 'model_'
            if (model_out_name == 'GFDL') | (model_out_name == 'NoahMPv401') | (model_out_name =='STEMMUS-SCOPE') | (model_out_name =='ACASA'):
                print('model_out_name=',model_out_name,'in GFDL, NoahMPv401, STEMMUS-SCOPE,ACASA')
                values = var_output[head+model_out_name]
            else:
                values = var_output[head+model_out_name]*(-1)
            var_output[head+model_out_name] = values

    if standardize == 'by_obs_mean':

        print('standardized_by_obs_mean')

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

        print('site_obs_mean',site_obs_mean)

    elif standardize == 'by_LAI':

        print('standardized_by_LAI')

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

    elif standardize == 'by_monthly_obs_mean':

            print('standardized_by_monthly_obs_mean')

            # read monthly mean
            var_monthly_input = pd.read_csv(f'./txt/all_sites_monthly/{var_name}_all_sites_monthly.csv',
                                            usecols=['month','site_name','obs'],na_values=[''])

            # Get all sites left
            sites_left        = np.unique(var_output["site_name"])

            # Calculute the mean obs for each site and use the mean to standardize the varibale of this file
            for site in sites_left:
                for mth in np.arange(1,13,1):

                    print('site=',site,'mth=',mth)

                    # Get the mask of this site
                    site_mask_tmp       = (var_output['site_name'] == site) & (var_output['month'] == mth)
                    site_mask_month     = (var_monthly_input['month'] == mth) & (var_monthly_input['site_name'] == site)

                    print('Point 8, np.any(site_mask_tmp)',np.any(site_mask_tmp))
                    print('!!! Point 9, np.any(site_mask_month)',np.any(site_mask_month))

                    # Mask the dataframe to get slide of the dataframe for this site
                    var_tmp             = var_output[site_mask_tmp]
                    print('var_tmp', var_tmp)

                    # get site monthly mean
                    site_obs_month      = var_monthly_input.loc[site_mask_month]['obs'].values
                    print('site_obs_month', site_obs_month)

                    # Standardize the different model's values by the obs mean for this site
                    for i, model_out_name in enumerate(model_out_list):
                        if 'obs' in model_out_name:
                            head = ''
                        else:
                            head = 'model_'
                        var_output.loc[site_mask_tmp, head+model_out_name] = var_tmp[head+model_out_name]/site_obs_month
            print('var_output',var_output)

    elif standardize == 'by_monthly_model_mean':

            print('standardized_by_monthly_model_mean')

            # read monthly mean
            var_monthly_input = pd.read_csv(f'./txt/all_sites_monthly/{var_name}_all_sites_monthly.csv',na_values=[''])

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


    if selected_raw_data:
        mask_VPD_tmp       = (var_output['VPD'] != np.nan)
        var_output_raw_data= var_output[mask_VPD_tmp]

        # file name
        message = ''

        if day_time:
            message = message + '_daytime'

        if IGBP_type != None:
            message = message + '_IGBP='+IGBP_type

        if clim_type != None:
            message = message + '_clim='+clim_type

        if standardize != None:
            message = message + '_standardized_'+standardize

        if clarify_site['opt']:
            message = message + '_clarify_site'

        if veg_fraction !=None:
            message = message + '_veg_frac='+str(veg_fraction[0])+'-'+str(veg_fraction[1])

        if country_code !=None:
            message = message +'_'+country_code

        # save data
        if var_name == 'NEE':
            var_name = 'NEP'

        folder_name = 'original'

        if standardize != None:
            folder_name = 'standardized_'+standardize

        if clarify_site['opt']:
            folder_name = folder_name+'_clarify_site'

        var_output_raw_data.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_coarse.csv')

    # ========== Divide dry and wet periods ==========

    # Calculate EF thresholds
    if bin_by == 'EF_obs':

        # select time step where obs_EF isn't NaN (when Qh<0 or Qle+Qh<10)
        EF_notNan_mask = ~ np.isnan(var_output['obs_EF'])
        var_output     = var_output[EF_notNan_mask]

        # print('np.any(EF_notNan_mask)', np.any(EF_notNan_mask))

        # Select EF<low_bound and EF>high_bound for each site to make sure
        # that every site can contribute to the final VPD lines
        for site_name in site_names:

            # select data for this site
            site_mask       = (var_output['site_name'] == site_name)

            print('In bin by EF, site_name=', site_name, 'np.any(site_mask)',np.any(site_mask))

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

    elif bin_by == 'EF_model':

        var_output_dry = copy.deepcopy(var_output)
        var_output_wet = copy.deepcopy(var_output)

        print( 'Check point 6, np.any(~np.isnan(var_output_dry["model_CABLE"]))=',
               np.any(~np.isnan(var_output_dry["model_CABLE"])) )

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

        print( 'Check point 7, np.any(~np.isnan(var_output_dry["model_CABLE"]))=',
               np.any(~np.isnan(var_output_dry["model_CABLE"])) )

    print('Finish dividing dry and wet periods')

    if selected_raw_data:
        mask_VPD_dry       = (var_output_dry['VPD'] != np.nan)
        mask_VPD_wet       = (var_output_wet['VPD'] != np.nan)

        # var_select_dry_tmp = var_output_dry[['VPD','obs','model_CABLE','model_STEMMUS-SCOPE']]
        var_select_dry     = var_output_dry[mask_VPD_dry]
        var_select_wet     = var_output_wet[mask_VPD_wet]

        # file name
        message = ''

        if day_time:
            message = message + '_daytime'

        if IGBP_type != None:
            message = message + '_IGBP='+IGBP_type

        if clim_type != None:
            message = message + '_clim='+clim_type

        if standardize != None:
            message = message + '_standardized_'+standardize

        if clarify_site['opt']:
            message = message + '_clarify_site'

        if veg_fraction !=None:
            message = message + '_veg_frac='+str(veg_fraction[0])+'-'+str(veg_fraction[1])

        if country_code !=None:
            message = message +'_'+country_code

        # save data
        if var_name == 'NEE':
            var_name = 'NEP'

        folder_name = 'original'

        if standardize != None:
            folder_name = 'standardized_'+standardize

        if clarify_site['opt']:
            folder_name = folder_name+'_clarify_site'

        if len(low_bound) >1 and len(high_bound) >1:
            if low_bound[1] > 1:
                var_select_dry.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'th_coarse.csv')
                var_select_wet.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_'+bin_by+'_'+str(high_bound[0])+'-'+str(high_bound[1])+'th_coarse.csv')
            else:
                var_select_dry.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'_coarse.csv')
                var_select_wet.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_'+bin_by+'_'+str(high_bound[0])+'-'+str(high_bound[1])+'_coarse.csv')
        elif len(low_bound) == 1 and len(high_bound) == 1:
            if low_bound > 1:
                var_select_dry.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound)+'th_coarse.csv')
                var_select_wet.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_'+bin_by+'_'+str(high_bound)+'th_coarse.csv')
            else:
                var_select_dry.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound)+'_coarse.csv')
                var_select_wet.to_csv(f'./txt/select_data_point/raw_data_{var_name}_VPD'+message+'_'+bin_by+'_'+str(high_bound)+'_coarse.csv')
    return


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    var_name       = 'GPP'  #'TVeg'
    bin_by         = 'EF_model' #'EF_model' #'EF_obs'#
    method         = 'bin_by_vpd' #'GAM'
    standardize    = 'None' # 'None'
                                   # 'by_obs_mean'
                                   # 'by_LAI'
                                   # 'by_monthly_obs_mean'
                                   # 'by_monthly_model_mean'

    day_time       = True
    energy_cor     = False
    selected_raw_data = True

    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','Noah-MP']

    if var_name == 'NEE':
        energy_cor = False

    # ================== dry_wet ==================
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    low_bound      = [0,0.2] #30
    high_bound     = [0.8,1.] #70
    veg_fraction   = None #[0.7,1]

    write_raw_data_var_VPD(var_name, site_names, PLUMBER2_path, bin_by=bin_by, low_bound=low_bound,
                    high_bound=high_bound, day_time=day_time,clarify_site=clarify_site,standardize=standardize,
                    models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction,
                    country_code=country_code,
                    energy_cor=energy_cor, method=method, selected_raw_data=selected_raw_data)
    gc.collect()
