import os
import re
import gc
import sys
import glob
import psutil
import copy
import netCDF4 as nc
import codecs
import numpy as np
import xarray as xr
import pandas as pd
import joblib
from pygam import LinearGAM, PoissonGAM,GAM, GammaGAM, s, f, l
from sklearn.model_selection import KFold
from scipy import stats, interpolate
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

def get_header(model_out_name):
    if 'obs' in model_out_name:
        return ''
    else:
        return 'model_'

def check_server():

    print("CPU usage:", psutil.cpu_percent())
    print("Memory usage:", psutil.virtual_memory().percent, "%")
    print("Disk usage:", psutil.disk_usage("/").percent, "%")  # Replace "/" with the desired path
    return

def get_region_info(region_name):

    print(region_name)

    if region_name == 'global':
        return {'name':'global', 'lat':None, 'lon':None}
    elif region_name == 'east_AU':
        return {'name':'east_AU', 'lat':[-44.5,-10], 'lon':[129,155]}
    elif region_name == 'west_EU':
        return {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
    elif region_name == 'north_Am':
        return {'name':'north_Am', 'lat':[25,52], 'lon':[-125,-65]}

def get_regional_site_list(region= {'name':'global','lat':None, 'lon':None}):

    '''
    Get the site name list for the selected region
    '''

    PLUMBER2_path      = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    all_site_path      = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names         = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_met_path)

    sites              = []

    for site_name in site_names:

        within_region = ((lat_dict[site_name] >= region['lat'][0]) &
                         (lat_dict[site_name] <= region['lat'][1]) &
                         (lon_dict[site_name] >= region['lon'][0]) &
                         (lon_dict[site_name] <= region['lon'][1]))
        if within_region:
            sites.append(site_name)
    regional_sites = {'name':region['name'],'sites':sites}
    return regional_sites

def decide_filename(day_time=False, summer_time=False, energy_cor=False,
                    IGBP_type=None, clim_type=None, time_scale=None, standardize=None,
                    country_code=None, selected_by=None, bounds=None, veg_fraction=None,
                    uncertain_type=None, method=None,LAI_range=None, add_Xday_mean_EF=None,
                    clarify_site={'opt':False,'remove_site':None},regional_sites=None,
                    data_selection=True):

    # file name
    file_message = ''

    if time_scale != None:
        file_message = file_message + '_' + time_scale

    if IGBP_type != None:
        file_message = file_message + '_PFT='+IGBP_type

    if clim_type != None:
        file_message = file_message + '_CLIM='+clim_type

    if veg_fraction !=None:
        # if selected based on vegetation fraction
        file_message = file_message + '_VF='+str(veg_fraction[0])+'-'+str(veg_fraction[1])

    if LAI_range !=None:
        # if selected based on LAI
        file_message = file_message + '_LAI='+str(LAI_range[0])+'-'+str(LAI_range[1])

    if country_code !=None:
        # if for a country/region
        file_message = file_message +'_'+country_code

    if clarify_site['opt'] and (not data_selection):
        # if remove 16 sites with problems in observation
        file_message = file_message + '_RM16'

    if day_time and (not data_selection):
        # if only daytime
        file_message = file_message + '_DT'

    if standardize != None:
        # if the data is standardized
        file_message = file_message + '_'+standardize

    if selected_by !=None:
        # which criteria used for binning the data
        file_message = file_message +'_'+selected_by
        if add_Xday_mean_EF != None:
            file_message = file_message + '_' + add_Xday_mean_EF + 'day_mean'

        if len(bounds) >1:
            # percentile
            if bounds[1] > 1:
                file_message = file_message + '_'+str(bounds[0])+'-'+str(bounds[1])+'th'
            else:
                file_message = file_message + '_'+str(bounds[0])+'-'+str(bounds[1])
        elif len(bounds) == 1 :
            # fraction
            if bounds[1] > 1:
                file_message = file_message + '_'+str(bounds[0])+'th'
            else:
                file_message = file_message + '_'+str(bounds[0])

    if method != None:
        file_message = file_message + '_' + method

    if uncertain_type != None and method == 'CRV_bins':
        file_message = file_message + '_' + uncertain_type

    if regional_sites != None:
        file_message = file_message+'_'+regional_sites['name']

    if data_selection:
        file_message = file_message+'_data_selected'

    folder_name = 'original'

    if standardize != None:
        folder_name = 'standardized_'+standardize

    if clarify_site['opt']:
        folder_name = folder_name+'_clarify_site'

    return folder_name, file_message

def get_model_out_list(var_name):

    # Using AR-SLu.nc file to get the model namelist
    f             = nc.Dataset("/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/AR-SLu.nc", mode='r')
    if var_name == 'Gs':
        model_in_list = f.variables['Qle_models']
    elif 'TVeg' in var_name:
        model_in_list = f.variables['TVeg_models']
    elif 'SMtop' in var_name:
        model_in_list = f.variables['Qle_models']
        # SM_names, soil_thicknesses = get_model_soil_moisture_info('AU-Tum')
        # model_in_list = list(SM_names.keys())
    elif var_name == 'SWdown':
        model_in_list = f.variables['Qle_models']
    elif 'VPD_caused' in var_name:
        model_in_list = f.variables['Qle_models']
    elif var_name == 'LAI':
        model_in_list = ['ORC2_r6593','ORC3_r8120','GFDL','QUINCY','NoahMPv401'] #'obs',
    else:
        model_in_list = f.variables[var_name + '_models']
    ntime         = len(f.variables['CABLE_time'])
    model_out_list= []

    # Compare each model's output time interval with CABLE hourly interval
    # If the model has hourly output then use the model simulation
    for model_in in model_in_list:
        if len(f.variables[f"{model_in}_time"]) == ntime and model_in != '1lin':
            model_out_list.append(model_in)

    # add obs to draw-out namelist
    if var_name in ['Qle','Qh','NEE','GPP','EF','LAI','Rnet','SWdown','Qle_VPD_caused']:
        model_out_list.append('obs')

    return model_out_list

def calculate_VPD_by_RH(rh, tair):

    '''
    calculate vpd
    '''
    print('calculating VPD')
    # set nan values
    rh   = np.where(rh<-1.,np.nan,rh)
    tair = np.where(tair<-100.,np.nan,tair)

    DEG_2_KELVIN = 273.15
    PA_TO_KPA    = 0.001

    # convert from K to C
    tair         -= DEG_2_KELVIN

    # saturation vapor pressure
    es = 100.0 * 6.112 * np.exp((17.67 * tair) / (243.5 + tair))

    # actual vapor pressure
    ea = rh/100. * es

    vpd = (es - ea) * PA_TO_KPA
    # vpd = np.where(vpd < 0.05, 0.05, vpd)

    return vpd

def calculate_VPD_by_Qair(qair, tair, press):
    '''
    calculate vpd
    Input:
          qair: kg/kg
          tair: K
          press: Pa
    Output:
          vpd: kPa
    '''

    # set nan values
    qair = np.where(qair==-9999.,np.nan,qair)
    tair = np.where(tair==-9999.,np.nan,tair)
    press= np.where(press<-9999.,np.nan,press)

    DEG_2_KELVIN = 273.15
    PA_TO_KPA    = 0.001
    PA_TO_HPA    = 0.01

    # convert back to Pa
    # press        /= PA_TO_HPA
    tair         -= DEG_2_KELVIN

    # saturation vapor pressure
    es = 100.0 * 6.112 * np.exp((17.67 * tair) / (243.5 + tair))

    # vapor pressure
    ea = (qair * press) / (0.622 + (1.0 - 0.622) * qair)

    print('np.shape(qair)',np.shape(qair))
    print('np.shape(tair)',np.shape(tair))
    print('np.shape(press)',np.shape(press))
    print('np.shape(es)',np.shape(es))
    print('np.shape(ea)',np.shape(ea))


    vpd = (es - ea) * PA_TO_KPA
    # vpd = np.where(vpd < 0.0, 0.0, vpd)

    return vpd

def change_model_name(model_in):

    if model_in == 'CABLE-POP-CN':
        model_out = 'CABLE-POP'

    elif model_in == 'CHTESSEL_Ref_exp1':
        model_out = 'CHTESSEL_1'

    elif model_in == 'CLM5a':
        model_out = 'CLM5'

    elif model_in == 'JULES_GL9_withLAI':
        model_out = 'JULES_GL9_LAI'

    elif model_in == 'NASAEnt':
        model_out = 'EntTBM'

    elif model_in == 'NoahMPv401':
        model_out = 'NoahMP'

    elif model_in == 'ORC2_r6593':
        model_out = 'ORCHIDEE2'

    elif model_in == 'ORC3_r8120':
        model_out = 'ORCHIDEE3'

    elif model_in == 'obs':
        model_out = 'Observed'
    else:
        model_out = model_in

    return model_out

def check_site_info(site_name,var_to_check):

    process1_names      = ['NoahMPv401_greenness','IGBP_type','climate_type']

    if var_to_check in process1_names:
        var_tmp = pd.read_csv('/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/Qle_all_sites.csv',
                  usecols=['site_name',var_to_check])
        return var_tmp.loc[(var_tmp['site_name'] == site_name), var_to_check].values[0]
    elif var_to_check == 'aridity_index':
        var_tmp = pd.read_csv('/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/Aridity_index_all_sites.csv',
                  usecols=['site_name',var_to_check])
        return var_tmp.loc[(var_tmp['site_name'] == site_name), var_to_check].values[0]

def load_default_list():

    # The site names
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]

    IGBP_types     = ['GRA','OSH', 'SAV', 'WSA', 'CSH', 'DBF', 'ENF', 'EBF', 'MF', 'WET', 'CRO']
    clim_types     = ['Af', 'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Csa', 'Csb', 'Cwa',
                      'Dfa', 'Dfb', 'Dfc', 'Dsb', 'Dsc', 'Dwa', 'Dwb', 'ET']

    model_names_all = [ "1lin","3km27", "6km729","6km729lag",
                        "LSTM_eb","LSTM_raw", "RF_eb","RF_raw",
                        "Manabe", "ManabeV2","PenmanMonteith",
                        "CABLE", "CABLE-POP-CN","CHTESSEL_ERA5_3",
                        "CHTESSEL_Ref_exp1","CLM5a","GFDL",
                        "JULES_GL9_withLAI","JULES_test","MATSIRO",
                        "NoahMPv401","ORC2_r6593", "ORC2_r6593_CO2",
                        "ORC3_r7245_NEE", "ORC3_r8120","STEMMUS-SCOPE",
                        "ACASA","LPJ-GUESS","MuSICA",
                        "NASAEnt","QUINCY", "SDGVM",] #"BEPS"

    model_names_select= ['CABLE', 'CABLE-POP-CN', 'CHTESSEL_ERA5_3',
                         'CHTESSEL_Ref_exp1', 'CLM5a', 'GFDL',
                         'JULES_GL9_withLAI', 'JULES_test',
                         'MATSIRO', 'MuSICA', 'NASAEnt',
                         'NoahMPv401', 'ORC2_r6593', 'ORC2_r6593_CO2',
                         'ORC3_r7245_NEE', 'ORC3_r8120', 'QUINCY',
                         'STEMMUS-SCOPE', 'obs'] #"BEPS"

    model_names_new_select= ['CABLE', 'CABLE-POP-CN',
                             'CHTESSEL_Ref_exp1', 'CLM5a', 'GFDL',
                             'JULES_GL9', 'JULES_GL9_withLAI',
                             'MATSIRO', 'MuSICA', 'NASAEnt',
                             'NoahMPv401', 'ORC2_r6593',
                             'ORC3_r8120', 'QUINCY',
                             'STEMMUS-SCOPE', 'obs'] #"BEPS"

    model_names_tveg  = ['CABLE', 'CABLE-POP-CN',
                         'CHTESSEL_Ref_exp1', 'CLM5a', 'GFDL',
                         'JULES_GL9_withLAI',
                         'MATSIRO', 'MuSICA',
                         'NoahMPv401', 'ORC2_r6593',
                         'ORC3_r8120', 'QUINCY'] #"BEPS"


    empirical_model   = ["1lin","3km27", "6km729","6km729lag",
                        "LSTM_eb","LSTM_raw", "RF_eb","RF_raw",]

    hydrological_model= ["Manabe", "ManabeV2","PenmanMonteith",]

    land_surface_model= ["CABLE", "CABLE-POP-CN","CHTESSEL_ERA5_3",
                         "CHTESSEL_Ref_exp1","CLM5a","GFDL",
                         "JULES_GL9_withLAI","JULES_test","MATSIRO",
                         "NoahMPv401","ORC2_r6593", "ORC2_r6593_CO2",
                         "ORC3_r7245_NEE", "ORC3_r8120","STEMMUS-SCOPE",]

    ecosystem_model  = ["ACASA","LPJ-GUESS","MuSICA",
                        "NASAEnt","QUINCY", "SDGVM",]

    model_names      = {'all_model': model_names_all,
                        'model_select':model_names_select,
                        'model_select_new':model_names_new_select,
                        'model_tveg': model_names_tveg,
                        'empirical_model':empirical_model,
                        'hydrological_model':hydrological_model,
                        'land_surface_model':land_surface_model,
                        'ecosystem_model':ecosystem_model}

    return site_names, IGBP_types, clim_types, model_names

def load_sites_in_country_list(country_code):

    # The site names
    if country_code != None:
        PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
        all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*"+country_code+"*.nc"))
        site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    else:
        site_names     = None
    return site_names


def fit_GAM_simple_for_all_sites(x_top, x_bot, x_interval, x_values, y_values,n_splines=4,spline_order=2):

    x_series   = np.arange(x_bot, x_top, x_interval)

    # x_values in gridsearch should be of shape [n_samples, m_features], for this case
    # m_features=1, so reshape x_values to [len(x_values),1]
    x_values_array = x_values.to_numpy()
    x_values       = x_values_array.reshape(len(x_values),1)

    # calculate mean value
    gam          = PoissonGAM(n_splines=n_splines,spline_order=spline_order).gridsearch(x_values, y_values) # n_splines=22
    # gam          = LinearGAM(n_splines=n_splines,spline_order=spline_order).gridsearch(x_values, y_values) # n_splines=22
    y_pred       = gam.predict(x_series)
    y_int        = gam.confidence_intervals(x_series, width=.95)

    return x_series, y_pred, y_int

def fit_GAM_complex(model_out_name, var_name, folder_name, file_message, x_top, x_bot, x_interval, x_values, y_values, dist_type='Linear', vpd_top_type='to_10'):

    '''
    In this method, it tests all parameter combinations to select the best parameter
    combination for each fold, and select from all folds to get the best combination.
    However, the best parameter combination should come from best average score of 5
    folds. So I wrote a new function to replace def fit_GAM_complex_old.
    '''
    print(model_out_name, 'x_top',x_top)


    if vpd_top_type == 'sample_larger_200':
        subfolder_name = f'{dist_type}_greater_200_samples'
        concave        = False

    elif vpd_top_type == 'to_10':
        subfolder_name = f'{dist_type}_to_10'
        concave        = True

    # In case no VPD bin has data points >= VPD_num_threshold
    if np.isnan(x_top):
        return np.nan, np.nan, np.nan

    # Remove nan values
    x_values_tmp = copy.deepcopy(x_values)
    y_values_tmp = copy.deepcopy(y_values)

    # copy.deepcopy: creates a complete, independent copy of an object
    # and its entire internal structure, including nested objects and any
    # references they contain.

    nonnan_mask = (~np.isnan(x_values_tmp)) & (~np.isnan(y_values_tmp))
    x_values    = x_values_tmp[nonnan_mask]
    y_values    = y_values_tmp[nonnan_mask]

    if len(x_values) <= 10:
        print("Alarm! Not enought sample")
        return np.nan, np.nan, np.nan

    # Set x_series
    x_series   = np.arange(x_bot, x_top, x_interval)

    # GAM parameter set 6 -- new try since optimized n_spline mostly = 10,
    # thus increase the range to find the best, sample_larger_200:
    # lam        = np.logspace(-3, 3, 11)#np.logspace(-3, 3, 21)  # Smoothing parameter range
    # n_splines  = np.arange(5, 14, 1)   # Number of splines per smooth term range

    # GAM parameter set 5 -- final for sample_larger_200:
    lam        = np.logspace(-3, 3, 11)#np.logspace(-3, 3, 21)  # Smoothing parameter range
    n_splines  = np.arange(3, 6, 1)   # Number of splines per smooth term range

    # Set up KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize empty lists for storing results
    models = []
    scores = []

    # Perform grid search
    for train_index, test_index in kf.split(x_values):
        X_train, X_test = x_values[train_index], x_values[test_index]
        y_train, y_test = y_values[train_index], y_values[test_index]

        X_train = X_train.reshape(-1, 1)

        print('X_train.shape', X_train.shape)
        print('X_test.shape', X_test.shape)
        print('y_train.shape', y_train.shape)
        print('y_test.shape', y_test.shape)

        # Define and fit GAM model
        if dist_type=='Linear':
            gam = LinearGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines) #  + f(0)
        elif dist_type=='Poisson':
            # concave assumption
            if concave:
                gam = PoissonGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave')).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)
            else:
                gam = PoissonGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)
        elif dist_type=='Gamma':
            if concave:
                # concave assumption
                gam = GammaGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave')).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)
            else:
                gam = GammaGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)


        # LinearGAM(s(0)) creates a GAM with a spline term
        # edge_knots: specify minimum and maximum domain of the spline function.
        #             To make GAM model covers the whole range of VPD, I manually set them as VPD boundaries

        # Alternative
        # gam = LinearGAM(s(0) + f(0)).fit(X_train, y_train)
        # Fits the GAM model to the data using default hyperparameters.

        # process4-4
        models.append(gam)
        # Evaluate model performance (replace with your preferred metric)
        score = gam.score(X_test, y_test) # gam.score: compute the explained deviance for a trained model for a given X data and y labels
        scores.append(score)

    # Find the best model based on average score
    # For gam.score which calcuate deviance, the best model's score is closet to 0
    best_model_index = np.argmin(np.abs(scores))
    best_model       = models[best_model_index]
    best_score       = scores[best_model_index]

    print(model_out_name,'scores',scores)
    print(model_out_name,'best_score',best_score)
    print(model_out_name,'best_model_index',best_model_index)
    print(model_out_name,'best_model',best_model)
    print(f"{model_out_name} best model parameters: {best_model.lam}, {best_model.n_splines}")

    # Save the best model using joblib
    joblib.dump(best_model, f"./txt/process4_output/{folder_name}/{subfolder_name}/GAM_fit/bestGAM_{var_name}{file_message}_{model_out_name}_{dist_type}.pkl")

    # Further analysis of the best model (e.g., plot smoothers, analyze interactions)
    y_pred       = best_model.predict(x_series)

    # Note that The code calculates 95% confidence intervals, but remember that confidence
    #           intervals should generally be calculated on the actual test data points,
    #           not a new set of equally spaced values like x_series
    y_int        = best_model.confidence_intervals(x_series, width=.95)

    # Create the scatter plot for X and Y
    plt.scatter(x_values, y_values, s=0.5, facecolors='none', edgecolors='blue',  alpha=0.5, label='data points')

    # Plot the line for X_predict and Y_predict
    plt.plot(x_series, y_pred, color='red', label='Predicted line')
    plt.fill_between(x_series,y_int[:,1],y_int[:,0], color='red', edgecolor="none", alpha=0.1) #  .

    # Add labels and title
    plt.xlabel('VPD')
    plt.ylabel('Qle')
    plt.title('Check the GAM fitted curve')
    # plt.xlim(0, 10)  # Set x-axis limits
    plt.ylim(0, 800)  # Set y-axis limits

    # Add legend
    plt.legend()

    plt.savefig(f'./check_plots/check_{var_name}_{model_out_name}_GAM_fitted_curve_{dist_type}.png',dpi=600)

    return x_series, y_pred, y_int


def fit_GAM_complex_old(model_out_name, var_name, folder_name, file_message, x_top, x_bot, x_interval, x_values, y_values, dist_type='Linear', vpd_top_type='to_10'):

    '''
    In this method, it tests all parameter combinations to select the best parameter
    combination for each fold, and select from all folds to get the best combination.
    However, the best parameter combination should come from best average score of 5
    folds. So I wrote a new function to replace def fit_GAM_complex_old.
    '''
    print(model_out_name, 'x_top',x_top)


    if vpd_top_type == 'sample_larger_200':
        subfolder_name = f'{dist_type}_greater_200_samples'
        concave        = False

    elif vpd_top_type == 'to_10':
        subfolder_name = f'{dist_type}_to_10'
        concave        = True

    # In case no VPD bin has data points >= VPD_num_threshold
    if np.isnan(x_top):
        return np.nan, np.nan, np.nan

    # Remove nan values
    x_values_tmp = copy.deepcopy(x_values)
    y_values_tmp = copy.deepcopy(y_values)

    # copy.deepcopy: creates a complete, independent copy of an object
    # and its entire internal structure, including nested objects and any
    # references they contain.

    nonnan_mask = (~np.isnan(x_values_tmp)) & (~np.isnan(y_values_tmp))
    x_values    = x_values_tmp[nonnan_mask]
    y_values    = y_values_tmp[nonnan_mask]

    if len(x_values) <= 10:
        print("Alarm! Not enought sample")
        return np.nan, np.nan, np.nan

    # Set x_series
    x_series   = np.arange(x_bot, x_top, x_interval)

    # GAM parameter set 6 -- new try since optimized n_spline mostly = 10,
    # thus increase the range to find the best, sample_larger_200:
    # lam        = np.logspace(-3, 3, 11)#np.logspace(-3, 3, 21)  # Smoothing parameter range
    # n_splines  = np.arange(5, 14, 1)   # Number of splines per smooth term range

    # GAM parameter set 5 -- final for sample_larger_200:
    lam        = np.logspace(-3, 3, 11)#np.logspace(-3, 3, 21)  # Smoothing parameter range
    n_splines  = np.arange(3, 11, 1)   # Number of splines per smooth term range

    # # GAM parameter set 4: => high VPD end surge quickly
    # lam        = np.logspace(-1, 5, 11)#np.logspace(-3, 3, 21)  # Smoothing parameter range
    # n_splines  = np.arange(3, 11, 1)   # Number of splines per smooth term range

    # # GAM parameter set 3:
    # lam        = np.logspace(-3, 3, 11)#np.logspace(-3, 3, 21)  # Smoothing parameter range
    # n_splines  = np.arange(3, 7, 1)   # Number of splines per smooth term range

    # # GAM parameter set 2:
    # lam        = np.logspace(-3, 3, 11)#np.logspace(-3, 3, 21)  # Smoothing parameter range
    # n_splines  = np.arange(3, 5, 1)   # Number of splines per smooth term range

    # GAM parameter set 1:
    #     lam        = np.logspace(-5, 5, 11)
    #     n_splines  = np.arange(3, 10, 1)

    # lam: Smoothing parameter controlling model complexity (21 values between 10^-3 and 10^3).
    # n_splines: Number of spline basis functions (7 values between 3 and 9).

    # Set up KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize empty lists for storing results
    models = []
    scores = []

    # Perform grid search
    for train_index, test_index in kf.split(x_values):
        X_train, X_test = x_values[train_index], x_values[test_index]
        y_train, y_test = y_values[train_index], y_values[test_index]

        X_train = X_train.reshape(-1, 1)

        print('X_train.shape', X_train.shape)
        print('X_test.shape', X_test.shape)
        print('y_train.shape', y_train.shape)
        print('y_test.shape', y_test.shape)

        # Define and fit GAM model
        if dist_type=='Linear':
            gam = LinearGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines) #  + f(0)
        elif dist_type=='Poisson':
            # concave assumption
            if concave:
                gam = PoissonGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave')).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)
            else:
                gam = PoissonGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)
        elif dist_type=='Gamma':
            if concave:
                # concave assumption
                gam = GammaGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave')).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)
            else:
                gam = GammaGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)


            # no concave assumption
            # gam = GammaGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)

            # process4-3
            # for n_spline in n_splines:
            #     gam = GammaGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave', spline_order=3, n_splines=n_spline)
            #           + l(0)).gridsearch(X_train, y_train, lam=lam)
            #     models.append(gam)

            #     # Evaluate model performance (replace with your preferred metric)
            #     score = gam.score(X_test, y_test) # gam.score: compute the explained deviance for a trained model for a given X data and y labels
            #     scores.append(score)

            # gam = GAM(s(0, edge_knots=[x_bot, x_top]),distribution='gamma', link='inverse').gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)

        # LinearGAM(s(0)) creates a GAM with a spline term
        # edge_knots: specify minimum and maximum domain of the spline function.
        #             To make GAM model covers the whole range of VPD, I manually set them as VPD boundaries

        # Alternative
        # gam = LinearGAM(s(0) + f(0)).fit(X_train, y_train)
        # Fits the GAM model to the data using default hyperparameters.

        # process4-4
        models.append(gam)
        # Evaluate model performance (replace with your preferred metric)
        score = gam.score(X_test, y_test) # gam.score: compute the explained deviance for a trained model for a given X data and y labels
        scores.append(score)

    # Find the best model based on average score
    # For gam.score which calcuate deviance, the best model's score is closet to 0
    best_model_index = np.argmin(np.abs(scores))
    best_model       = models[best_model_index]
    best_score       = scores[best_model_index]

    print(model_out_name,'scores',scores)
    print(model_out_name,'best_score',best_score)
    print(model_out_name,'best_model_index',best_model_index)
    print(model_out_name,'best_model',best_model)
    print(f"{model_out_name} best model parameters: {best_model.lam}, {best_model.n_splines}")

    # Save the best model using joblib
    joblib.dump(best_model, f"./txt/process4_output/{folder_name}/{subfolder_name}/GAM_fit/bestGAM_{var_name}{file_message}_{model_out_name}_{dist_type}.pkl")

    # Further analysis of the best model (e.g., plot smoothers, analyze interactions)
    y_pred       = best_model.predict(x_series)

    # Note that The code calculates 95% confidence intervals, but remember that confidence
    #           intervals should generally be calculated on the actual test data points,
    #           not a new set of equally spaced values like x_series
    y_int        = best_model.confidence_intervals(x_series, width=.95)

    # Create the scatter plot for X and Y
    plt.scatter(x_values, y_values, s=0.5, facecolors='none', edgecolors='blue',  alpha=0.5, label='data points')

    # Plot the line for X_predict and Y_predict
    plt.plot(x_series, y_pred, color='red', label='Predicted line')
    plt.fill_between(x_series,y_int[:,1],y_int[:,0], color='red', edgecolor="none", alpha=0.1) #  .

    # Add labels and title
    plt.xlabel('VPD')
    plt.ylabel('Qle')
    plt.title('Check the GAM fitted curve')
    # plt.xlim(0, 10)  # Set x-axis limits
    plt.ylim(0, 800)  # Set y-axis limits

    # Add legend
    plt.legend()

    plt.savefig(f'./check_plots/check_{var_name}_{model_out_name}_GAM_fitted_curve_{dist_type}.png',dpi=600)

    return x_series, y_pred, y_int

def fit_GAM_complex_new(model_out_name, var_name, folder_name, file_message, x_top,
    x_bot, x_interval, x_values, y_values, dist_type='Linear', vpd_top_type='to_10'):

    print(model_out_name, 'x_top',x_top)

    if vpd_top_type == 'sample_larger_200':
        subfolder_name = f'{dist_type}_greater_200_samples'
        concave        = False

    elif vpd_top_type == 'to_10':
        subfolder_name = f'{dist_type}_to_10'
        concave        = True

    # In case no VPD bin has data points >= VPD_num_threshold
    if np.isnan(x_top):
        return np.nan, np.nan, np.nan

    # Remove nan values
    x_values_tmp = copy.deepcopy(x_values)
    y_values_tmp = copy.deepcopy(y_values)

    # copy.deepcopy: creates a complete, independent copy of an object
    # and its entire internal structure, including nested objects and any
    # references they contain.

    nonnan_mask = (~np.isnan(x_values_tmp)) & (~np.isnan(y_values_tmp))
    x_values    = x_values_tmp[nonnan_mask]
    y_values    = y_values_tmp[nonnan_mask]

    if len(x_values) <= 10:
        print("Alarm! Not enought sample")
        return np.nan, np.nan, np.nan

    # Set x_series
    x_series   = np.arange(x_bot, x_top, x_interval)

    # GAM parameter set7 -- lam -3,3; n_splines 3,11 doesn't work work well so change to new ranges:
    lams           = np.logspace(-3, 2, 10)  # Smoothing parameter range
    n_splines      = np.arange(3, 11, 1)     # Number of splines per smooth term range
    parameter_grid = [(lam, n_spline) for lam in lams for n_spline in n_splines]

    # lam: Smoothing parameter controlling model complexity (21 values between 10^-3 and 10^3).
    # n_splines: Number of spline basis functions (7 values between 3 and 9).

    # Set up KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Store results for each parameter set
    param_results = []

    for lam, n_spline in parameter_grid:

        print('lam',lam,'n_spline',n_spline)

        fold_scores = []

        try:
            for train_index, test_index in kf.split(x_values):
                X_train, X_test = x_values[train_index], x_values[test_index]
                y_train, y_test = y_values[train_index], y_values[test_index]

                X_train = X_train.reshape(-1, 1)

                # Define and fit GAM model based on the dist_type
                if dist_type == 'Linear':
                    gam = LinearGAM(s(0, edge_knots=[x_bot, x_top]), lam=lam, n_splines=n_spline).fit(X_train, y_train)
                elif dist_type == 'Poisson':
                    if concave:
                        gam = PoissonGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave'), lam=lam, n_splines=n_spline).fit(X_train, y_train)
                    else:
                        gam = PoissonGAM(s(0, edge_knots=[x_bot, x_top]), lam=lam, n_splines=n_spline).fit(X_train, y_train)
                elif dist_type == 'Gamma':
                    if concave:
                        gam = GammaGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave'), lam=lam, n_splines=n_spline).fit(X_train, y_train)
                    else:
                        gam = GammaGAM(s(0, edge_knots=[x_bot, x_top]), lam=lam, n_splines=n_spline).fit(X_train, y_train)

                # Evaluate model performance
                score = gam.score(X_test, y_test)  # compute explained deviance
                fold_scores.append(score)

        except Exception as e:
            print(f'Error for {model_out_name}, lam={lam}, n_spline={n_spline}: {e}')
            continue

        # Calculate the average performance for the current parameter set
        if fold_scores:
            avg_score = np.mean(fold_scores)
            param_results.append((lam, n_spline, avg_score))

    # Find the parameter set with the best average performance

    # Print the best parameters and their score
    print(f"{model_out_name}, param_results: {param_results}")

    best_params = min(param_results, key=lambda x: np.abs(x[2]))
    best_lam, best_n_spline, best_score = best_params

    # Print the best parameters and their score
    print(f"Best Parameters: lam={best_lam}, n_splines={best_n_spline} with average score={best_score}")

    # Fit the best model with the best parameters on the full dataset
    if dist_type == 'Linear':
        best_model = LinearGAM(s(0, edge_knots=[x_bot, x_top]), lam=best_lam, n_splines=best_n_spline).fit(x_values, y_values)
    elif dist_type == 'Poisson':
        if concave:
            best_model = PoissonGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave'), lam=best_lam, n_splines=best_n_spline).fit(x_values, y_values)
        else:
            best_model = PoissonGAM(s(0, edge_knots=[x_bot, x_top]), lam=best_lam, n_splines=best_n_spline).fit(x_values, y_values)
    elif dist_type == 'Gamma':
        if concave:
            best_model = GammaGAM(s(0, edge_knots=[x_bot, x_top], constraints='concave'), lam=best_lam, n_splines=best_n_spline).fit(x_values, y_values)
        else:
            best_model = GammaGAM(s(0, edge_knots=[x_bot, x_top]), lam=best_lam, n_splines=best_n_spline).fit(x_values, y_values)

    # Save the best model using joblib
    joblib.dump(best_model, f"./txt/process4_output/{folder_name}/{subfolder_name}/GAM_fit/bestGAM_{var_name}{file_message}_{model_out_name}_{dist_type}.pkl")

    # Generate predictions and confidence intervals
    y_pred   = best_model.predict(x_series)
    y_int    = best_model.confidence_intervals(x_series, width=.95)

    # Create the scatter plot for X and Y
    plt.scatter(x_values, y_values, s=0.5, facecolors='none', edgecolors='blue',  alpha=0.5, label='data points')

    # Plot the line for X_predict and Y_predict
    plt.plot(x_series, y_pred, color='red', label='Predicted line')
    plt.fill_between(x_series,y_int[:,1],y_int[:,0], color='red', edgecolor="none", alpha=0.1) #  .

    # Add labels and title
    plt.xlabel('VPD')
    plt.ylabel('Qle')
    plt.title('Check the GAM fitted curve')
    # plt.xlim(0, 10)  # Set x-axis limits
    plt.ylim(0, 800)  # Set y-axis limits

    # Add legend
    plt.legend()
    plt.savefig(f'./check_plots/check_{var_name}_{model_out_name}_GAM_fitted_curve_{dist_type}.png',dpi=600)

    return x_series, y_pred, y_int

def fit_GAM_CMIP6_predict(model_out_name, file_output, x_series, x_values, y_values, dist_type='Linear'):

    # Remove nan values

    x_interval = x_series[1]-x_series[0]
    x_bot      = x_series[0]-x_interval/2.
    x_top      = x_series[-1]+x_interval/2.

    x_values_tmp = copy.deepcopy(x_values)
    y_values_tmp = copy.deepcopy(y_values)

    nonnan_mask = (~np.isnan(x_values_tmp)) & (~np.isnan(y_values_tmp))
    x_values    = x_values_tmp[nonnan_mask]
    y_values    = y_values_tmp[nonnan_mask]

    if len(x_values) <= 10:
        print("Alarm! Not enought sample")
        return np.nan, np.nan, np.nan

    # GAM parameter set 5 -- final for sample_larger_200:
    lam        = np.logspace(-3, 3, 11)#np.logspace(-3, 3, 21)  # Smoothing parameter range
    n_splines  = np.arange(3, 11, 1)   # Number of splines per smooth term range

    # Set up KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize empty lists for storing results
    models = []
    scores = []

    # Perform grid search
    for train_index, test_index in kf.split(x_values):
        X_train, X_test = x_values[train_index], x_values[test_index]
        y_train, y_test = y_values[train_index], y_values[test_index]

        X_train = X_train.reshape(-1, 1)

        # Define and fit GAM model
        if dist_type=='Linear':
            gam = LinearGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines) #  + f(0)
        elif dist_type=='Poisson':
            gam = PoissonGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)
        elif dist_type=='Gamma':
            gam = GammaGAM(s(0, edge_knots=[x_bot, x_top])).gridsearch(X_train, y_train, lam=lam, n_splines=n_splines)

        # process4-4
        models.append(gam)
        # Evaluate model performance (replace with your preferred metric)
        score = gam.score(X_test, y_test) # gam.score: compute the explained deviance for a trained model for a given X data and y labels
        scores.append(score)

    # Find the best model based on average score
    # For gam.score which calcuate deviance, the best model's score is closet to 0
    best_model_index = np.argmin(np.abs(scores))
    best_model       = models[best_model_index]
    best_score       = scores[best_model_index]

    print(model_out_name,'scores',scores)
    print(model_out_name,'best_score',best_score)
    print(model_out_name,'best_model_index',best_model_index)
    print(model_out_name,'best_model',best_model)
    print(f"{model_out_name} best model parameters: {best_model.lam}, {best_model.n_splines}")

    # Save the best model using joblib
    joblib.dump(best_model, f"./txt/CMIP6/GAM_fit/bestGAM_{file_output}.pkl")

    # Further analysis of the best model (e.g., plot smoothers, analyze interactions)
    y_pred       = best_model.predict(x_series)
    y_int        = best_model.confidence_intervals(x_series, width=.95)

    # Save output
    var             = pd.DataFrame(x_series, columns=['bin_series'])
    var['vals']     = y_pred
    var['vals_top'] = y_int[:,0]
    var['vals_bot'] = y_int[:,1]
    var.to_csv(f'./txt/CMIP/GAM_fit/GAM_{file_output}.csv')

    # Create the scatter plot for X and Y
    plt.scatter(x_values, y_values, s=0.5, facecolors='none', edgecolors='blue',  alpha=0.5, label='data points')

    # Plot the line for X_predict and Y_predict
    plt.plot(x_series, y_pred, color='red', label='Predicted line')
    plt.fill_between(x_series,y_int[:,1],y_int[:,0], color='red', edgecolor="none", alpha=0.1) #  .

    # Add labels and title
    plt.xlabel('VPD')
    plt.ylabel('Qle')
    plt.title('Check the GAM fitted curve')
    plt.ylim(0, 800)  # Set y-axis limits

    # Add legend
    plt.legend()

    plt.savefig(f'./check_plots/check_CMIP6_GAM_fitted_{file_out_message}_{dist_type}.png',dpi=600)

    return

def read_best_GAM_model(var_name, model_out_name, folder_name, file_message, x_values, dist_type=None,
                        vpd_top_type='sample_larger_200',confidence_intervals=False):
    '''
    Read the fitted GAM curve and based on x_values to calcuate predicted Y
    '''

    print('dist_type',dist_type)

    # Load the model using joblib
    # if dist_type == None:
    #     best_model = joblib.load(f"./txt/process4_output/{folder_name}/Gamma_concave/GAM_fit/bestGAM_{var_name}{file_message}_{model_out_name}.pkl")
    # else:
    #     best_model = joblib.load(f"./txt/process4_output/{folder_name}/Gamma_concave/GAM_fit/bestGAM_{var_name}{file_message}_{model_out_name}_{dist_type}.pkl")

    if vpd_top_type == 'sample_larger_200':
        subfolder_message= 'greater_200_samples'
    elif vpd_top_type == 'to_10':
        subfolder_message= 'to_10'

    if dist_type == None:
        best_model = joblib.load(f"./txt/process4_output/{folder_name}/{dist_type}_{subfolder_message}/GAM_fit/bestGAM_{var_name}{file_message}_{model_out_name}.pkl")
    else:
        best_model = joblib.load(f"./txt/process4_output/{folder_name}/{dist_type}_{subfolder_message}/GAM_fit/bestGAM_{var_name}{file_message}_{model_out_name}_{dist_type}.pkl")

    # Predict using new data
    y_pred = best_model.predict(x_values)

    if confidence_intervals:
        y_int = best_model.confidence_intervals(x_values, width=.95)
        return y_pred, y_int
    else:
        return y_pred

def fit_spline(x_top, x_bot, x_interval, x_values,y_values,n_splines=4,spline_order=2):

    x_series   = np.arange(x_bot, x_top, x_interval)

    # x_values in gridsearch should be of shape [n_samples, m_features], for this case
    # m_features=1, so reshape x_values to [len(x_values),1]
    x_values_array = x_values.to_numpy()
    x_values       = x_values_array.reshape(len(x_values),1)

    tck  = interpolate.splrep(x_values, y_values, s=len(x_series))
    y_pred = interpolate.BSpline(*tck)(x_series)

    return x_series, y_pred

def smooth_vpd_series(values, window_size=11, order=3, smooth_type='S-G_filter',deriv=0):

    """
    Smooth the data using a window with requested size.
    Input:
    array values[nx, ny]

    Type option:
    S-G_filter: Savitzky-Golay filter
    smoothing: mean
    """


    window_half     = int(window_size/2.)

    # if len(np.shape(values))==1:
    nx          = len(values)
    vals_smooth = np.full([nx], np.nan)

    if smooth_type=='S-G_filter':
        vals_smooth = savgol_filter(values, window_size, order, mode='nearest',deriv=deriv)
    elif smooth_type=='rolling':
        if np.sum(~np.isnan(values)) > 0:
            value_tmp   = pd.DataFrame(values, columns=['var'])
            vals_smooth = value_tmp.rolling(window=window_size, min_periods=1).mean().values
        else:
            vals_smooth = values
        print('vals_smooth',vals_smooth)
    elif smooth_type=='smoothing':
        for j in np.arange(window_half,nx-window_half):
            vals_smooth[j] = np.nanmean(values[j-window_half:j+window_half])
        vals_smooth = np.roll(vals_smooth, 10, fill_value='NaN')
    # elif

    # elif len(np.shape(values))==2:
    #     nx              = len(values[:,0])
    #     ny              = len(values[0,:])
    #     vals_smooth     = np.full([nx,ny], np.nan)

    #     for i in np.arange(nx):
    #         if type=='S-G_filter':
    #             vals_smooth[i,:] = savgol_filter(values[i,:], window_size, order)
    #         elif type=='smoothing':
    #             for j in np.arange(window_half,ny-window_half):
    #                 vals_smooth[i,j] = np.nanmean(values[i,j-window_half:j+window_half])

    return vals_smooth

def get_key_words(varname):

    match varname:
        case 'TVeg':
            key_word      = "trans"
            key_word_not  = ["evap","transmission","pedo","electron","stomatal","elasticity"]
        case 'Qle':
            key_word      = 'latent'
            key_word_not  = ['None']
        case 'Qh':
            key_word      = 'sensible'
            key_word_not  = ['vegetation','soil',] # 'corrected'
        case 'Qg':
            key_word      = 'ground'
            key_word_not  = ['radiation', 'snow', 'temperature', 'evaporation', 'water', 'carbon']
        case 'NEE':
            key_word      = 'exchange'
            key_word_not  = ['None']
        case 'GPP':
            key_word      = "gross primary"
            key_word_not  = ['wrt','from']
        case 'Rnet':
            key_word      = "net radiation"
            key_word_not  = ['None']
        case 'SWnet':
            key_word      = "net shortwave radiation"
            key_word_not  = ['None']
        case 'LWnet':
            key_word      = "net longwave radiation"
            key_word_not  = ['None']

    return key_word, key_word_not

def check_variable_exists_in_one_model(PLUMBER2_path, varname, site_name, model_name,
                                       key_word, key_word_not=None):

    # Set input file path
    file_path    = glob.glob(PLUMBER2_path+model_name +"/*"+site_name+"*.nc")
    var_exist    = False
    try:
        with nc.Dataset(file_path[0], 'r') as dataset:
            for var_name in dataset.variables:
                if varname.lower() == var_name.lower():
                    var_name_in_model = var_name
                    var_exist         = True
                elif varname.lower() in var_name.lower():
                    variable  = dataset.variables[var_name]
                    if hasattr(variable, 'long_name'):
                        long_name = variable.long_name.lower()
                        if key_word in long_name and all(not re.search(key_not, long_name) for key_not in key_word_not):
                            var_name_in_model = var_name
                            var_exist          = True
                    else:
                        var_name_in_model = var_name
                        var_exist         = True
                else:
                    variable  = dataset.variables[var_name]

                    # Check whether long_name exists
                    if hasattr(variable, 'long_name'):
                        long_name = variable.long_name.lower()  # Convert description to lowercase for case-insensitive search

                        # Check whether key_word exists
                        # make sure key_word in long_name and all key_word_not are not in key_word_not
                        if key_word in long_name and all(not re.search(key_not, long_name) for key_not in key_word_not):
                            # print(long_name)
                            var_name_in_model = var_name
                            var_exist = True
                            # print(f"The word '{key_word}' is in the description of variable '{var_name}'.")
                            break  # Exit the loop once a variable is found

    except Exception as e:
        print(f"An error occurred: {e}, {site_name}, {model_name}, {file_path}")

    # variable doesn't exist
    if not var_exist:
        var_name_in_model = 'None'

    return var_name_in_model

def check_variable_exists(PLUMBER2_path, varname, site_name, model_names):
    print(varname)
    # get key words
    key_word, key_word_not = get_key_words(varname)

    # file path
    my_dict      = {}

    for j, model_name in enumerate(model_names):

        # print(model_name)

        var_name_in_model = check_variable_exists_in_one_model(PLUMBER2_path, varname, site_name, model_name,
                                                               key_word, key_word_not)
        # if Rnet doesn't exist
        if varname == 'Rnet' and var_name_in_model == 'None':
            SWnet_key_word, SWnet_key_word_not = get_key_words('SWnet')
            LWnet_key_word, LWnet_key_word_not = get_key_words('LWnet')
            SWnet_in_model = check_variable_exists_in_one_model(PLUMBER2_path, 'SWnet', site_name, model_name,
                                                               SWnet_key_word, SWnet_key_word_not)
            LWnet_in_model = check_variable_exists_in_one_model(PLUMBER2_path, 'LWnet', site_name, model_name,
                                                               LWnet_key_word, LWnet_key_word_not)

            if SWnet_in_model == 'SinAng':
                # correct the SWnet var name for ORC models
                SWnet_in_model = 'SWnet'

            # if SWnet and LWnet don't exist the same time
            if SWnet_in_model == 'None' or LWnet_in_model == 'None':
                Qle_key_word, Qle_key_word_not = get_key_words('Qle')
                Qh_key_word, Qh_key_word_not   = get_key_words('Qh')
                Qg_key_word, Qg_key_word_not   = get_key_words('Qg')


                Qle_in_model = check_variable_exists_in_one_model(PLUMBER2_path, 'Qle', site_name, model_name,
                                                                   Qle_key_word, Qle_key_word_not)
                Qh_in_model  = check_variable_exists_in_one_model(PLUMBER2_path, 'Qh', site_name, model_name,
                                                                   Qh_key_word, Qh_key_word_not)
                Qg_in_model  = check_variable_exists_in_one_model(PLUMBER2_path, 'Qg', site_name, model_name,
                                                                   Qg_key_word, Qg_key_word_not)

                # print(Qle_in_model)

                # none of Qle, Qh, Qh exists, then use SWnet or LWnet
                if Qle_in_model == 'None' and Qh_in_model == 'None' and Qg_in_model == 'None':
                    if SWnet_in_model == 'None' and LWnet_in_model ==  'None':
                        my_dict[model_name] = 'None'
                    else:
                        my_dict[model_name] = [SWnet_in_model, LWnet_in_model]
                else: # all of Qle, Qh, Qh exist
                    my_dict[model_name] = [Qle_in_model, Qh_in_model, Qg_in_model]

            else:
                my_dict[model_name] = [SWnet_in_model, LWnet_in_model]

        else:
            my_dict[model_name] = var_name_in_model

    if varname == 'Qg':
        # manually set 'hfdsn': heat flux into soil/snow including snow melt and lake / snow light transmission
        # as CLM5a's Qg
        my_dict['CLM5a'] = 'hfdsn'

    return my_dict

def check_variable_units(PLUMBER2_path, varname, site_name, model_names, key_word, key_word_not=None):

    # file path
    my_dict      = {}

    for j, model_name in enumerate(model_names):
        # print(model_name)

        # Set input file path
        file_path    = glob.glob(PLUMBER2_path+model_name +"/*"+site_name+"*.nc")
        var_exist    = False
        try:
            with nc.Dataset(file_path[0], 'r') as dataset:
                for var_name in dataset.variables:
                    # print(var_name)
                    if varname.lower() in var_name.lower():
                        my_dict[model_name] = dataset.variables[var_name].units
                        var_exist = True
                    else:
                        variable  = dataset.variables[var_name]

                        # Check whether long_name exists
                        if hasattr(variable, 'long_name'):
                            long_name = variable.long_name.lower()  # Convert description to lowercase for case-insensitive search

                            # Check whether key_word exists
                            # make sure key_word in long_name and all key_word_not are not in key_word_not
                            if key_word in long_name and all(not re.search(key_not, long_name) for key_not in key_word_not):
                                # print(long_name)
                                my_dict[model_name] = dataset.variables[var_name].units
                                var_exist = True
                                # print(f"The word '{key_word}' is in the description of variable '{var_name}'.")
                                break  # Exit the loop once a variable is found
        except Exception as e:
            print(f"An error occurred: {e}")

        # variable doesn't exist
        if not var_exist:
            my_dict[model_name] = 'None'

    return my_dict

def read_lat_lon(site_names, PLUMBER2_met_path):

    # file path
    lat_dict      = {}
    lon_dict      = {}

    for site_name in site_names:


        # Set input file path
        file_path    = glob.glob(PLUMBER2_met_path+"/*"+site_name+"*.nc")

        f            = nc.Dataset(file_path[0], mode='r')
        lat_tmp      = f.variables['latitude'][0,0].data
        lon_tmp      = f.variables['longitude'][0,0].data

        # convert the array to a floating-point number
        lat_dict[site_name] = float(lat_tmp)
        lon_dict[site_name] = float(lon_tmp)
        f.close()

    # print('lat_dict',lat_dict)
    # print('lon_dict',lon_dict)

    return lat_dict, lon_dict

def read_IGBP_veg_type(site_names, PLUMBER2_met_path):

    # file path
    IGBP_dict      = {}

    for site_name in site_names:

        # print(site_name)

        # Set input file path
        file_path    = glob.glob(PLUMBER2_met_path+"/*"+site_name+"*.nc")

        f            = nc.Dataset(file_path[0], mode='r')

        # Decode the string as Unicode
        IGBP_tmp     = codecs.decode(f.variables['IGBP_veg_short'][:].data, 'utf-8')

        # Remove spaces from the beginning and end of the string
        IGBP_dict[site_name] = IGBP_tmp.strip()

        f.close()

    # print('IGBP_dict',IGBP_dict)

    return IGBP_dict

def read_climate_class(lat, lon):

    """
    Read the climate_class value of the nearest pixel to lat and lon.

    Parameters:
        nc_file (str): The path to the netCDF4 file.
        lat (float): The latitude of the site.
        lon (float): The longitude of the site.

    Returns:
        int: The climate_class value of the nearest pixel.
    """
    climate_class_path = '/g/data/w97/mm3972/data/Köppen-Geiger_climate_classification/Beck_KG_V1/Beck_KG_V1_present_0p0083.nc'
    f                  = nc.Dataset(climate_class_path)

    latitude  = f.variables['latitude'][:]
    longitude = f.variables['longitude'][:]
    # print('latitude',latitude)
    # print('longitude',longitude)
    # Find the indices of the nearest pixels to lat and lon.
    lat_idx = np.argmin(np.abs(latitude - lat))
    lon_idx = np.argmin(np.abs(longitude - lon))

    # Read the climate_class value of the nearest pixel.
    # print('lat_idx, lon_idx',lat_idx, lon_idx)
    climate_class = f.variables['climate_class'][lat_idx, lon_idx]
    class_name = f.variables['class_name'][:]
    # print('climate_class',climate_class)
    # print('class_name',class_name[int(climate_class)-1])

    return class_name[int(climate_class)-1]

def regrid_data(lat_in, lon_in, lat_out, lon_out, input_data, method='linear',threshold=None):

    if len(np.shape(lat_in)) == 1:
        lon_in_2D, lat_in_2D = np.meshgrid(lon_in,lat_in)
        lon_in_1D            = np.reshape(lon_in_2D,-1)
        lat_in_1D            = np.reshape(lat_in_2D,-1)
    elif len(np.shape(lat_in)) == 2:
        lon_in_1D            = np.reshape(lon_in,-1)
        lat_in_1D            = np.reshape(lat_in,-1)
    else:
        print("ERROR: lon_in has ", len(np.shape(lat_in)), "dimensions")

    if len(np.shape(lat_out)) == 1:
        lon_out_2D, lat_out_2D = np.meshgrid(lon_out,lat_out)
    elif len(np.shape(lat_out)) == 2:
        lon_out_2D            = lon_out
        lat_out_2D            = lat_out
    else:
        print("ERROR: lon_out has ", len(np.shape(lat_in)), "dimensions")

    value_tmp = np.reshape(input_data,-1)

    # Check NaN - input array shouldn't have NaN
    if threshold is None:
        mask_values     = ~np.isnan(value_tmp)
    else:
        #print("np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0) ",np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0))
        mask_values     = np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0)

    value     = value_tmp[mask_values]
    # ======= CAUTION =======
    lat_in_1D = lat_in_1D[mask_values]  # here I make nan in values as the standard
    lon_in_1D = lon_in_1D[mask_values]
    #print("shape value = ", np.shape(value))
    #print("shape lat_in_1D = ", np.shape(lat_in_1D))
    #print("shape lon_in_1D = ", np.shape(lon_in_1D))
    # =======================
    #print("value =",value)
    #print("lat_in_1D =",lat_in_1D)
    #print("lon_in_1D =",lon_in_1D)
    Value = griddata((lon_in_1D, lat_in_1D), value, (lon_out_2D, lat_out_2D), method=method)

    return Value

def set_IGBP_colors():

    # file path
    IGBP_colors = {
                    'CRO':'gold',
                    'GRA':'yellowgreen',
                    'EBF':'lightgreen',
                    'ENF':'forestgreen',
                    'MF':'deepskyblue',
                    'DBF':'aquamarine',#'brown',
                    'OSH':'red',
                    'CSH':'coral',
                    'SAV':'pink',
                    'WSA':'violet',
                    'WET':'blue',
                    }

    return IGBP_colors

def set_clim_colors():

    # file path
    clim_colors = {
                    # A: Tropical
                    'Af': 'forestgreen',      # Tropical rainforest climate
                    'Am': 'darkolivegreen',   # Tropical monsoon climate
                    'Aw': 'lightgreen',       # Tropical savanna climate
                    # B: Dry red
                    'BWh': 'red',  # Hot deserts
                    'BWk': 'violet' ,  # Cold deserts
                    'BSh': 'hotpink',  # Hot semi-arid
                    'BSk': 'pink',  # Cold semi-arid
                    # C: Temperate blue
                    'Csa': 'peru', # Hot-summer Mediterranean climates
                    'Csb': 'tan', # Warm-summer Mediterranean climates
                    'Cwa': 'orange', # Dry-winter subtropical climates
                    'Cfa': 'gold', # Humid subtropical climates
                    'Cfb': 'yellow', # Oceanic climates
                    # D: Continental brown/yellow
                    'Dsb': 'c',  # Mediterranean-influenced warm-summer humid continental climate
                    'Dwa': 'aqua', # Monsoon-influenced hot-summer humid continental climate
                    'Dwb': 'deepskyblue',  # Monsoon-influenced warm-summer humid continental climate
                    'Dfa': 'blue',  # Hot-summer humid continental climate
                    'Dfb': 'dodgerblue',  # Warm-summer humid continental climate
                    'Dfc': 'royalblue',  # Subarctic climate
                    'Dsc': 'Navy', # cold, dry summer climate with cool summers
                    # E: Polar
                    'ET': 'gray' # Tundra climate;
                    }

    return clim_colors

def set_model_colors_Gs_based():

    # file path
    model_colors = {
                    'obs': 'black',
                    'CMIP6':'firebrick',
                    'CABLE':'deepskyblue', # Medlyn
                    'CABLE-POP-CN':'deepskyblue', # Medlyn
                    'CHTESSEL_ERA5_3':'red', # No Gs model
                    'CHTESSEL_Ref_exp1':'red', # No Gs model
                    'CLM5a':'deepskyblue', # Medlyn
                    'GFDL':'lightseagreen', # Wolf
                    'JULES_GL9':'firebrick', # Jacobs
                    'JULES_GL9_withLAI':'firebrick', # Jacobs
                    'JULES_test':'deepskyblue', # Medlyn
                    'LPJ-GUESS':'yellow', # Collatz
                    'MATSIRO':'gold',# Ball-Berry
                    'MuSICA':'green', # Leuning
                    'NASAEnt':'gold', # Ball-Berry
                    'NoahMPv401':'gold', # Ball-Berry
                    'ORC2_r6593':'pink', # Yin and Struik
                    'ORC2_r6593_CO2':'gold', # Ball-Berry
                    'ORC3_r7245_NEE':'pink', # Yin and Struik
                    'ORC3_r8120':'pink', # Yin and Struik
                    'QUINCY':'gold', # Ball-Berry
                    'SDGVM':'deepskyblue' , # Medlyn
                    'STEMMUS-SCOPE':'grey' , # unknown
                    }

    return model_colors

def set_model_colors():

    # file path
    model_colors = {
                    'obs': 'black',
                    'CMIP6':'firebrick',
                    'CABLE':'darkblue',
                    'CABLE-POP-CN':'blue',
                    'CHTESSEL_ERA5_3':'coral',
                    'CHTESSEL_Ref_exp1':'cornflowerblue',
                    'CLM5a':'deepskyblue',
                    'GFDL':'c',
                    'JULES_GL9':'aquamarine',
                    'JULES_GL9_withLAI':'yellowgreen',
                    'JULES_test':'forestgreen',
                    'LPJ-GUESS':'darkolivegreen',
                    'MATSIRO':'forestgreen',
                    'MuSICA':'lime',
                    'NASAEnt':'gold' , # 'yellow'
                    'NoahMPv401':'orange' ,
                    'ORC2_r6593':'pink',#'limegreen'
                    'ORC2_r6593_CO2':'pink',
                    'ORC3_r7245_NEE':'red',
                    'ORC3_r8120':'deeppink',
                    'QUINCY':'mediumorchid',
                    'SDGVM': 'darkviolet',
                    'STEMMUS-SCOPE':'purple',# ,
                    }

    # model_colors = {
    #                 'obs': 'black',
    #                 'obs_cor': 'dimgrey',
    #                 '1lin': 'lightcoral' ,
    #                 '3km27': 'indianred',
    #                 '6km729': 'firebrick',
    #                 '6km729lag':'red',
    #                 'LSTM_eb': 'coral',
    #                 'LSTM_raw': 'pink',
    #                 'RF_eb': 'tomato',
    #                 'RF_raw': 'deeppink',
    #                 'Manabe':'violet',
    #                 'ManabeV2':'darkviolet',
    #                 'PenmanMonteith': 'purple',
    #                 'CABLE':'darkblue',
    #                 'CABLE-POP-CN':'blue',
    #                 'CHTESSEL_ERA5_3':'cornflowerblue',
    #                 'CHTESSEL_Ref_exp1':'dodgerblue',
    #                 'CLM5a':'deepskyblue',
    #                 'GFDL':'c',
    #                 'JULES_GL9_withLAI':'aquamarine',
    #                 'JULES_test':'lightseagreen',
    #                 'MATSIRO':'darkcyan',
    #                 'NoahMPv401':'darkolivegreen',
    #                 'ORC2_r6593':'forestgreen',
    #                 'ORC2_r6593_CO2':'limegreen',
    #                 'ORC3_r7245_NEE':'lime',
    #                 'ORC3_r8120':'lightgreen',
    #                 'STEMMUS-SCOPE':'yellowgreen',
    #                 'ACASA':'yellow',
    #                 'LPJ-GUESS':'orange',
    #                 'MuSICA':'gold',
    #                 'NASAEnt': 'goldenrod',
    #                 'QUINCY':'peru',
    #                 'SDGVM':'sandybrown',
    #                 }


    return model_colors

def conduct_quality_control(varname, data_input,zscore_threshold=2, gap_fill='nan'):

    '''
    Please notice EF has nan values
    '''

    z_scores    = np.abs(stats.zscore(data_input, nan_policy='omit'))
    data_output = np.where(z_scores > zscore_threshold, np.nan, data_input)

    if gap_fill=='interpolate':

        print('Gap filling by linear interpolation')

        if 'EF' not in varname:

            # Iterate through the data to replace NaN with the average of nearby non-NaN values
            for i in range(1, len(data_output) - 1):
                if np.isnan(data_output[i]):
                    prev_index = i - 1
                    next_index = i + 1

                    # find the closest non nan values
                    while prev_index >= 0 and np.isnan(data_output[prev_index]):
                        prev_index -= 1

                    while next_index < len(data_output) and np.isnan(data_output[next_index]):
                        next_index += 1

                    # use average them
                    if prev_index >= 0 and next_index < len(data_output):
                        prev_non_nan = data_output[prev_index]
                        next_non_nan = data_output[next_index]
                        data_output[i] = (prev_non_nan + next_non_nan) / 2.0

        return data_output

    elif gap_fill=='nan':
        print('Gap filling by NaN values')
        return data_output

def convert_into_kg_m2_s(data_input, var_units):

    d_2_s = 24*60*60
    if 'W' in var_units and 'm' in var_units and '2' in var_units:
        print('converting ', var_units)
        data_output = data_input * 86400 / 2454000 /d_2_s
    return data_output

def convert_from_umol_m2_s_into_gC_m2_s(data_input, var_units):

    # convert from umol/m2/s to gC/m2/s
    umol_2_mol  = 0.000001
    mol_2_gC    = 12
    print('converting ', var_units)
    data_output = data_input*umol_2_mol*mol_2_gC

    return data_output

def get_model_soil_moisture_info(site_name):

    '''
    CLM5: Increased soil vertical resolution (20 soil layers + 5 bedrock layers), thus the last 5 layers
        are bedrock layers which don't have soil moisture data. [The Community Land Model Version 5: Description
        of New Features, Benchmarking, and Impact of Forcing Uncertainty]
    MATSIRO: As the default setting, the soil has six layers whose thicknesses are defined by the depth boundaries
        of 5, 20, 75, 100, 200, and 1000 cm from the surface. [Description of MATSIRO6]
        However, I communicated with Qiang Guo, who worked in Tokyo University, he told the thickness are
        0.05, 0.20, 0.75, 1., 2., 8 meter.
    '''
    # Set models with simulated soil mositure
    SM_names      = { 'CABLE':'SoilMoist',
                      'CABLE-POP-CN':'SoilMoist',
                    #   'CHTESSEL_ERA5_3':'SoilMoist',
                      'CHTESSEL_Ref_exp1':'SoilMoist',
                      'CLM5a':'mrlsl',
                      'GFDL':'SoilMoist',
                      'JULES_GL9':'SoilMoist',
                      'JULES_GL9_withLAI':'SoilMoist',
                    #   'JULES_test':'SoilMoist',
                      'MATSIRO':'SoilMoistV',
                      'MuSICA':'SoilMoist',
                      'NoahMPv401':'SoilMoist',
                      'ORC2_r6593':'SoilMoist',
                    #   'ORC3_r7245_NEE':'SoilMoist',
                      'ORC3_r8120':'SoilMoist',
                      'STEMMUS-SCOPE':'SoilMoist'} #  'SDGVM':'RootMoist',

    # get soil thickness in MuSICA model
    MuSICA_path = glob.glob(f"/g/data/w97/mm3972/data/PLUMBER2/MuSICA/{site_name}*.nc")

    if MuSICA_path:
        f                = nc.Dataset(MuSICA_path[0])
        MuSICA_thickness = f.variables['dz_soil'][:]
        MuSICA_thickness = MuSICA_thickness.tolist()
    else:
        MuSICA_thickness = [np.nan]

    # Set soil layer thickness
    soil_thicknesses = {'CABLE':[0.022,0.058, 0.154, 0.409, 1.085, 2.872],
                        'CABLE-POP-CN':[0.022,0.058, 0.154, 0.409, 1.085, 2.872],
                        'CHTESSEL_ERA5_3':[0.07, 0.21, 0.72, 1.89],
                        'CHTESSEL_Ref_exp1':[0.07, 0.21, 0.72, 1.89],
                        'JULES_test': [0.1, 0.25, 0.65, 2.0],
                        'JULES_GL9': [0.1, 0.25, 0.65, 2.0],
                        'JULES_GL9_withLAI': [0.1, 0.25, 0.65, 2.0],
                        'MATSIRO': [0.05, 0.20, 0.75, 1., 2., 8],
                        'NoahMPv401': [0.10, 0.30, 0.60,1.],
                        'CLM5a':[0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4,
                                 0.44, 0.54, 0.64, 0.74, 0.84, 0.94, 1.04, 1.14, 2.39, 4.675534, 7.63519,
                                 11.14, 15.11543],
                        'GFDL': [0.02,0.04,0.04,0.05,0.05,0.1,0.1,0.2,0.2,0.2,
                                  0.4,0.4,0.4,0.4,0.4,1,1,1,1.5,2.5],
                        'MuSICA': MuSICA_thickness,
                        'ORC2_r6593':[0.0009775171, 0.003910068, 0.009775171, 0.02150538, 0.04496579,
                                      0.09188661, 0.1857283, 0.3734115, 0.7487781, 1.499511, 2],
                        'ORC3_r7245_NEE':[0.0009775171, 0.003910068, 0.009775171, 0.02150538, 0.04496579,
                                          0.09188661, 0.1857283, 0.3734115, 0.7487781, 1.499511, 2],
                        'ORC3_r8120':[0.0009775171, 0.003910068, 0.009775171, 0.02150538, 0.04496579,
                                      0.09188661, 0.1857283, 0.3734115, 0.7487781, 1.499511, 2],
                        'STEMMUS-SCOPE':[ 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                                        0.02, 0.02, 0.02, 0.02, 0.02, 0.025, 0.025, 0.025, 0.025, 0.05, 0.05,
                                        0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
                                        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15,
                                        0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]}

    return SM_names, soil_thicknesses

def bootstrap_ci( data, statfunction=np.average, alpha = 0.05, n_samples = 100):

    """
    Purpose calculate confidence interval by bootstrap method
    Code source: https://stackoverflow.com/questions/44392978/compute-a-confidence-interval-from-sample-data-assuming-unknown-distribution/66008548#66008548
    """

    import warnings

    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0],))

    alphas = np.array([alpha/2, 1 - alpha/2])
    nvals  = np.round((n_samples - 1) * alphas).astype(int)
    if np.any(nvals < 10) or np.any(nvals >= n_samples-10):
        warnings.warn("Some values used extremal samples; results are probably unstable. "
                      "Try to increase n_samples")

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()

    boot_indexes = bootstrap_ids(data, n_samples)
    # print('boot_indexes',boot_indexes)
    stat         = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    # print('stat',stat)
    stat.sort(axis=0)

    return stat[nvals]
