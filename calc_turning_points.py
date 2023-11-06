import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
import pwlf
from kneed import KneeLocator
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *
from plot_script import *
from PLUMBER2_VPD_common_utils import *

def calc_derivative(model_out_list, vpd_series, values, option='S-G_filter', window_size=11, order=3):

    derivative = {}

    if option == 'manually':
        nmodel          = len(model_out_list)
        vpd_series_len  = len(vpd_series)
        derivative_vals = np.full([nmodel,vpd_series_len], np.nan)
        vpd_interval    = vpd_series[1]-vpd_series[0]

        for i,model_out_name in enumerate(model_out_list):
            for j in np.arange(1,vpd_series_len-1):
                derivative_vals[i,j] = (values[i,j+1]-values[i,j-1])/(2*vpd_interval)
            derivative[model_out_name] = derivative_vals[i,:]
    elif option == 'S-G_filter':
        for i,model_out_name in enumerate(model_out_list):
            derivative[model_out_name] = savgol_filter(values[i,:], window_size, order, deriv=1, mode='nearest') #  deriv=1: calc 1-order derivative

    return derivative

def find_turning_points_by_gradient(model_out_list, vpd_series, vals_smooth, threshold=0.01, smooth_option='S-G_filter', smooth_window_size=11, smooth_order=3):

    # Method inspired by
    #   https://www.appsloveworld.com/python/788/how-to-detect-a-turning-point-of-a-graph-with-python
    #   https://blog.csdn.net/u012005313/article/details/84035371
    # Method
    #   When the increasing gradient lower than a threshold then think it is at the turning point

    nmodel          = len(model_out_list)
    vpd_series_len  = len(vpd_series)

    max_vals        = np.zeros(len(model_out_list))
    max_loc_index   = np.zeros(len(model_out_list))
    max_loc_vpd     = np.zeros(len(model_out_list))

    derivative      = calc_derivative(model_out_list, vpd_series, vals_smooth,
                                      option=smooth_option, window_size=smooth_window_size,
                                      order=smooth_order)

    for i, model_out_name in enumerate(model_out_list):

        derivative_tmp   = derivative[model_out_name]

        print(model_out_name, 'derivative_tmp',derivative_tmp)

        try:
            max_loc_index[i] = np.nanargmax(derivative_tmp < threshold)
            max_vals[i]      = vals_smooth[model_out_name][index]
            max_loc_vpd[i]   = vpd_series[index]
        except:
            max_loc_index[i] = np.nan
            max_vals[i]      = 0.
            max_loc_vpd[i]   = -9999.

    turning_points = {}

    for i, model_out_name in enumerate(model_out_list):
        turning_points[model_out_name] = [max_loc_vpd[i],max_vals[i]]

    return turning_points

def find_turning_points_by_max_peaks(model_out_list, vpd_series, vals_smooth, smooth_option='S-G_filter', smooth_window_size=11, smooth_order=3):

    # Method
    #   Find the turning points and find the max value in these turning points as the one for
    #   this curve
    # output: peak_values[nmodel]

    nmodel          = len(model_out_list)
    vpd_series_len  = len(vpd_series)

    peak_values     = np.full([nmodel,vpd_series_len], np.nan)
    tmp             = np.full(vpd_series_len, np.nan)

    derivative      = calc_derivative(model_out_list, vpd_series, vals_smooth,
                                      option=smooth_option, window_size=smooth_window_size,
                                      order=smooth_order)

    for i, model_out_name in enumerate(model_out_list):
        derivative_tmp = derivative[model_out_name]
        for j in np.arange(0,vpd_series_len-1):
            # if they are not the beginning and ending points
            if (~ np.isnan(derivative_tmp[j])) and (~ np.isnan(derivative_tmp[j+1])):
                # if it is maximum
                if derivative_tmp[j] > 0 and derivative_tmp[j+1] < 0:
                    if abs(derivative_tmp[j]) < abs(derivative_tmp[j+1]):
                        tmp[j] = vpd_series[j]
                    else:
                        tmp[j] = vpd_series[j+1]
                # if it is minimum
            elif derivative_tmp[j] < 0 and derivative_tmp[j+1] > 0:
                    if abs(derivative_tmp[j]) < abs(derivative_tmp[j+1]):
                        tmp[j] = vpd_series[j]*(-1)
                    else:
                        tmp[j] = vpd_series[j+1]*(-1)

        peak_values[i,:] = tmp

    # only keep the value at the turning points
    vals_smooth    = np.where(~np.isnan(peak_values), vals_smooth, np.nan)

    print('np.any(~np.isnan(vals_smooth))',np.any(~np.isnan(vals_smooth)))

    max_vals      = np.zeros(len(model_out_list))
    max_loc_index = np.zeros(len(model_out_list))
    max_loc_vpd   = np.zeros(len(model_out_list))

    for i, model_out_name in enumerate(model_out_list):
        max_vals[i]      = np.nanmax(vals_smooth[i,:])
        try:
            max_loc_index[i] = np.nanargmax(vals_smooth[i,:])
        except:
            max_loc_index[i] = np.nan

    for i, index in enumerate(max_loc_index):
        if np.isnan(index):
            max_loc_vpd[i] = -9999.
        else:
            max_loc_vpd[i] = vpd_series[index]

    print('max_vals',max_vals)
    print('max_loc_vpd',max_loc_vpd)

    turning_points = {}
    for i, model_out_name in enumerate(model_out_list):
        turning_points[model_out_name] = [max_loc_vpd[i],max_vals[i]]

    print('turning_points',turning_points)
    return turning_points

def find_turning_points_by_kneed(model_out_list, vpd_series, vals_smooth):

    # Method source:  https://kneed.readthedocs.io/en/latest/parameters.html#curve
    # Parameters:
    # S: The sensitivity parameter allows us to adjust how aggressive we want Kneedle to
    #    be when detecting knees. Smaller values for S detect knees quicker, while larger
    #    values are more conservative. Put simply, S is a measure of how many “flat” points
    #    we expect to see in the unmodified data curve before declaring a knee

    turning_points = {}

    for i, model_out_name in enumerate(model_out_list):

        kneedle = KneeLocator(vpd_series, vals_smooth[i,:],
                              S=11.0, curve="concave", direction="increasing",online=False,
                              interp_method='interp1d') #interp_method="polynomial",polynomial_degree=2)#

        turning_points[model_out_name] = [kneedle.knee, kneedle.knee_y]

    print('turning_points',turning_points)

    return turning_points

def find_turning_points_by_cdf(model_out_list, vpd_series, vals_smooth):

    turning_points = {}

    for i, model_out_name in enumerate(model_out_list):

        # Calculate the cumulative distribution function (CDF)
        cdf = np.cumsum(vals_smooth[i,:])

        # Calculate the 2-sided moving average of the CDF
        moving_mean = pd.Series(cdf).rolling(window=3, center=True).mean(fill_value=np.nan)

        # Calculate the differences between the moving average values
        diff = moving_mean.diff()

        # Identify the index of the maximum difference
        idxmin = diff.idxmin()
        print('idxmin',idxmin)

        if idxmin>0:
            turning_points[model_out_name] = [vpd_series[idxmin], vals_smooth[i,idxmin]]
        else:
            turning_points[model_out_name] = [np.nan, np.nan]

        fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[5,5],sharex=True, sharey=False, squeeze=True) #
        ax.plot(vpd_series,vals_smooth[i,:],c='orange')
        ax.plot(vpd_series,cdf,c='red')
        ax.plot(vpd_series,moving_mean,c='blue')
        ax.plot(vpd_series,diff,c='black')
        ax.scatter(turning_points[model_out_name][0],turning_points[model_out_name][1],c='orange')

        fig.savefig("./plots/check_find_turning_points_by_cdf_"+model_out_name+".png",bbox_inches='tight',dpi=300) # '_30percent'

    print('turning_points',turning_points)
    return turning_points

def find_turning_points_by_piecewise_regression(model_out_list, vpd_series, vals_smooth, var_name, piece_num=5):

    # Example : https://www.5axxw.com/questions/simple/nl9brn


    turning_points = {}
    slopes         = {}

    for i, model_out_name in enumerate(model_out_list):
        print('model_out_name',model_out_name)
        if var_name == 'NEE':
            if model_out_name in ['GFDL','NoahMPv401','STEMMUS-SCOPE','ACASA']:
                print(model_out_name,',vals_smooth[i,:]=vals_smooth[i,:]')
            else:
                print(model_out_name,',vals_smooth[i,:]=vals_smooth[i,:] * (-1)')
                vals_smooth[i,:] = vals_smooth[i,:] * (-1)

        values       = vals_smooth[i,:]
        vpds         = np.copy(vpd_series)

        not_nan_mask = (~np.isnan(values)) & (~np.isnan(vpds))
        values       = values[not_nan_mask]
        vpds         = vpds[not_nan_mask]

        if len(values)>5:
            print('len(values)',len(values))
            # Create a PiecewiseLinFit object.
            my_pwlf = pwlf.PiecewiseLinFit(vpds, values,disp_res=True)

            # Fit a piecewise linear function to the data with 3 breakpoints.
            my_pwlf.fit(piece_num)

            # Calculate the slopes of the lines.
            piecewise_slopes = my_pwlf.slopes
            piecewise_breaks = my_pwlf.fit_breaks

            # Select the break points I want to plot
            slope2      = piecewise_slopes[1:]*piecewise_slopes[0:-1]
            slope_index = np.argmin(slope2)
            print('slope2',slope2,'slope_index',slope_index)


            # if slope2[slope_index]<0:
            # Find the index of the element with the smallest absolute difference.
            vpd_diff      = np.abs(vpd_series - piecewise_breaks[slope_index+1])
            closest_index = np.argmin(vpd_diff)

            # Calculate the breaking points.
            turning_points[model_out_name]  = [vpd_series[closest_index],  vals_smooth[i,closest_index]]
            # turning_points[model_out_name]  = [piecewise_breaks[1], my_pwlf.predict(piecewise_breaks[1])[0]]
            slopes[model_out_name]          = [piecewise_slopes[0],piecewise_slopes[1]]
            # else:
            #     turning_points[model_out_name] = [-9999., np.nan]
            #     slopes[model_out_name]          = [np.nan,np.nan]
        else:
            turning_points[model_out_name] = [-9999., np.nan]
            slopes[model_out_name]          = [np.nan,np.nan]

    # print('turning_points',turning_points)
    # print('slopes',slopes)

    return turning_points, slopes


def single_plot_lines(turning_points, model_out_list, x_values, y_values ,message=None):

    # ============ Setting for plotting ============
    cmap     = plt.cm.rainbow #YlOrBr #coolwarm_r

    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[8,6],sharex=True, sharey=False, squeeze=True) #

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

    for i, model_out_name in enumerate(model_out_list):

        line_color = model_colors[model_out_name]
        plot       = ax.plot(x_values, y_values[i,:], lw=2.0, color=line_color, alpha=0.7, label=model_out_name)
        plot       = ax.scatter(turning_points[model_out_name][0],turning_points[model_out_name][1], marker='o', color=line_color, s=10)
        plot       = ax.axhline(y=0.0, color='black', linestyle='-.', linewidth=1)

    ax.set_xlim(0, 7.)
    ax.set_ylim(0, 200)

    ax.legend(fontsize=6, frameon=False, ncol=3)

    fig.savefig("./plots/check_single_plot_lines_"+message+'.png',bbox_inches='tight',dpi=300) # '_30percent'

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    # The site names
    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    # site_names     = ["AU-Tum"]

    print(site_names)

    var_name       = 'Qle'  #'TVeg'
    bin_by         = 'EF_model' #'EF_model' #'EF_model' #
    IGBP_types     = ['CRO', 'CSH', 'DBF', 'EBF','EBF', 'ENF', 'GRA', 'MF', 'OSH', 'WET', 'WSA', 'SAV']
    clim_types     = ['Af', 'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Csa', 'Csb', 'Cwa',
                      'Dfa', 'Dfb', 'Dfc', 'Dsb', 'Dsc', 'Dwa', 'Dwb', 'ET']

    day_time       = True
    energy_cor     = True
    low_bound      = [0,0.2]#[0.8,1.0]
    method         = 'bin_by_vpd' #'GAM'

    if var_name == 'NEE':
        energy_cor     = False

    peak_values = {}

    # for IGBP_type in IGBP_types:

    # ============== read data ==============
    message      = ''

    if day_time:
        message  = message + '_daytime'

    try:
        message  = message + '_IGBP='+IGBP_type
    except:
        print(' ')

    try:
        message  = message + '_clim='+clim_type
    except:
        print(' ')

    # ================= Read in csv file =================
    if len(low_bound) >1:
        if low_bound[1] > 1:
            var_bin_by_VPD = pd.read_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'th_'+method+'.csv')
        else:
            var_bin_by_VPD = pd.read_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound[0])+'-'+str(low_bound[1])+'_'+method+'.csv')
    elif len(low_bound) == 1:
        if low_bound > 1:
            var_bin_by_VPD = pd.read_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound)+'th_'+method+'.csv')
        else:
            var_bin_by_VPD = pd.read_csv(f'./txt/{var_name}_VPD'+message+'_'+bin_by+'_'+str(low_bound)+'_'+method+'.csv')

    print('var_bin_by_VPD',var_bin_by_VPD)

    # Get model namelists
    model_out_list = []
    for column_name in var_bin_by_VPD.columns:
        if "_vals" in column_name:
            model_out_list.append(column_name.split("_vals")[0])

    vpd_series      = var_bin_by_VPD['vpd_series']

    nmodel          = len(model_out_list)
    nvpd            = len(vpd_series)

    # ================= Smoothing =================
    window_size = 11
    order       = 3
    smooth_type = 'S-G_filter'

    # Smoothing the curve and remove vpd_num < 100.
    vals_smooth = np.zeros((nmodel,nvpd))

    for i, model_out_name in enumerate(model_out_list):

        vals_smooth_tmp = smooth_vpd_series(var_bin_by_VPD[model_out_name+'_vals'],
                            window_size=window_size, order=order,
                            smooth_type=smooth_type)
        vals_vpd_num    = var_bin_by_VPD[model_out_name+'_vpd_num']

        vals_smooth[i,:]= np.where(vals_vpd_num>100,vals_smooth_tmp,np.nan)

    # Find the turning points
    turning_points = find_turning_points_by_kneed(model_out_list, vpd_series, vals_smooth)

    # turning_points = find_turning_points_by_gradient(model_out_list, vpd_series, vals_smooth, derivative, threshold=1)
    # turning_points = find_turning_points_by_max_peaks(model_out_list, vpd_series, vals_smooth, derivative)

    if len(low_bound) >1:
        if low_bound[1] > 1:
            message = var_name+'_VPD'+message+'_EF_'+str(low_bound[0])+'-'+str(low_bound[1])+'th'
        else:
            message = var_name+'_VPD'+message+'_EF_'+str(low_bound[0])+'-'+str(low_bound[1])
    elif len(low_bound) == 1:
        if low_bound > 1:
            message = var_name+'_VPD'+message+'_EF_'+str(low_bound)+'th'
        else:
            message = var_name+'_VPD'+message+'_EF_'+str(low_bound)

    single_plot_lines(turning_points, model_out_list, x_values=vpd_series, y_values=vals_smooth, message=message)
