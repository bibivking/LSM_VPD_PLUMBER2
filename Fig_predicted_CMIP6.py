'''
Including:
    def plot_var_VPD_uncertainty
    def plot_var_VPD_line_box
    def plot_var_VPD_line_box_three_cols
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
import seaborn as sns
import pandas as pd
import netCDF4 as nc
from matplotlib.patches import Polygon
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from PLUMBER2_VPD_common_utils import *

def plot_predicted_CMIP6_boxplot(CMIP6_txt_path, var_name, model_list, scenarios,
                                 region={'name':'global','lat':None, 'lon':None}, dist_type=None):

    # ============ Setting up plot ============
    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[12,6],sharex=False, sharey=False, squeeze=True)

    plt.subplots_adjust(wspace=0.09, hspace=0.02)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # read all metrics files
    for i, model_in in enumerate(model_list):

        # set line color
        line_color = model_colors[model_in]

        for scenario in scenarios:

            if dist_type!=None:
                metrics  = pd.read_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_{var_name}_{scenario}_{model_in}_{region["name"]}_{dist_type}.csv', na_values=[''])
            else:
                metrics  = pd.read_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_{var_name}_{scenario}_{model_in}_{region["name"]}.csv', na_values=[''])
            mean     = metrics.loc[0,model_in]
            p25      = metrics.loc[1,model_in]
            p75      = metrics.loc[2,model_in]
            min      = metrics.loc[3,model_in]
            max      = metrics.loc[4,model_in]

            # mean_east_AU  = pd.read_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_{var_name}_{scenario}_{model_in}_east_AU.csv', na_values=['']).loc[0,model_in]
            # mean_west_EU  = pd.read_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_{var_name}_{scenario}_{model_in}_west_EU.csv', na_values=['']).loc[0,model_in]
            # mean_north_Am = pd.read_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_{var_name}_{scenario}_{model_in}_north_Am.csv', na_values=['']).loc[0,model_in]

            # print('mean,p25,p75,min,max',mean,p25,p75,min,max)

            # Set x-tick
            if scenario == 'historical':
                xaxis_s = 0.5+i*3
                xaxis_e = 1.5+i*3
                alpha   = 1.
                mean_hist = mean
                p25_hist = p25
                p75_hist = p75
            elif scenario == 'ssp245':
                xaxis_s = 1.5+i*3
                xaxis_e = 2.5+i*3
                alpha   = 0.65
                mean_diff = mean-mean_hist
                p25_diff = p25 - p25_hist
                p75_diff = p75 - p75_hist

            # Draw the box
            # print([xaxis_s, p25], [xaxis_s, p75], [xaxis_e, p75], [xaxis_e, p25])
            ax.add_patch(Polygon([[xaxis_s, p25], [xaxis_s, p75],
                                  [xaxis_e, p75], [xaxis_e, p25]],
                                closed=True, color=line_color, fill=True, alpha=alpha, linewidth=0.1))

            # Draw the mean line
            # if scenario == 'historical' and model_in == 'obs':
            #     ax.plot([xaxis_s,xaxis_e], [mean,mean], color = 'white', linewidth=0.5)
            # else:
            #     ax.plot([xaxis_s,xaxis_e], [mean,mean], color = almost_black, linewidth=0.5)
            ax.plot([xaxis_s,xaxis_e], [mean,mean], color = 'white', linewidth=1.)


            # Draw the p25 p75
            ax.plot([xaxis_s, xaxis_e], [p25, p25], color = almost_black, linewidth=1.)
            ax.plot([xaxis_s, xaxis_e], [p75, p75], color = almost_black, linewidth=1.)

            ax.plot([xaxis_s, xaxis_s], [p25, p75], color = almost_black, linewidth=1.)
            ax.plot([xaxis_e, xaxis_e], [p25, p75], color = almost_black, linewidth=1.)

            # Draw the max and min
            # ax.plot([xaxis_s+0.1, xaxis_e-0.1], [min, min], color = almost_black, linewidth=0.5)
            # ax.plot([xaxis_s+0.1, xaxis_e-0.1], [max, max], color = almost_black, linewidth=0.5)
            # ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p75, max], color = almost_black, linewidth=0.5)
            # ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p25, min], color = almost_black, linewidth=0.5)

            # if region['name'] == 'global':
            #     if i == 0 and scenario == 'ssp245':
            #         ax.plot((xaxis_s+xaxis_e)/2,mean_east_AU,  marker='o', c='white', alpha=0.5, markersize=6, label="East Australia", markeredgecolor="black")
            #         ax.plot((xaxis_s+xaxis_e)/2,mean_west_EU,  marker='^', c='white', alpha=0.5, markersize=6, label="West Europe", markeredgecolor="black")
            #         ax.plot((xaxis_s+xaxis_e)/2,mean_north_Am, marker='*', c='white', alpha=0.5, markersize=7, label="North America",markeredgecolor="black")
            #     else:
            #         ax.plot((xaxis_s+xaxis_e)/2,mean_east_AU,  marker='o', c='white', alpha=0.5, markersize=6, markeredgecolor="black")
            #         ax.plot((xaxis_s+xaxis_e)/2,mean_west_EU,  marker='^', c='white', alpha=0.5, markersize=6, markeredgecolor="black")
            #         ax.plot((xaxis_s+xaxis_e)/2,mean_north_Am, marker='*', c='white', alpha=0.5, markersize=7, markeredgecolor="black")


        ax.text(xaxis_s-0.8, p75+1 , f"{p25_hist:.0f}", va='bottom', ha='center', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        ax.text(xaxis_s+0.7, p75+1 , f"{p25_diff:+.0f}", va='bottom', ha='center',c='red', rotation_mode='anchor',transform=ax.transData, fontsize=8)

        ax.text(xaxis_s-0.8, p75+5 , f"{mean_hist:.0f}", va='bottom', ha='center', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        ax.text(xaxis_s+0.7, p75+5 , f"{mean_diff:+.0f}", va='bottom', ha='center',c='red', rotation_mode='anchor',transform=ax.transData, fontsize=8)

        ax.text(xaxis_s-0.8, p75+9 , f"{p75_hist:.0f}", va='bottom', ha='center', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        ax.text(xaxis_s+0.7, p75+9 , f"{p75_diff:+.0f}", va='bottom', ha='center',c='red', rotation_mode='anchor',transform=ax.transData, fontsize=8)


    ax.legend(fontsize=9, frameon=False)#, ncol=2)

    ax.set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    # ax.set_xlabel("models", fontsize=12)

    ax.set_xticks(np.arange(1.5,len(model_list)*3,3))
    ax.set_xticklabels(model_list,fontsize=9, rotation=90)

    ax.set_xlim(0.,0.2+len(model_list)*3)
    ax.set_ylim(0,170)
    ax.tick_params(axis='y', labelsize=12)

    if dist_type !=None:
        fig.savefig(f"./plots/plot_predicted_CMIP6_boxplot_{region['name']}_{dist_type}.png",bbox_inches='tight',dpi=300) # '_30percent'
    else:
        fig.savefig(f"./plots/plot_predicted_CMIP6_boxplot_{region['name']}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

def plot_predicted_CMIP6_diff_violin(CMIP6_txt_path, var_name, model_list, scenario,
                                     region={'name':'global','lat':None, 'lon':None}, dist_type=None):

    # ============ Setting up plot ============
    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[12,6],sharex=False, sharey=False, squeeze=True)

    plt.subplots_adjust(wspace=0.09, hspace=0.02)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # read all files
    var          = {}

    for i, model_in in enumerate(model_list):

        if dist_type!=None:
            file_in = f'{CMIP6_txt_path}/refuse/predicted_CMIP6_DT_{var_name}_{scenario}_{model_in}_{region["name"]}_{dist_type}_no_head.csv'
        else:
            file_in = f'{CMIP6_txt_path}/refuse/predicted_CMIP6_DT_{var_name}_{scenario}_{model_in}_{region["name"]}.csv'

        if model_in == 'CABLE':
            var_tmp           = pd.read_csv(file_in, header=None, names=['index','CMIP6', model_in], na_values=[''], usecols=[1, 2])#,nrows=200)
        else:
            var_tmp[model_in] = pd.read_csv(file_in, header=None, names=['index', model_in], na_values=[''], usecols=[1])#,nrows=200)


    for model_in in model_list:
        # if model_in != 'obs':
        #     var_input[model_in] = var_input[model_in] - var_input['obs']
        var_temp          = var_tmp[model_in].values - var_tmp['CMIP6'].values
        var_tmp[model_in] = np.where(np.abs(var_temp)<1000., var_temp, np.nan)

    for model_in in model_list:
        if model_in == 'CABLE':
            var_diff            = pd.DataFrame(var_tmp[model_in].values,columns=['diff'])
            var_diff['model_in']= model_in
        else:
            var_t             = pd.DataFrame(var_tmp[model_in].values,columns=['diff'])
            var_t['model_in'] = model_in
            var_diff          = pd.concat([var_diff, var_t], ignore_index=True) 
            var_t             = None
        # var_diff = var_tmp.pop("CMIP6")

    # Method 1: failed due to too many data points
    # plot = sns.violinplot( data=var_diff,x='model_in',y='diff', ax= ax)


    # Method 3:
    plot = go.Figure(data=go.Violin(y=var_diff['diff'], box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                               x0='model_in'))#,ax=ax)
    # ax.set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    # ax.set_xlabel("Land surface models", fontsize=12)
    
    if dist_type !=None:
        plot.write_image(f"./plots/plot_predicted_CMIP6_violin_{region['name']}_{dist_type}.png")
    else:
        plot.write_image(f"./plots/plot_predicted_CMIP6_violin_{region['name']}.png")

    
    # # Method 2:
    # import altair as alt
    # from vega_datasets import data
    #
    # plot = alt.Chart(var_diff).transform_density(
    #         # 'Latent_Heat',
    #         # as_=['Miles_per_Gallon', 'density'],
    #         # extent=[5, 50],
    #         # groupby=['Origin']
    #     ).mark_area(orient='horizontal').encode(
    #         alt.X('density:Q')
    #             .stack('center')
    #             .impute(None)
    #             .title(None)
    #             .axis(labels=False, values=[0], grid=False, ticks=True),
    #         alt.Y('Miles_per_Gallon:Q'),
    #         alt.Color('Origin:N'),
    #         alt.Column('Origin:N')
    #             .spacing(0)
    #             .header(titleOrient='bottom', labelOrient='bottom', labelPadding=0)
    #     ).configure_view(
    #         stroke=None
    #     )

    # ax.legend(fontsize=9, frameon=False)#, ncol=2)

    # ax.set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    # ax.set_xlabel("Land surface models", fontsize=12)

    # # ax.set_ylim(-200,200)
    # ax.tick_params(axis='y', labelsize=12)

    # if dist_type !=None:
    #     fig.savefig(f"./plots/plot_predicted_CMIP6_violin_{region['name']}_{dist_type}.png",bbox_inches='tight',dpi=300) # '_30percent'
    # else:
    #     fig.savefig(f"./plots/plot_predicted_CMIP6_violin_{region['name']}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return


def plot_predicted_CMIP6_regions(CMIP6_txt_path, var_name, model_list, region_list, scenarios):

    model_list.append('CMIP6')

    # ============ Setting up plot ============
    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[12,6],sharex=False, sharey=False, squeeze=True)

    plt.subplots_adjust(wspace=0.09, hspace=0.02)

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

    props        = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    nmodel       = len(model_list)
    mean_all     = np.zeros((nmodel,2,4))

    # read all metrics files
    for i, model_in in enumerate(model_list):
        # set line color
        line_color = model_colors[model_in]

        for j, scenario in enumerate(scenarios):

            for k, region_name in enumerate(region_list):
                metrics         = pd.read_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_{var_name}_{scenario}_{model_in}_{region_name}.csv', na_values=[''])
                mean_all[i,j,k] = metrics.loc[0,model_in]

                if scenario == 'historical':
                    xaxis   = 1+k*3
                    alpha   = 0.9
                elif scenario == 'ssp245':
                    xaxis   = 2+k*3
                    alpha   = 0.5

                ax.scatter(xaxis,mean_all[i,j,k],  s=6 ,c=line_color, alpha=alpha)

    # ax.legend(fontsize=7, frameon=False, ncol=2)

    ax.set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)

    ax.set_xticks(np.arange(1.5,len(region_list)*3,3))
    ax.set_xticklabels(region_list,fontsize=9, rotation=90)

    ax.set_xlim(0.2,0.2+len(region_list)*3)
    ax.tick_params(axis='y', labelsize=12)

    fig.savefig(f"./plots/plot_predicted_CMIP6_boxplot_regions.png",bbox_inches='tight',dpi=300) # '_30percent'

    return


if __name__ == "__main__":

    # Get model lists
    CMIP6_txt_path    = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6'
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list = model_names['model_select']
    scenarios  = ['historical','ssp245']
    var_name   = 'Qle'
    dist_type  = "Poisson" #None # 'Poisson'

    model_list.append('CMIP6')

    region     = {'name':'global', 'lat':None, 'lon':None}
    # plot_predicted_CMIP6_boxplot(CMIP6_txt_path, var_name, model_list, scenarios, region=region, dist_type=dist_type)
    # region     = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
    # plot_predicted_CMIP6_boxplot(CMIP6_txt_path, var_name, model_list, scenarios, region=region)
    # region     = {'name':'north_Am', 'lat':[25,58], 'lon':[-125,-65]}
    # plot_predicted_CMIP6_boxplot(CMIP6_txt_path, var_name, model_list, scenarios, region=region)
    # region     = {'name':'east_AU', 'lat':[-44.5,-22], 'lon':[138,155]}
    # plot_predicted_CMIP6_boxplot(CMIP6_txt_path, var_name, model_list, scenarios, region=region)

    CMIP6_models  = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'CMCC-ESM2','EC-Earth3',
                     'KACE-1-0-G', 'MIROC6', 'MIROC-ES2L','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
                     'MRI-ESM2-0']

    scenario      = 'ssp245'
    model_list    = ['CABLE','CABLE-POP-CN']
    plot_predicted_CMIP6_diff_violin(CMIP6_txt_path, var_name, model_list, scenario,
                                     region=region, dist_type=dist_type)

    # region_list = ['global','east_AU','west_EU','north_Am',]
    # plot_predicted_CMIP6_regions(CMIP6_txt_path, var_name, model_list, region_list, scenarios)
