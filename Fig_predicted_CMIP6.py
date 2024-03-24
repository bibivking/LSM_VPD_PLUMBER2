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
from calc_LSM_lead_CMIP6_uncertaity import calc_stat
from signal import signal, SIGPIPE, SIG_DFL
# import altair as alt

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

def plot_predicted_CMIP6_diff_boxplot(CMIP6_txt_path, var_name, model_list, region={'name':'global','lat':None, 'lon':None}, dist_type=None):

    # ============ Pre setting =============
    order = {'west_EU': '(a)',
             'north_Am': '(b)',
             'east_AU': '(c)', }

    # ============ Setting up plot ============
    fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[6,4], sharex=False, sharey=False, squeeze=True)

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

    model_list.remove('obs')
    # read all metrics files
    
    changed_name = []
    for i, model_in in enumerate(model_list):

        # set line color
        line_color = model_colors[model_in]

        if dist_type!=None:
            metrics  = pd.read_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_filtered_by_VPD_{var_name}_diff_{scenario}_{region["name"]}_{dist_type}.csv', na_values=[''])
        else:
            metrics  = pd.read_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_filtered_by_VPD_{var_name}_diff_{scenario}_{region["name"]}.csv', na_values=[''])
        mean     = metrics.loc[0,model_in]
        p25      = metrics.loc[1,model_in]
        p75      = metrics.loc[2,model_in]
        min      = metrics.loc[3,model_in]
        max      = metrics.loc[4,model_in]

        # Set x-tick
        xaxis_s = 0.2+i*1
        xaxis_e = 0.8+i*1
        alpha   = 1.
        mean_hist = mean
        p25_hist = p25
        p75_hist = p75

        # Draw the box
        ax.add_patch(Polygon([[xaxis_s, p25], [xaxis_s, p75],
                                [xaxis_e, p75], [xaxis_e, p25]],
                            closed=True, color=line_color, fill=True, alpha=alpha, linewidth=0.1))

        # Draw the mean line
        ax.plot([xaxis_s,xaxis_e], [mean,mean], color = 'white', linewidth=1.)


        # Draw the p25 p75
        ax.plot([xaxis_s, xaxis_e], [p25, p25], color = almost_black, linewidth=1.)
        ax.plot([xaxis_s, xaxis_e], [p75, p75], color = almost_black, linewidth=1.)

        ax.plot([xaxis_s, xaxis_s], [p25, p75], color = almost_black, linewidth=1.)
        ax.plot([xaxis_e, xaxis_e], [p25, p75], color = almost_black, linewidth=1.)

        # Draw the max and min
        ax.plot([xaxis_s+0.1, xaxis_e-0.1], [min, min], color = almost_black, linewidth=0.8)
        ax.plot([xaxis_s+0.1, xaxis_e-0.1], [max, max], color = almost_black, linewidth=0.8)
        ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p75, max], color = almost_black, linewidth=0.8)
        ax.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p25, min], color = almost_black, linewidth=0.8)
        ax.axhline(y=0, color=almost_black, linestyle='--', linewidth=0.5)#
        
        # ax.text(xaxis_s-0.8, p75+1 , f"{p25_hist:.0f}", va='bottom', ha='center', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        # ax.text(xaxis_s+0.7, p75+1 , f"{p25_diff:+.0f}", va='bottom', ha='center',c='red', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        #
        # ax.text(xaxis_s-0.8, p75+5 , f"{mean_hist:.0f}", va='bottom', ha='center', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        # ax.text(xaxis_s+0.7, p75+5 , f"{mean_diff:+.0f}", va='bottom', ha='center',c='red', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        #
        # ax.text(xaxis_s-0.8, p75+9 , f"{p75_hist:.0f}", va='bottom', ha='center', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        # ax.text(xaxis_s+0.7, p75+9 , f"{p75_diff:+.0f}", va='bottom', ha='center',c='red', rotation_mode='anchor',transform=ax.transData, fontsize=8)
        #
        changed_name.append(change_model_name(model_in))

    # ax.legend(fontsize=9, frameon=False)#, ncol=2)
    ax.text(0.05, 0.9, order[region["name"]], va='bottom', ha='center', rotation_mode='anchor',transform=ax.transAxes, fontsize=14)

    ax.set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    # ax.set_xlabel("models", fontsize=12)

    ax.set_xticks(np.arange(0.5,len(model_list),1))
    ax.set_xticklabels(changed_name,fontsize=9, rotation=90)

    ax.set_xlim(0.,0.6+len(model_list)*1)
    ax.set_ylim(-48,48)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title(region["name"], fontsize=12)

    if dist_type !=None:
        fig.savefig(f"./plots/plot_predicted_CMIP6_diff_boxplot_{region['name']}_{dist_type}.png",bbox_inches='tight',dpi=300) # '_30percent'
    else:
        fig.savefig(f"./plots/plot_predicted_CMIP6_diff_boxplot_{region['name']}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

def save_predicted_metrics_CMIP6_diff(CMIP6_txt_path, var_name, model_list, CMIP6_models, scenario,
                            region={'name':'global','lat':None, 'lon':None}, dist_type=None,
                            outlier_method='percentile', min_percentile=0.05, max_percentile=0.95,
                            reduce_sample=False):

    # ============== Save predicted CMIP6 diff ==============
    # read all files
    var_diff     = {}
    var          = {}

    # read different curve estimated CMIP6
    for i, model_in in enumerate(model_list):

        # if region['name'] == 'global':
        #     # read file names
        #     if dist_type!=None:
        #         file_in = f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}_no_head.csv'
        #     else:
        #         file_in = f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv'

        #     if model_in == 'CABLE':
        #         var[model_in] = pd.read_csv(file_in, header=None, names=['index','CMIP6', model_in], na_values=[''], usecols=[2])#,nrows=200)
        #     else:
        #         var[model_in] = pd.read_csv(file_in, header=None, names=['index', model_in], na_values=[''], usecols=[1])#,nrows=200)
        # else:
        var_tmp = []
        # read each predicted CMIP6 in to the same LSM's array
        for j, CMIP6_model in enumerate(CMIP6_models):
            print('CMIP6_model',CMIP6_model,'model_in',model_in)
            # read file names
            if dist_type!=None:
                file_in = f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}_{dist_type}.csv'
            else:
                file_in = f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_{scenario}_{CMIP6_model}_{model_in}_{region["name"]}.csv'

            var_tmp.extend(pd.read_csv(file_in, na_values=[''], usecols=[model_in])[model_in].tolist()) #,nrows=200)

        print('len(var_tmp) is', len(var_tmp))

        sample_num = len(var_tmp)

        if reduce_sample:
            var[model_in] = np.random.choice(var_tmp,round(sample_num*0.01))
        else:
            var[model_in] = var_tmp

    # Calcuate difference
    for model_in in model_list:
        # total = len(var[model_in])
        if model_in != 'obs':
            print(model_in)
            # print("var[model_in][0]", var[model_in][0], " var['obs'][0]", var['obs'][0])
            # var_diff[model_in] = [x - y for x, y in zip(var[model_in], var['obs'])]
            var_diff[model_in] = [x - y for x, y in zip(var[model_in], var['obs'])]

    # Do not loop obs
    model_list.remove('obs')
    #
    # # Put all data into a dataframe
    # for i, model_in in enumerate(model_list):
    #     print('Processing',model_in)
    #     if i == 0:
    #         df_var_diff             = pd.DataFrame(var_diff[model_in],columns=['diff'])
    #         df_var_diff['model_in'] = model_in
    #     else:
    #         var_t             = pd.DataFrame(var_diff[model_in],columns=['diff'])
    #         var_t['model_in'] = model_in
    #         df_var_diff       = pd.concat([df_var_diff, var_t], ignore_index=True)
    #         var_t             = None
    #
    # # save the difference
    # if reduce_sample:
    #     message = '_reduce_sample'
    # else:
    #     message = ''
    #
    # if dist_type!=None:
    #     df_var_diff.to_csv(f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_diff_{scenario}_{region["name"]}_{dist_type}{message}.csv' )
    # else:
    #     df_var_diff.to_csv(f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_diff_{scenario}_{region["name"]}{message}.csv')

    # ============== Calculate diff metrics ==============
    print('model_list is', model_list)

    if not reduce_sample:
        for i, model_in in enumerate(model_list):

            print('Calculate metrics of',model_in)

            # data_in = pd.DataFrame(var_diff[model_in], columns=[model_in])

            if i == 0:
                metrics = pd.DataFrame(calc_stat(var_diff[model_in], outlier_method=outlier_method, min_percentile=min_percentile, max_percentile=max_percentile),
                          columns=[model_in])
            else:
                metrics[model_in] = calc_stat(var_diff[model_in], outlier_method=outlier_method, min_percentile=min_percentile, max_percentile=max_percentile)

        if dist_type!=None:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_filtered_by_VPD_{var_name}_diff_{scenario}_{region["name"]}_{dist_type}.csv')
        else:
            metrics.to_csv(f'{CMIP6_txt_path}/metrics_CMIP6_DT_filtered_by_VPD_{var_name}_diff_{scenario}_{region["name"]}.csv')

    return

def plot_predicted_CMIP6_diff_violin(CMIP6_txt_path, var_name, model_list, CMIP6_models, scenario,
                            region={'name':'global','lat':None, 'lon':None}, dist_type=None,
                            reduce_sample=False):

    # To avoid the error we need to make the terminal run the code efficiently without
    # catching the SIGPIPE signal, so for these, we can add the below code at the top of the python program.

    # signal(SIGPIPE,SIG_DFL)

    if reduce_sample:
        message = '_reduce_sample'
    else:
        message = ''

    # Read data
    if dist_type!=None:
        df_var_diff = pd.read_csv(f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_diff_{scenario}_{region["name"]}_{dist_type}{message}.csv', na_values=[''] )
    else:
        df_var_diff = pd.read_csv(f'{CMIP6_txt_path}/predicted_CMIP6_DT_filtered_by_VPD_{var_name}_diff_{scenario}_{region["name"]}{message}.csv', na_values=[''])

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

    # Method 1: failed due to too many data points
    plot = sns.violinplot( data=df_var_diff, x='model_in',y='diff', ax= ax)

    # # Method 2
    # boxplot = alt.Chart().mark_boxplot(color='black').encode(
    #     alt.Y(f'Latent Heat (W m^{-2})')
    # ).properties(width=200)

    # violin = alt.Chart().transform_density(
    #     'Latent Heat (W m^{-2})',
    #     as_=['Miles_per_Gallon', 'density'],
    #     extent=[5, 50],
    #     groupby=['Origin']
    # ).mark_area(orient='horizontal').encode(
    #     y='Miles_per_Gallon:Q',
    #     color='Origin:N',
    #     x=alt.X(
    #         'density:Q',
    #         stack='center',
    #         impute=None,
    #         title=None,
    #         scale=alt.Scale(nice=False,zero=False),
    #         axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
    #     ),
    # )
    # alt.layer(violin, boxplot, data=df_var_diff.cars()).facet(column='Origin:N').
    #         resolve_scale(x=alt.ResolveMode("independent")).show()

    # # Method 3:
    # plot = go.Figure(data=go.Violin(x=df_var_diff['model_in'], y=df_var_diff['diff'], box_visible=True, line_color='black',
    #                            meanline_visible=False, fillcolor='lightseagreen', opacity=0.6))#,ax=ax)#x0='model_in'
    # # ax.set_ylabel("Latent Heat (W m$\mathregular{^{-2}}$)", fontsize=12)
    # # ax.set_xlabel("Land surface models", fontsize=12)

    # # Add labels and legend (replace placeholders with actual labels)
    # plot.update_layout(
    #     xaxis_title="Land surface models",
    #     yaxis_title="Latent Heat (W m^{-2})",
    #     legend_title="Model",
    #     # Set background color to white
    #     plot_bgcolor='white',
    #     paper_bgcolor='white',  # Set paper background color to white (optional)
    #     # violin_fillcolor='none',
    #     # xaxis_showgrid=True,  # Add grid lines on x-axis
    #     # yaxis_showgrid=True,  # Add grid lines on y-axis
    #     # xaxis_linecolor='black',  # Set axis line color
    #     # yaxis_linecolor='black',  # Set axis line color
    # )

    # if dist_type !=None:
    #     plot.write_image(f"./plots/plot_predicted_CMIP6_violin_{region['name']}_{dist_type}.png",engine='kaleido')
    #     # plot.savefig(f"./plots/plot_predicted_CMIP6_violin_{region['name']}_{dist_type}.png")
    # else:
    #     plot.write_image(f"./plots/plot_predicted_CMIP6_violin_{region['name']}.png",engine='kaleido')
    #     # plot.savefig(f"./plots/plot_predicted_CMIP6_violin_{region['name']}.png")


    # # Method 4
    # from superviolin import violinplot

    # # Extract the columns you want to plot (assuming 'category' and 'value')
    # category_column = "category"
    # value_column = "value"

    # # Generate the violin plot and save as PNG

    # if dist_type !=None:
    #     output_path = f"./plots/plot_predicted_CMIP6_violin_{region['name']}_{dist_type}.png"
    # else:
    #     output_path = f"./plots/plot_predicted_CMIP6_violin_{region['name']}.png"

    # violinplot(
    #     df_var_diff['diff'],  # Convert pandas Series to list
    #     category=df_var_diff['model_in'],  # Convert Series to list
    #     output=output_path,  # Specify output path
    #     format="png",  # Specify PNG format
    # )

    # print(f"Violin plot saved to: {output_path}")

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
    model_list = model_names['model_select_new']
    scenarios  = ['historical','ssp245']
    var_name   = 'Qle'
    dist_type  = "Gamma" #None # 'Poisson'
    outlier_method='percentile'
    
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
    model_list    = model_names['model_select_new']
    reduce_sample = False
    min_percentile= 0.15
    max_percentile= 0.85

    save_predicted_metrics_CMIP6_diff(CMIP6_txt_path, var_name, model_list, CMIP6_models, scenario,
                                region=region, dist_type=dist_type, reduce_sample=reduce_sample,
                                min_percentile=min_percentile, max_percentile=max_percentile,
                                outlier_method=outlier_method)

    # region     = {'name':'west_EU', 'lat':[35,60], 'lon':[-12,22]}
    region     = {'name':'north_Am', 'lat':[25,58], 'lon':[-125,-65]}
    # region     = {'name':'east_AU', 'lat':[-44.5,-22], 'lon':[138,155]}

    # plot_predicted_CMIP6_diff_boxplot(CMIP6_txt_path, var_name, model_list, region=region, dist_type=dist_type)

    # region_list = ['global','east_AU','west_EU','north_Am',]
    # plot_predicted_CMIP6_regions(CMIP6_txt_path, var_name, model_list, region_list, scenarios)
