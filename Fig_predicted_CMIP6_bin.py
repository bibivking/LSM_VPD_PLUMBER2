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
import pandas as pd
import netCDF4 as nc
from matplotlib.patches import Polygon
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *
from calc_LSM_lead_CMIP6_uncertaity import calc_stat


def plot_predicted_CMIP6_pdf_bin(CMIP6_txt_path, var_name, model_list,
                             region={'name':'global','lat':None, 'lon':None}, dist_type=None):

    # ============ Setting up plot ============
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

    # =============== Set up figure =================
    fig = plt.figure(figsize=[12,7])#, constrained_layout = True)
    gs  = fig.add_gridspec(5, 5)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[1:4, 0:3])
    ax3 = fig.add_subplot(gs[1:4, 3:5])
    plt.subplots_adjust(wspace=0.32, hspace=0.15)

    # Set the colors for different models
    model_colors = set_model_colors()

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    if region['name'] == 'west_EU':
        order = ['(a)','(b)','(c)']
    elif region['name'] == 'north_Am':
        order = ['(d)','(e)','(f)']
    elif region['name'] == 'east_AU':
        order = ['(g)','(h)','(i)']


    # ====================== ax1 ======================
    model_in               = 'CABLE'
    # var_EF_historical      = pd.read_csv(f'./txt/CMIP6/binned/bin_EF_CMIP6_DT_filtered_by_VPD_Qle_historical_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
    # var_EF_ssp245          = pd.read_csv(f'./txt/CMIP6/binned/bin_EF_CMIP6_DT_filtered_by_VPD_Qle_ssp245_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
    # var_VPD_historical     = pd.read_csv(f'./txt/CMIP6/binned/bin_VPD_CMIP6_DT_filtered_by_VPD_Qle_historical_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
    # var_VPD_ssp245         = pd.read_csv(f'./txt/CMIP6/binned/bin_VPD_CMIP6_DT_filtered_by_VPD_Qle_ssp245_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
    var_EF_historical      = pd.read_csv(f'./txt/CMIP6/binned/bin_EF_bin_by_EF_annual_hist_CMIP6_DT_filtered_by_VPD_EF_historical_{region["name"]}_{dist_type}.csv',na_values=[''])
    var_EF_ssp245          = pd.read_csv(f'./txt/CMIP6/binned/bin_EF_bin_by_EF_annual_hist_CMIP6_DT_filtered_by_VPD_EF_ssp245_{region["name"]}_{dist_type}.csv',na_values=[''])
    var_VPD_historical     = pd.read_csv(f'./txt/CMIP6/binned/bin_VPD_bin_by_EF_annual_hist_CMIP6_DT_filtered_by_VPD_VPD_historical_{region["name"]}_{dist_type}.csv',na_values=[''])
    var_VPD_ssp245         = pd.read_csv(f'./txt/CMIP6/binned/bin_VPD_bin_by_EF_annual_hist_CMIP6_DT_filtered_by_VPD_VPD_ssp245_{region["name"]}_{dist_type}.csv',na_values=[''])
    # EF_series              = var_EF_historical['bin_series']
    # pdf_EF_historical      = var_EF_historical['pdf']*100.
    # pdf_EF_ssp245          = var_EF_ssp245['pdf']*100.
    # VPD_series             = var_VPD_historical['bin_series']
    # pdf_VPD_historical     = var_VPD_historical['pdf']*100.
    # pdf_VPD_ssp245         = var_VPD_ssp245['pdf']*100.

    EF_series              = var_EF_historical['bin_series']
    pdf_EF_historical      = var_EF_historical['vals']
    pdf_EF_ssp245          = var_EF_ssp245['vals']

    VPD_series             = var_VPD_historical['bin_series']
    pdf_VPD_historical     = var_VPD_historical['vals']
    pdf_VPD_ssp245         = var_VPD_ssp245['vals']

    ax4  = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # plot = ax1.plot(EF_series, pdf_EF_historical, lw=1, color='black', linestyle='solid',
    #                         alpha=1., label='EF_hist')
    # plot = ax1.plot(EF_series, pdf_EF_ssp245, lw=1, color='black', linestyle='dashed',
    #                         alpha=1., label='EF_ssp245')

    plot = ax1.axhline(y=0, color='grey', linestyle='dotted')
    plot = ax1.plot(EF_series, (pdf_EF_ssp245-pdf_EF_historical),
                    lw=1, color='black', linestyle='solid', alpha=1., label='EF_ssp245')

    # plot = ax4.plot(VPD_series, pdf_VPD_historical, lw=1, color='red', linestyle='solid',
    #                         alpha=1., label='VPD_hist')
    # plot = ax4.plot(VPD_series, pdf_VPD_ssp245, lw=1, color='red', linestyle='dashed',
    #                         alpha=1., label='VPD_ssp245')

    plot = ax4.plot(VPD_series,  (pdf_VPD_ssp245-pdf_VPD_historical),
                    lw=1, color='red', linestyle='solid', alpha=1., label='VPD_ssp245')


    ax1.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    ax1.set_xticklabels(['','','','','','','','',''], fontsize=12)

    ax1.text(0.04, 0.74, order[0], va='bottom', ha='center', rotation_mode='anchor',transform=ax1.transAxes, fontsize=16)
    ax1.set_xlim(0.,1.0)
    ax1.set_ylim(-0.06,0.06)
    ax1.set_yticks([-0.03,0,0.03])
    ax1.set_yticklabels(['-0.03','0','0.03'], fontsize=12)
    ax4.set_ylim(-0.5,0.5)
    ax4.set_yticks([-0.3,0,0.3])
    ax4.set_yticklabels(['-0.3','0','0.3'], fontsize=12)

    ax1.set_ylabel('ΔEF',color='black', fontsize=12)
    ax4.set_ylabel('ΔVPD (kPa)',color='red', fontsize=12)
    # ax1.set_xlabel("EF", fontsize=12)

    # ====================== ax2 ======================
    for i, model_in in enumerate(model_list):

        print('model_in',model_in)

        # set line color
        line_color = model_colors[model_in]

        if model_in == "obs":
            lw = 2.5
        else:
            lw = 1.5

        # ============= Reading data ==============
        var_EF_hist_historical = pd.read_csv(f'./txt/CMIP6/binned/bin_EF_annual_hist_CMIP6_DT_filtered_by_VPD_Qle_historical_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
        var_EF_hist_ssp245     = pd.read_csv(f'./txt/CMIP6/binned/bin_EF_annual_hist_CMIP6_DT_filtered_by_VPD_Qle_ssp245_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
        # var_EF_hist_historical = pd.read_csv(f'./txt/CMIP6/binned/bin_EF_CMIP6_DT_filtered_by_VPD_Qle_historical_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
        # var_EF_hist_ssp245     = pd.read_csv(f'./txt/CMIP6/binned/bin_EF_CMIP6_DT_filtered_by_VPD_Qle_ssp245_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
        # var_EF_hist_historical = pd.read_csv(f'./txt/CMIP6/binned/bin_VPD_CMIP6_DT_filtered_by_VPD_Qle_historical_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])
        # var_EF_hist_ssp245     = pd.read_csv(f'./txt/CMIP6/binned/bin_VPD_CMIP6_DT_filtered_by_VPD_Qle_ssp245_{model_in}_{region["name"]}_{dist_type}.csv',na_values=[''])

        diff_EF_hist           = (var_EF_hist_ssp245['vals'] - var_EF_hist_historical['vals'])/var_EF_hist_historical['vals']*100
        mask_val               = ~np.isnan(diff_EF_hist)
        diff_EF_hist_tmp       = diff_EF_hist[mask_val]
        EF_series_tmp          = EF_series[mask_val]
        # print(diff_EF_hist)
        diff_EF_hist_smooth    = smooth_vpd_series(diff_EF_hist_tmp, window_size=11, order=2, smooth_type='S-G_filter')

        # Set x-tick
        plot = ax2.plot(EF_series_tmp, diff_EF_hist_smooth, lw=lw, color=line_color, linestyle='solid',
                                alpha=1., label=f'{change_model_name(model_in)}') #edgecolor='none', c='red' .rolling(window=10).mean()
        # if region['name'] == 'east_AU':
        plot = ax2.axhline(y=0, color='grey',lw=0.5,  linestyle='dotted')
        ax2.text(0.04, 0.9, order[1], va='bottom', ha='center', rotation_mode='anchor',transform=ax2.transAxes, fontsize=16)
    # ax2.set_xticks(np.arange(1.5,len(model_list)*3,3))
    # ax2.set_xticklabels(changed_name,fontsize=11, ha='right',rotation=30)

    if region['name'] == 'west_EU':
        ax2.set_ylim(-4,10)
        ax2.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        ax2.set_xticklabels(['','','','','','','','',''], fontsize=12)
        ax2.legend(fontsize=7, frameon=False, ncol=3)
    elif region['name'] == 'north_Am':
        ax2.set_ylim(-4,10)
        ax2.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        ax2.set_xticklabels(['','','','','','','','',''], fontsize=12)
    elif region['name'] == 'east_AU':
        ax2.set_ylim(-4,10)
        ax2.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        ax2.set_xticklabels(['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'], fontsize=12)
        ax2.set_xlabel("Evaporative Fraction", fontsize=12)

    ax2.set_xlim(0.,1.0)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_ylabel("ΔLatent heat (%)", fontsize=12)

    # =================== ax3 ====================
    # if 'obs' in model_list:
    #     model_list.remove('obs')

    # read all metrics files
    changed_name = []
    for i, model_in in enumerate(model_list):
        if model_in != 'obs':
            # set line color
            line_color = model_colors[model_in]

            if dist_type!=None:
                metrics  = pd.read_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_DT_filtered_by_VPD_{var_name}_diff_ssp245_{region["name"]}_{dist_type}.csv', na_values=[''])
            else:
                metrics  = pd.read_csv(f'{CMIP6_txt_path}/metrics/metrics_CMIP6_DT_filtered_by_VPD_{var_name}_diff_ssp245_{region["name"]}.csv', na_values=[''])

            mean     = metrics.loc[0,model_in]
            p25      = metrics.loc[1,model_in]
            p75      = metrics.loc[2,model_in]
            min      = metrics.loc[3,model_in]
            max      = metrics.loc[4,model_in]

            # Set x-tick
            xaxis_s = 0.1+i*1
            xaxis_e = 0.9+i*1
            alpha   = 1.

            # Draw the box
            ax3.add_patch(Polygon([[xaxis_s, p25], [xaxis_s, p75],
                                    [xaxis_e, p75], [xaxis_e, p25]],
                                closed=True, color=line_color, fill=True, alpha=alpha, linewidth=0.1))

            # Draw the mean line
            ax3.plot([xaxis_s,xaxis_e], [mean,mean], color = 'white', linewidth=1.)

            # Draw the p25 p75
            ax3.plot([xaxis_s, xaxis_e], [p25, p25], color = almost_black, linewidth=1.)
            ax3.plot([xaxis_s, xaxis_e], [p75, p75], color = almost_black, linewidth=1.)

            ax3.plot([xaxis_s, xaxis_s], [p25, p75], color = almost_black, linewidth=1.)
            ax3.plot([xaxis_e, xaxis_e], [p25, p75], color = almost_black, linewidth=1.)

            # Draw the max and min
            ax3.plot([xaxis_s+0.1, xaxis_e-0.1], [min, min], color = almost_black, linewidth=0.8)
            ax3.plot([xaxis_s+0.1, xaxis_e-0.1], [max, max], color = almost_black, linewidth=0.8)
            ax3.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p75, max], color = almost_black, linewidth=0.8)
            ax3.plot([(xaxis_s+xaxis_e)/2, (xaxis_s+xaxis_e)/2], [p25, min], color = almost_black, linewidth=0.8)
            ax3.axhline(y=0, color=almost_black, linestyle='--', linewidth=0.5)#

            changed_name.append(change_model_name(model_in))

    # ax[row,col].legend(fontsize=9, frameon=False)#, ncol=2)
    ax3.text(0.06, 0.9, order[2], va='bottom', ha='center', rotation_mode='anchor',transform=ax3.transAxes, fontsize=16)

    if region['name'] == 'east_AU':
        ax3.set_xticks(np.arange(0.5,len(model_list[:-1]),1))
        ax3.set_xticklabels(changed_name, fontsize=9, ha='right',rotation=30)
    else:
        ax3.set_xticks(np.arange(0.5,len(model_list[:-1]),1))
        ax3.set_xticklabels(['','','','','','','','','','','','','','',''])

    ax3.set_xlim(-0.2,0.2+len(model_list[:-1])*1)

    ax3.set_ylim(-60,45)
    ax3.set_yticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40])
    ax3.set_yticklabels(['-60','-50','-40','-30','-20','-10','0','10','20','30','40'])
    ax3.tick_params(axis='y', labelsize=12)

    fig.savefig(f"./plots/plot_predicted_CMIP6_pdf_bin_{region['name']}.png",bbox_inches='tight',dpi=300) # '_30percent'

    return

if __name__ == "__main__":

    # Get model lists
    CMIP6_txt_path = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/txt/CMIP6'
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    model_list     = model_names['model_select_new']
    var_name       = 'Qle'
    dist_type      = "Gamma" #None # 'Poisson'

    # ====================== plot_predicted_CMIP6_pdf_bin =======================
    model_list     = model_names['model_select_new']
    region_names   = ['north_Am','west_EU', 'east_AU']

    for region_name in region_names:
        region = get_region_info(region_name)
        plot_predicted_CMIP6_pdf_bin(CMIP6_txt_path, var_name, model_list, region=region, dist_type=dist_type)
