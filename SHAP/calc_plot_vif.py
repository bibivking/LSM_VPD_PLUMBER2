import os
import argparse
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calc_vif(df_select, select_site=None):

    # 添加常数列（截距项）
    X_const = add_constant(df_select)

    # 计算每个变量的 VIF
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X_const.columns
    vif_data['VIF'] = [variance_inflation_factor(X_const.values, i)
                    for i in range(X_const.shape[1])]
    
    vif_data.to_csv(f"./predictors_vif/vif_{select_site or 'all'}.txt")
    print(f"✅ VIF txt saved to f./predictors_vif/vif_{select_site or 'all'}.txt")
    
    return

def plot_vif(select_site=None, threshold=5):

    vif_data = pd.read_csv(
        f"./predictors_vif/vif_{select_site or 'all'}.txt",
        index_col=0
    ).sort_values('VIF', ascending=False)

    # Create a category instead of raw colors
    vif_data['Flag'] = np.where(vif_data['VIF'] > threshold, 'High VIF', 'OK')

    palette = {
        'High VIF': 'red',
        'OK': 'steelblue'
    }

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='VIF',
        y='Variable',
        data=vif_data,
        hue='Flag',
        palette=palette,
        dodge=False
    )

    plt.axvline(threshold, color='orange', linestyle='--',
                label=f'Threshold = {threshold}')

    plt.xlabel('VIF')
    plt.ylabel('Predictor')
    plt.title('Variance Inflation Factor (VIF) per Predictor')
    plt.legend()
    plt.tight_layout()

    os.makedirs("./plots/predictors_vif", exist_ok=True)
    plt.savefig(
        f"./plots/predictors_vif/vif_{select_site or 'all'}.png",
        dpi=300
    )
    plt.close()

    print(f"✅ VIF plot saved to ./plots/predictors_vif/vif_{select_site or 'all'}.png")

if __name__ == "__main__":

    # Assign arguments to variables"
    select_model = "CABLE"
    select_site  = "AU-Tum"
    folder_path  = "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP"
    test         = False

    # Debug print to check values (optional)
    print(f"Model: {select_model}, Site: {select_site}, Folder: {folder_path}, test: {test}")

    # --------------- Load dataset --------------
    if select_model == 'obs':
        var_input = pd.read_csv(
            '/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/Qle_all_sites.csv',
            usecols=['time', 'site_name', 'IGBP_type', 'climate_type', select_model, 'VPD', 'obs_Tair', 'obs_Qair', 'obs_Precip', 'obs_SWdown'],
            na_values=['']
        )     
    else:
        var_input = pd.read_csv(
            '/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/Qle_all_sites.csv',
            usecols=['time', 'site_name', 'IGBP_type', 'climate_type', 'model_'+select_model, 'VPD', 'obs_Tair','obs_Qair',  'obs_Precip', 'obs_SWdown'],
            na_values=['']
        ) 

    if select_model == 'obs':
        var_input['SMtop1m'] = pd.read_csv(
            '/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/SMtop1m_all_sites.csv',
            usecols=['model_mean_SMtop1m'], na_values=['']
        )
    else:
        var_input['SMtop1m'] = pd.read_csv(
            '/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/SMtop1m_all_sites.csv',
            usecols=[select_model+'_SMtop1m'], na_values=['']
        )

    models_with_own_LAI = ['ORC2_r6593', 'ORC3_r8120', 'GFDL', 'SDGVM', 'QUINCY', 'NoahMPv401', 'JULES_GL9_withLAI']

    if select_model in models_with_own_LAI:
        var_input['LAI'] = pd.read_csv(
            '/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/LAI_all_sites_parallel.csv',
            usecols=[select_model+'_LAI'], na_values=['']
        )
    else:
        var_input['LAI'] = pd.read_csv(
            '/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/txt/process1_output/LAI_all_sites_parallel.csv',
            usecols=['obs_LAI'], na_values=['']
        )

    # Select data for training
    if select_site == None:
        var_select = var_input.dropna()
    else:
        var_select = var_input[var_input['site_name'] == select_site].dropna()
    
    if test:
        var_select = var_select.sample(frac=0.01, random_state=42, replace=False)

    # Extract features and target
    feature_names = feature_names = ['VPD', 'obs_Tair', 'obs_Qair','obs_Precip', 'obs_SWdown', 'SMtop1m', 'LAI',]
    # ['obs_Qair', 'obs_Tair', 'obs_Precip', 'obs_SWdown', 'SMtop1m', 'LAI',] # 尝试通过Qair+Tair的交互贡献来代表vpd的贡献
    df_select     = var_select[feature_names]

    # --------------- Check Variance Inflation Factor (VIF) between predictors --------------
    calc_vif(df_select, select_site)
    plot_vif(select_site)