import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_shap_distributions(df_shap, select_model, select_site, feature_names, score_metric=None, test=False):
    """
    Plot SHAP value distributions for each predictor
    """

    plt.figure(figsize=(8, len(feature_names) * 0.6))

    shap_long = df_shap[feature_names].melt(
        var_name="Feature", value_name="SHAP"
    )

    sns.violinplot(
        data=shap_long,
        y="Feature",
        x="SHAP",
        inner="quartile",
        scale="width",
        cut=0
    )

    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.tight_layout()

    case_name = f"{select_model}_{select_site or 'all'}"

    plt.title(case_name)
    plt.savefig(f"./plots/shap/{case_name}/shap_violin_predictors_{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.png", dpi=300)

def plot_shap_importance(df_shap, select_model, select_site, feature_names, score_metric=None,  test=False):
    mean_abs_shap = (
        df_shap[feature_names]
        .abs()
        .mean()
        .sort_values(ascending=True)
    )

    case_name = f"{select_model}_{select_site or 'all'}"

    plt.figure(figsize=(6, len(feature_names) * 0.5))
    mean_abs_shap.plot(kind="barh")
    plt.xlabel("Mean |SHAP|")
    plt.title(f"SHAP Feature Importance {case_name}")
    plt.tight_layout()

    plt.savefig(f"./plots/shap/{case_name}/shap_importance_{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.png", dpi=300)

def plot_shap_dependence( df_shap, df_depended, select_model, select_site, feature_names,score_metric=None,test=False,):
    n_features = len(feature_names)

    # -------- layout settings --------
    ncols = 3 if n_features >= 3 else n_features
    nrows = math.ceil(n_features / ncols)

    # figure size: 每个子图给固定空间
    fig_width  = 4.5 * ncols
    fig_height = 3.8 * nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False
    )

    # axes 统一成一维，方便循环
    axes = axes.flatten() if n_features > 1 else [axes]

    # -------- plotting --------
    for ax, f in zip(axes, feature_names):
        ax.scatter(
            df_depended,
            df_shap[f],
            s=5,
            alpha=0.3
        )
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP value")
        ax.set_title(f)

    # 多余的 subplot 删掉（防止空白）
    for ax in axes[len(feature_names):]:
        ax.remove()

    # -------- global title --------
    case_name = f"{select_model}_{select_site or 'all'}"

    fig.suptitle(
        f"SHAP dependence plots ({case_name})",
        fontsize=14,
        y=0.98
    )

    # 自动调整间距（不重合的关键）
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # -------- save --------
    out_dir = f"./plots/shap/{case_name}"
    os.makedirs(out_dir, exist_ok=True)

    fig.savefig(
        f"{out_dir}/shap_dependence_all_features_"
        f"{select_model}_{select_site or 'all'}_"
        f"{score_metric}{'_test' if test else ''}.png",
        dpi=300
    )
    plt.close(fig)


if __name__ == "__main__":

    # ========== Set up argument parser ==========
    parser = argparse.ArgumentParser(description="Run SHAP analysis for XGBoost model.")

    parser.add_argument("--select_model", type=str, default="CABLE",  required=True, help="Model name (e.g., CABLE)")
    parser.add_argument("--select_site", type=str, default=None,  required=False, help="Site name (e.g., AU-Tum)")
    parser.add_argument("--folder_path", type=str, default="/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP",
                                         required=False, help="Path to the script folder")
    parser.add_argument("--score_metric", type=str, required=False, default='r2', help="scoring metric for GridSearchCV")
    parser.add_argument("--test", action="store_true", help="whether to run on a small sample for testing")
    
    # Parse arguments
    args = parser.parse_args()

    # Assign arguments to variables
    select_model = args.select_model
    select_site  = args.select_site
    folder_path  = args.folder_path
    score_metric = args.score_metric
    test         = args.test  

    # ========== Set case name ==========
    case_name    = f"{select_model}_{select_site or 'all'}"
    os.makedirs(f"./plots/shap/{case_name}", exist_ok=True)
    
    # ========== Read in data ==========
    shap_file_in  = f"{folder_path}/txt/{case_name}/shap_values/shap_values_{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.csv"
    df_shap       = pd.read_csv(shap_file_in)     

    feature_names = ['obs_Qair', 'obs_Tair', 'obs_Precip', 'obs_SWdown', 'SMtop1m', 'LAI']

    plot_shap_distributions(df_shap, select_model, select_site, feature_names, score_metric, test)
    plot_shap_importance(df_shap, select_model, select_site, feature_names, score_metric, test)

    data_file_in  = f"{folder_path}/txt/{case_name}/data_slide/data_selectec_{case_name}{'_test' if test else ''}.csv"
    df_select     = pd.read_csv(data_file_in)   
    
    if select_model == 'obs':
        df_depended = df_select["obs"]
    else:
        df_depended = df_select['model_'+select_model] 

    plot_shap_dependence(df_shap, df_depended, select_model, select_site, feature_names, score_metric, test)