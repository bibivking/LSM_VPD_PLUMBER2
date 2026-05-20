import os
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt

def plot_shap_var(df_merged, case_name, score_metric, feature_names, test=False,
                  column_filter=None, var_range_filter=None):

    df = df_merged.copy()

    if column_filter is not None:
        df = df[df[column_filter[0]] == column_filter[1]].copy()

    if var_range_filter is not None:
        df = df[
            (df[var_range_filter[0]] >= var_range_filter[1]) &
            (df[var_range_filter[0]] <= var_range_filter[2])
        ].copy()

    n_features = len(feature_names)
    ncols = 3 if n_features >= 3 else n_features
    nrows = math.ceil(n_features / ncols)

    fig_width = 4.5 * ncols
    fig_height = 3.8 * nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False
    )

    axes = axes.flatten() if n_features > 1 else [axes]

    for ax, f in zip(axes, feature_names):
        shap_col = f"{f}_shap"

        if f not in df.columns:
            raise KeyError(f"Column '{f}' not found.")
        if shap_col not in df.columns:
            raise KeyError(f"Column '{shap_col}' not found.")

        ax.scatter(df[f], df[shap_col], s=5, alpha=0.3)
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP value")
        ax.set_title(f)

    for ax in axes[len(feature_names):]:
        ax.remove()

    fig.suptitle(f"SHAP dependence plots ({case_name})", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = f"./plots/shap/{case_name}"
    os.makedirs(out_dir, exist_ok=True)

    filter_tag = ""
    if column_filter is not None:
        filter_tag += f"_{column_filter[0]}-{column_filter[1]}"
    if var_range_filter is not None:
        filter_tag += f"_{var_range_filter[0]}-{var_range_filter[1]}-{var_range_filter[2]}"

    out_file = (
        f"{out_dir}/shap_dependence_all_features_"
        f"{case_name}_{score_metric}"
        f"{'_test' if test else ''}"
        f"{filter_tag}.png"
    )

    fig.savefig(out_file, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP analysis for XGBoost model.")
    parser.add_argument("--select_model", type=str, default="CABLE", required=True)
    parser.add_argument("--select_site", type=str, default=None, required=False)
    parser.add_argument("--folder_path", type=str,
                        default="/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP",
                        required=False)
    parser.add_argument("--batch_size", type=int, required=False, default=5000)
    parser.add_argument("--score_metric", type=str, required=False, default="r2")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    select_model = args.select_model
    select_site = args.select_site
    folder_path = args.folder_path
    batch_size = args.batch_size
    score_metric = args.score_metric
    test = args.test

    print(f"Model: {select_model}, Site: {select_site}, Folder: {folder_path}")

    case_name = f"{select_model}_{select_site or 'all'}"

    df_obs = pd.read_csv(
        f"{folder_path}/txt/{case_name}/data_slide/data_selectec_{case_name}{'_test' if test else ''}.csv",
        usecols=["time", "VPD", "site_name", "obs_Qair", "obs_Tair", "obs_Precip", "obs_SWdown", "SMtop1m", "LAI"]
    )

    df_shap = pd.read_csv(
        f"{folder_path}/txt/{case_name}/shap_values/shap_values_{case_name}_{score_metric}{'_test' if test else ''}.csv"
    )

    df_shap["time"] = pd.to_datetime(df_shap["time"])
    df_obs["time"] = pd.to_datetime(df_obs["time"])

    df_merged = pd.merge(
        df_shap,
        df_obs,
        on=["time", "site_name"],
        how="inner",
        suffixes=("_shap", "")
    )

    column_filter = None # ["site_name", "AU-Tum"]
    var_range_filter = ["VPD", 4, 5]
    feature_names = ["obs_Qair", "obs_Tair", "obs_Precip", "obs_SWdown", "SMtop1m", "LAI"]

    plot_shap_var(
        df_merged,
        case_name,
        score_metric,
        feature_names,
        test=test,
        column_filter=column_filter,
        var_range_filter=var_range_filter
    )