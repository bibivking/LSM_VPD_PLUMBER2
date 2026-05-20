import os
import shap
import time
import joblib
import argparse
import seaborn as sns
import numpy as np
import pandas as pd
import multiprocessing
import scipy.stats as stats
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer,  r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from joblib import Parallel, delayed

def get_variables(case_name, folder_path, select_model, select_site=None, test=False):

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


    data_dir = os.path.join(folder_path, f"txt/{case_name}/data_slide") # txt 前面不可以加/ 否则会被当成根目录

    os.makedirs(data_dir, exist_ok=True)
    
    var_select.to_csv(f"{data_dir}/data_selectec_{case_name}{'_test' if test else ''}.csv")
    print(var_select.head())

    return  

def kling_gupta_efficiency(y_true, y_pred):

    # Custom function to calculate Kling-Gupta Efficiency (KGE)
    """Compute Kling-Gupta Efficiency (KGE)"""

    obs_mean, sim_mean = np.mean(y_true), np.mean(y_pred)
    obs_std, sim_std = np.std(y_true), np.std(y_pred)
    
    # Compute Pearson correlation coefficient
    r, _ = pearsonr(y_true, y_pred)  

    # Compute bias ratio (β)
    beta = sim_mean / obs_mean  

    # Compute variability ratio (γ)
    gamma = (sim_std / sim_mean) / (obs_std / obs_mean)  

    # Compute KGE score
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

    return kge

def train_xgboost(X: np.ndarray, y: np.ndarray, folder_path, select_model=None, select_site=None, 
                  score_metric='r2', n_splits=5, test_size=0.2, test=False):
    '''
    score_metric: 
        Option 1, 'r2', R² measures how well your model captures the variance in the target variable.
               2, 'kge_scorer', use the custom function, Kling-Gupta Efficiency (KGE).
                   KGE can more comprehensively evaluate the predicted mean, variability, and correlation.
    '''

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Set up XGBoost regressor
    regressor = XGBRegressor(objective="reg:squarederror", # defines how the model computes loss and updates trees.
                             tree_method="gpu_hist",
                             # device='cuda:0,cuda:1',  # CHANGED: Enable multi-GPU, xgboost 1.7.6, the version I install doesn't support using multiple GPUs 
                             verbosity=2)

    # Define the grid search parameters
    # Optimize plan 1
    # param_grid = {
    #     'learning_rate': [0.05, 0.1, 0.15, 0.2],
    #     'max_depth': [7, 8, 9, 10, 11],
    #     'subsample': [0.6, 0.7, 0.8, 0.9], #0.2, 0.4, 
    #     'n_estimators': [100, 200, 300, 400, 500, 600, 700]
    # }

    # Optimize plan 2
    param_grid = {
        'max_depth': [6, 8, 10, 12],  
        'learning_rate': [0.01, 0.03, 0.05],  
        'n_estimators': [1000, 2000, 3000],  
        'subsample': [0.6, 0.8, 1.0],  
        # 'reg_alpha': [0.1, 0.5, 1],  #  L1 regularization  
        'gamma': [0, 0.1, 0.2]  #  Minimum loss reduction required to make a split
    }


    if score_metric == 'kge_scorer':
       
        # Convert KGE function into a scorer for GridSearchCV
        kge_scorer = make_scorer(kling_gupta_efficiency, greater_is_better=True)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            regressor, 
            param_grid, 
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
            scoring=kge_scorer, 
            n_jobs=-1,    
            verbose=3,  # Show progress updates
            error_score="raise",
        )

    elif score_metric == 'r2':

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            regressor, 
            param_grid, 
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
            scoring='r2', 
            n_jobs=-1,    
            verbose=3,  # Show progress updates
            error_score="raise",
        )
    
    print(f"Starting GridSearchCV with {multiprocessing.cpu_count()} CPUs...")
    start_time = time.time()
    
    # Train the model using grid search
    grid_search.fit(X_train, y_train)

    end_time = time.time()
    print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds.")

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_
    print("Best hyperparameters found:", best_params)
    
    # Train the best model on the full training set
    best_regressor = grid_search.best_estimator_

    # Evaluate the final model on the test set
    final_predictions = best_regressor.predict(X_test)
    print("Out-of-sample predictions:", final_predictions)

    # Save the best trained model
    joblib.dump(best_regressor, f"{folder_path}/best_xgboost_model/best_xgboost_{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.pkl")

    return best_regressor, X_train, X_test, y_train, y_test, final_predictions

def train_xgboost_bayes(case_name, df_select, folder_path, select_model=None, select_site=None, 
                        search_space=None, score_metric='r2', n_splits=5, test_size=0.2, n_iter=50,
                        test=False, model_save_format='json'):
    
    """
    Train an XGBoost regressor using Bayesian Optimization.

    Parameters:
    - score_metric: 'r2' or 'kge'
    - n_iter: Number of Bayesian optimization iterations
    """

    # === Set X ===
    feature_names = list(df_select.columns)      # <<<【修改点 1：新增】
    X             = df_select.values

    if select_model == 'obs':
        y         = var_select[select_model].values
    else:
        y         = var_select['model_'+select_model].values

    print("NaN values in y:", np.isnan(y).sum())
    print("Infinite values in y:", np.isinf(y).sum())
    print("Max value in y:", np.max(y))
    print("Min value in y:", np.min(y))


    # === Prepare folders ===
    report_dir     = os.path.join(folder_path, f"txt/{case_name}/best_xgboost_models/reports")
    model_pred_dir = os.path.join(folder_path, f"txt/{case_name}/best_xgboost_models/predictions")
    model_dir      = os.path.join(folder_path, f"txt/{case_name}/best_xgboost_models/models")

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(model_pred_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # === Split data ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # === Define model ===
    regressor = XGBRegressor(objective="reg:squarederror", tree_method="gpu_hist", verbosity=2)

    # === Define scoring ===
    if score_metric == 'kge':
        kge_scorer = make_scorer(kling_gupta_efficiency, greater_is_better=True)
        scoring = kge_scorer
    else:
        scoring = 'r2'

    # === Bayesian Optimization ===
    bayes_search = BayesSearchCV(
        estimator=regressor,
        search_spaces=search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=3,
        random_state=42,
        error_score="raise"
    )

    # Set early stopping
    fit_params = {
        'early_stopping_rounds': 50,
        'eval_set': [(X_test, y_test)],
        'verbose': False
    }
    
    print(f"Starting BayesSearchCV with {multiprocessing.cpu_count()} CPUs for {n_iter} iterations...")
    start_time = time.time()

    bayes_search.fit(X_train, y_train, **fit_params)

    end_time = time.time()
    print(f"Bayesian optimization completed in {end_time - start_time:.2f} seconds.")

    # === Save xgboost report ===
    report_path = os.path.join(
        report_dir,
        f"xgboost_bayes_{select_model or 'model'}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.txt"
    )

    with open(report_path, "w") as f:
        f.write(f"=== XGBoost Bayesian Optimization Report ===\n")
        f.write(f"Model: {select_model}\n")
        f.write(f"Site: {select_site}\n")
        f.write(f"Metric: {score_metric}\n")
        f.write(f"Total iterations: {n_iter}\n")
        f.write(f"CV folds: {n_splits}\n")
        f.write(f"Elapsed time: {end_time - start_time:.2f} seconds\n\n")

        results = pd.DataFrame(bayes_search.cv_results_)
        results_sorted = results.sort_values(by='rank_test_score')
        f.write("=== All Trials ===\n")
        for i, row in results_sorted.iterrows():
            params = row['params']
            mean_score = row['mean_test_score']
            f.write(f"Iter {i+1:03d} | Score: {mean_score:.4f} | Params: {params}\n")

        f.write("\n=== Best Parameters ===\n")
        f.write(str(bayes_search.best_params_))
        f.write("\nBest Score: {:.4f}\n".format(bayes_search.best_score_))

    print(f"✅ Report saved to: {report_path}")

    # === Save best model ===  
    # 保存的 .pkl 文件不仅包含最优的超参数，还包含使用这些超参数训练完成的 XGBoost 模型
    best_params = bayes_search.best_params_
    print("Best hyperparameters found:", best_params)

    # 获取最优模型
    best_regressor = bayes_search.best_estimator_

    if model_save_format == 'pkl':
        model_path = os.path.join(
            model_dir,
            f"best_xgboost_bayes_{select_model or 'model'}_{select_site or 'all'}_"
            f"{score_metric}{'_test' if test else ''}.pkl"
        )

        joblib.dump(
            {
                "model": best_regressor,
                "features": feature_names    # <<<【修改点 2：新增】
            },
            model_path
        )
        print(f"✅ Model + feature order saved to: {model_path}")

    elif model_save_format == 'json':
        model_path = os.path.join(
            model_dir,
            f"best_xgboost_bayes_{select_model or 'model'}_{select_site or 'all'}_"
            f"{score_metric}{'_test' if test else ''}.json"
        )

        best_regressor.get_booster().save_model(model_path)
        print(f"✅ Model saved as JSON: {model_path}")

        # <<<【修改点 2：新增】单独保存 feature 顺序
        feature_path = model_path.replace(".json", "_features.txt")
        with open(feature_path, "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")

        print(f"✅ Feature order saved to: {feature_path}")

    # === Plot test result ===
    test_predictions = best_regressor.predict(X_test)
    plot_train_xgboost(case_name, y_test, test_predictions, folder_path, select_model, select_site, test)

    # === Save predictions ===
    final_predictions = best_regressor.predict(X)

    df_select['y']      = y
    df_select['y_pred'] = final_predictions
    pred_path = os.path.join(
        model_pred_dir,
        f"predictions_{select_model or 'model'}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.csv"
    )
    df_select.to_csv(pred_path, index=False)
    print(f"✅ Predictions saved to: {pred_path}")

    return 

def plot_train_xgboost(case_name, y_test, y_pred, folder_path, select_model=None, select_site=None, test=False):

    """
    Generates and saves three plots:
    1. Predicted vs. Observed Scatter Plot
    2. Residual Plot
    3. Histogram of Residuals

    Parameters:
    - y_test: True target values (array-like)
    - y_pred: Predicted target values (array-like)
    - select_model: Model name to include in the saved filename (default: "default_model")
    - select_site: Site name to include in the saved filename (default: "default_site")

    Returns:
    - Saves the figure to a file.
    """

    # Compute residuals
    residuals = y_test - y_pred

    # Compute statistical metrics
    r2            = r2_score(y_test, y_pred)
    rmse          = np.sqrt(mean_squared_error(y_test, y_pred))
    mean_residual = np.mean(residuals)
    std_residual  = np.std(residuals)
    skewness      = stats.skew(residuals)
    kurtosis      = stats.kurtosis(residuals)

    # Create subplots (3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. Predicted vs. Observed Scatter Plot ---
    axes[0].scatter(y_test, y_pred, alpha=0.6, facecolors="none", edgecolors="k")
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--", lw=2)  # 45-degree line
    axes[0].set_xlabel("Original Values")
    axes[0].set_ylabel("XGBoost Predictions")
    axes[0].set_title("XGBoost vs. Original")
    axes[0].grid(True)
    axes[0].text(0.05, 0.9, f"$R^2$: {r2:.3f}\nRMSE: {rmse:.3f}", transform=axes[0].transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    # --- 2. Residual Plot ---
    axes[1].scatter(y_pred, residuals, alpha=0.6, facecolors="none", edgecolors="k")
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")
    axes[1].grid(True)
    axes[1].text(0.05, 0.9, f"Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}", transform=axes[1].transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    # --- 3. Histogram of Residuals ---
    axes[2].hist(residuals, bins=25, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Residuals")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Histogram of Residuals")
    axes[2].text(0.05, 0.9, f"Skew: {skewness:.3f}\nKurtosis: {kurtosis:.3f}", transform=axes[2].transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))


    os.makedirs(f"{folder_path}/plots/{case_name}", exist_ok=True)
    
    plt.title(case_name)

    # Save the figure
    plt.savefig(f"{folder_path}/plots/{case_name}/plot_xgboost_{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.png", dpi=300, bbox_inches="tight")

    # Close the figure to free memory
    plt.close(fig)
    return

def calc_shap(case_name, df_select, df_info, folder_path,
              select_model=None, select_site=None, score_metric='r2',
              batch_size=5000, test=False, model_save_format='json'):
    

    model_path = f"{folder_path}/txt/{case_name}/best_xgboost_models/models/best_xgboost_bayes_{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.{model_save_format}"
    
    if model_save_format == 'pkl':
        bundle         = joblib.load(model_path)
        best_regressor = bundle["model"]
        feature_names  = bundle["features"]

        X = df_select[feature_names].values

    elif model_save_format == 'json':
        feature_path = model_path.replace(".json", "_features.txt")

        with open(feature_path) as f:
            feature_names = [l.strip() for l in f]

        X = df_select[feature_names].values

    # --- SHAP Analysis ---
    explainer = shap.TreeExplainer(best_regressor)

    # ------------------------------------------------------------------
    # GPU-safe SHAP computation:
    # - use batching to control memory
    # - DO NOT use joblib / multiprocessing
    # ------------------------------------------------------------------
    shap_values_list = []

    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        batch = X[i:i + batch_size]
        shap_batch = explainer(batch).values
        shap_values_list.append(shap_batch)

        # Optional progress print (useful on HPC)
        print(f"Computed SHAP for samples {i}–{min(i+batch_size, n_samples)}")

    # Combine all SHAP values
    shap_values = np.vstack(shap_values_list)

    # To pandas dataframe
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # Set up Matplotlib backend (use Agg if running on a server)
    plt.switch_backend('Agg')

    # Create SHAP summary plot
    shap.summary_plot(shap_values, shap_df, show=False)

    plt.title(case_name)

    os.makedirs(f"{folder_path}/plots/{case_name}", exist_ok=True)

    # Save the plot as a PNG file
    plt.savefig(
        f"{folder_path}/plots/{case_name}/plot_shap_summary_"
        f"{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.png",
        dpi=300, bbox_inches="tight"
    )


    # Clear the plot
    plt.clf()

    # Get the base (expected) SHAP value
    shap_df['base_shap_value'] = explainer.expected_value * np.ones(len(shap_df))

    # Align index before merge
    df_info.reset_index(drop=True, inplace=True)

    # Concatenate column-wise
    merged_df = pd.concat([df_info, shap_df], axis=1)

    # Save SHAP values

    os.makedirs(f"{folder_path}/txt/{case_name}/shap_values", exist_ok=True)

    merged_df.to_csv(
        f"{folder_path}/txt/{case_name}/shap_values/shap_values_"
        f"{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.csv",
        index=False
    )

    return

def compare_shap_predictor(models, X, feature_names, predictor_name, model_labels=None, bins=50, normalize=False):
    # MMY !!! 需要调试！！！！！
    """
    Compare the SHAP contribution of a single predictor across multiple XGBoost models.

    Parameters:
    -----------
    models : list
        List of trained XGBoost models (one per dataset).
    X : np.ndarray or pd.DataFrame
        Feature matrix (same predictors for all models).
    feature_names : list
        Names of predictors in X.
    predictor_name : str
        Name of the predictor to compare.
    model_labels : list, optional
        Names of models/datasets for legend.
    bins : int
        Number of bins to divide predictor range for averaging SHAP values.
    normalize : bool
        If True, normalize SHAP values by mean absolute SHAP across all predictors for each sample.

    Returns:
    --------
    None. Plots the comparison curve.
    """

    if isinstance(X, np.ndarray):
        X_array = X
    else:
        X_array = X.values
    
    idx = feature_names.index(predictor_name)

    if model_labels is None:
        model_labels = [f'Model {i+1}' for i in range(len(models))]

    plt.figure(figsize=(8,6))

    for model, label in zip(models, model_labels):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_array)  # [n_samples, n_features]

        if normalize:
            shap_vals = shap_vals / np.mean(np.abs(shap_vals), axis=1)[:, None]

        # Bin the predictor values
        predictor_values = X_array[:, idx]
        grid = np.linspace(predictor_values.min(), predictor_values.max(), bins+1)
        mean_shap = []
        for i in range(len(grid)-1):
            mask = (predictor_values >= grid[i]) & (predictor_values < grid[i+1])
            if mask.sum() > 0:
                mean_shap.append(shap_vals[mask, idx].mean())
            else:
                mean_shap.append(np.nan)

        plt.plot(grid[:-1], mean_shap, label=label)

    plt.xlabel(predictor_name)
    plt.ylabel('SHAP contribution')
    plt.title(f'SHAP contribution vs {predictor_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    return 

def calc_shap_vpd_contrib(case_name, df_select, folder_path, 
                          select_model=None, select_site=None, score_metric=None, 
                          batch_size=5000, test=False, model_save_format='json'):
    # MMY !!! 需要调试！！！！！
    """
    Calculate SHAP interaction values and compute VPD contribution 
    based on Tair+Qair as approximation.

    Parameters:
    - X: feature matrix (numpy array or DataFrame)
    - feature_names: list of feature names in the same order as X
    - folder_path: folder to load the trained model
    - select_model: model name
    - select_site: site name (optional)
    - batch_size: for SHAP calculation in chunks
    """

    # Load the trained model
    model_path = f"{folder_path}/txt/{case_name}/best_xgboost_models/models/best_xgboost_bayes_{select_model}_{select_site or 'all'}_{score_metric}{'_test' if test else ''}.{model_save_format}"
    
    if model_save_format == 'pkl':
        bundle         = joblib.load(model_path)
        best_regressor = bundle["model"]
        feature_names  = bundle["features"]

        X = df_select[feature_names].values

    elif model_save_format == 'json':
        feature_path = model_path.replace(".json", "_features.txt")

        with open(feature_path) as f:
            feature_names = [l.strip() for l in f]

        X = df_select[feature_names].values

    explainer      = shap.TreeExplainer(best_regressor)
    n_samples      = X.shape[0]

    # Compute SHAP interaction values in batches
    def compute_shap_inter(batch):
        return explainer.shap_interaction_values(batch)

    shap_inter_list = Parallel(n_jobs=-1)(
        delayed(compute_shap_inter)(X[i:i+batch_size]) 
        for i in range(0, n_samples, batch_size)
    )

    # Combine into a single array
    shap_inter = np.vstack(shap_inter_list)  # shape: (n_samples, n_features, n_features)
    n_features = len(feature_names)

    # Indices of Tair and Qair
    i_tair = feature_names.index('obs_Tair')
    i_qair = feature_names.index('obs_Qair')
    G_vpd = {i_tair, i_qair}

    # Compute VPD contribution per sample
    VPD_contrib = np.zeros(n_samples)
    for s in range(n_samples):
        mat = shap_inter[s]
        # Qair主效应
        main_q = mat[i_qair, i_qair]
        # Tair×Qair交互
        inter_tq = mat[i_tair, i_qair]
        # Qair与其他特征交互(除Tair)
        cross_half = 0.0
        for j in range(n_features):
            if j not in G_vpd:
                cross_half += 0.5 * mat[i_qair, j]
        # 总贡献
        VPD_contrib[s] = main_q + inter_tq + cross_half

    # 构造结果 DataFrame
    shap_columns = feature_names.copy()
    shap_columns.append('VPD_contrib')

    df_shap = pd.DataFrame(np.zeros((n_samples, len(shap_columns))), columns=shap_columns)

    # 填充原始主效应 SHAP 值（主效应+交互值对角线）
    for idx, fname in enumerate(feature_names):
        df_shap[fname] = shap_inter[:, idx, idx]

    # 添加 VPD 贡献
    df_shap['VPD_contrib'] = VPD_contrib

    # 保存
    os.makedirs(f"{folder_path}/txt/{case_name}/shap_values", exist_ok=True)

    save_path = f"{folder_path}/txt/{case_name}/shap_values/shap_values_{select_model}_{select_site or 'all'}_{score_metric}_vpd{'_test' if test else ''}.csv"
    df_shap.to_csv(save_path, index=False)
    print(f"✅ SHAP values with VPD contribution saved to: {save_path}")

    return df_shap

if __name__ == "__main__":

    # ============ Set up argument parser ============
    parser = argparse.ArgumentParser(description="Run SHAP analysis for XGBoost model.")

    parser.add_argument("--select_model", type=str, default="CABLE",  required=True, help="Model name (e.g., CABLE)")
    parser.add_argument("--select_site", type=str, default=None,  required=False, help="Site name (e.g., AU-Tum)")
    parser.add_argument("--folder_path", type=str, default="/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/SHAP",
                                         required=False, help="Path to the script folder")
    parser.add_argument("--batch_size", type=int, required=False, default=5000, help="Batch size for SHAP computation")
    parser.add_argument("--score_metric", type=str, required=False, default='r2', help="scoring metric for GridSearchCV")
    parser.add_argument("--test", action="store_true", help="whether to run on a small sample for testing")

    # Parse arguments
    args = parser.parse_args()

    # Assign arguments to variables
    select_model  = args.select_model
    select_site   = args.select_site
    folder_path   = args.folder_path
    batch_size    = args.batch_size  # Default is 5000 if not provided
    score_metric  = args.score_metric
    test          = args.test  

    # Debug print to check values (optional)
    print(f"Model: {select_model}, Site: {select_site}, Folder: {folder_path}, Batch Size: {batch_size}")


    # ============ Get data ============
    case_name  = f"{select_model}_{select_site or 'all'}"
    get_variables(case_name, folder_path, select_model, select_site, test)

    var_select = pd.read_csv(f"{folder_path}/txt/{case_name}/data_slide/data_selectec_{case_name}{'_test' if test else ''}.csv")
    
    # ============ Extract features and target ============
    '''
    Attention: feature_names must be consistent in the whole python file!!! 
    '''

    # feature_names = ['time','site_name','IGBP_type','climate_type', 'VPD', 'obs_Tair', 'obs_Qair','obs_Precip', 'obs_SWdown', 'SMtop1m', 'LAI',]
    # feature_names   = ['VPD', 'obs_Tair', 'obs_Precip', 'obs_SWdown', 'SMtop1m', 'LAI',]
    feature_names = ['obs_Qair', 'obs_Tair', 'obs_Precip', 'obs_SWdown', 'SMtop1m', 'LAI',] # 尝试通过Qair+Tair的交互贡献来代表vpd的贡献
    df_select     = var_select[feature_names]

    # =============== Train XGBoost model ===============
    
    # Define Bayesian search space
    ## Option 1 
    #search_space = {
    #    'max_depth': Integer(4, 16),
    #    'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
    #     'n_estimators': Integer(100, 1000),
    #    'subsample': Real(0.6, 1.0),
    #    'gamma': Real(0, 0.3)
    #}

    ## Option 2 
    search_space = {
        'max_depth': Integer(3, 12),
        'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
        'min_child_weight':Integer(1,10),
        'n_estimators': Integer(100, 1000),
        'subsample': Real(0.6, 0.9),
        'gamma': Real(0, 0.3)
    }

    train_xgboost_bayes(case_name, df_select, folder_path, select_model, select_site, 
                        search_space, score_metric=score_metric, test=test, model_save_format='pkl')

    # =============== Calcuate SHAP ===============
    df_info       = var_select[['time','site_name','IGBP_type','climate_type']]
    calc_shap(case_name, df_select, df_info, folder_path, 
              select_model=select_model, select_site=select_site, score_metric=score_metric,
              batch_size=batch_size, test=test, model_save_format='pkl')

    # =============== Calcuate SHAP VPD contribution ===============
    calc_shap_vpd_contrib(case_name, df_select, folder_path=folder_path, 
                        select_model=select_model, select_site=select_site, 
                        score_metric=score_metric, test=test, model_save_format='pkl')
                        