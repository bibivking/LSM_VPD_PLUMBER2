import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from xgboost import XGBRegressor
import logging
import multiprocessing
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np

def 
# Get predictions for the test set
y_pred = model.predict(X_test)

# Compute residuals
residuals = y_test - y_pred

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 1. Predicted vs. Observed Scatter Plot ---
axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--", lw=2)  # 45-degree line
axes[0].set_xlabel("Observed Values")
axes[0].set_ylabel("Predicted Values")
axes[0].set_title("Predicted vs. Observed")
axes[0].grid(True)

# --- 2. Residual Plot ---
axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")
axes[1].grid(True)

# --- 3. Histogram of Residuals ---
axes[2].hist(residuals, bins=25, edgecolor="black", alpha=0.7)
axes[2].set_xlabel("Residuals")
axes[2].set_ylabel("Frequency")
axes[2].set_title("Histogram &#8203;:contentReference[oaicite:0]{index=0}&#8203;
