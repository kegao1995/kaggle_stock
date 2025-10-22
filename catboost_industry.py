# %%
"""
Per-industry CatBoost training and prediction pipeline.

This script mirrors the data-loading setup from sample_code.py but
trains a dedicated CatBoostRegressor for each industry slice (f_3).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("kaggle/input/stocks-return-prediction-v-2")
TRAIN_FILE = DATA_ROOT / "train_data.pkl"
TEST_FILE = DATA_ROOT / "test_data.pkl"
SAMPLE_SUBMISSION_FILE = DATA_ROOT / "sample_submission.csv"
OUTPUT_FILE = Path("catboost_industry_submission.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate core feature engineering with safe numeric handling."""
    df_feat = df.copy()

    feature_cols = [f"f_{i}" for i in range(7)]

    # Ensure numeric behaviour for engineered fields
    df_feat["f_3"] = pd.to_numeric(df_feat["f_3"], errors="coerce")

    # Row-wise statistics
    df_feat["feat_mean"] = df_feat[feature_cols].mean(axis=1)
    df_feat["feat_std"] = df_feat[feature_cols].std(axis=1)
    df_feat["feat_max"] = df_feat[feature_cols].max(axis=1)
    df_feat["feat_min"] = df_feat[feature_cols].min(axis=1)

    # Pairwise interactions (limit to top 3 for tractability)
    for i in range(3):
        for j in range(i + 1, 3):
            df_feat[f"f_{i}_x_f_{j}"] = df_feat[f"f_{i}"] * df_feat[f"f_{j}"]

    # Safe ratios (epsilon avoids division-by-zero)
    for i in range(1, 4):
        df_feat[f"f_0_div_f_{i}"] = df_feat["f_0"] / (df_feat[f"f_{i}"] + 1e-8)

    # Stock-level aggregate statistics
    for i in range(3):
        df_feat[f"f_{i}_stock_mean"] = df_feat.groupby("code")[f"f_{i}"].transform("mean")

    return df_feat


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix excluding identifiers and target."""
    drop_cols = {"code", "date", "y"}
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_industry_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Dict[float, CatBoostRegressor]:
    """Train one CatBoost model per industry code."""
    models: Dict[float, CatBoostRegressor] = {}

    industries = sorted(train_df["f_3"].dropna().unique())
    print(f"Training {len(industries)} industry-specific models...")

    for industry in industries:
        industry_mask = train_df["f_3"] == industry
        industry_train = train_df.loc[industry_mask]

        if len(industry_train) < 50:
            print(f"Skipping industry {industry}: insufficient rows ({len(industry_train)})")
            continue

        X = industry_train[feature_cols].drop(columns=["f_3"], errors="ignore")
        y = industry_train["y"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        model = CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            iterations=1000,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE,
            early_stopping_rounds=50,
            verbose=False,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"Industry {industry}: validation RMSE={val_rmse:.4f} (n={len(industry_train)})")

        models[industry] = model

    missing = set(test_df["f_3"].dropna().unique()) - set(models.keys())
    if missing:
        print(f"Warning: {len(missing)} industries in test lack trained models: {sorted(missing)}")

    return models


def generate_predictions(
    models: Dict[float, CatBoostRegressor],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    """Produce predictions aligning with test_df index order."""
    preds = pd.Series(np.nan, index=test_df.index)

    for industry, model in models.items():
        test_mask = test_df["f_3"] == industry
        if not test_mask.any():
            continue
        X_test = test_df.loc[test_mask, feature_cols].drop(columns=["f_3"], errors="ignore")
        preds.loc[test_mask] = model.predict(X_test)

    # Fallback: assign global mean to rows without predictions
    if preds.isna().any():
        global_mean = train_df["y"].mean()
        num_missing = preds.isna().sum()
        print(f"Filling {num_missing} missing predictions with global mean {global_mean:.6f}")
        preds.fillna(global_mean, inplace=True)

    return preds


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== CatBoost per-industry training ===")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing training data at {TRAIN_FILE}")

    train_raw = pd.read_pickle(TRAIN_FILE)
    test_raw = pd.read_pickle(TEST_FILE)
    submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)

    # Ensure industry code (f_3) semantics are preserved pre-encoding
    train_raw["f_3"] = train_raw["f_3"].astype(float)
    test_raw["f_3"] = pd.to_numeric(test_raw["f_3"], errors="coerce")

    train_feat = create_features(train_raw)
    test_feat = create_features(test_raw)

    feature_cols = list(build_feature_matrix(train_feat).columns)
    print(f"Total feature count per industry model: {len(feature_cols) - 1}")  # minus dropped f_3

    models = train_industry_models(train_feat, test_feat, feature_cols)
    if not models:
        raise RuntimeError("No industry models were trained. Check data coverage.")

    predictions = generate_predictions(models, train_feat, test_feat, feature_cols)

    submission["y_pred"] = predictions.values
    submission.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved predictions to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

# %%
