# %%
"""
Per-industry LightGBM training and prediction pipeline.

This script mirrors the data-loading setup from sample_code.py but
trains a dedicated LightGBM regressor for each industry slice (f_3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("kaggle/input/stocks-return-prediction-v-2")
TRAIN_FILE = DATA_ROOT / "train_data.pkl"
TEST_FILE = DATA_ROOT / "test_data.pkl"
SAMPLE_SUBMISSION_FILE = DATA_ROOT / "sample_submission.csv"
OUTPUT_FILE = Path("lightgbm_industry_submission.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric behaviour for required base features."""
    df_feat = df.copy()
    df_feat["f_3"] = pd.to_numeric(df_feat["f_3"], errors="coerce")
    return df_feat


BASE_FEATURES = ["f_0", "f_1", "f_2", "f_4", "f_5", "f_6", "f_3"]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix with selected base features."""
    cols = [c for c in BASE_FEATURES if c in df.columns]
    return df[cols]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _best_iteration(model: LGBMRegressor) -> int | None:
    """Return best iteration if available."""
    best_iter = getattr(model, "best_iteration_", None)
    return int(best_iter) if best_iter else None


def train_industry_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Dict[float, LGBMRegressor]:
    """Train one LightGBM model per industry code."""
    models: Dict[float, LGBMRegressor] = {}

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

        model = LGBMRegressor(
            objective="regression",
            n_estimators=1500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        best_iter = _best_iteration(model)
        val_pred = model.predict(X_val, num_iteration=best_iter)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"Industry {industry}: validation RMSE={val_rmse:.4f} (n={len(industry_train)})")

        models[industry] = model

    missing = set(test_df["f_3"].dropna().unique()) - set(models.keys())
    if missing:
        print(f"Warning: {len(missing)} industries in test lack trained models: {sorted(missing)}")

    return models


def generate_predictions(
    models: Dict[float, LGBMRegressor],
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
        best_iter = _best_iteration(model)
        preds.loc[test_mask] = model.predict(X_test, num_iteration=best_iter)

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
    print("=== LightGBM per-industry training ===")

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
