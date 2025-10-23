# %%
"""
Enhanced LightGBM training pipeline with temporal validation.

Improvements:
- Treats f_3 as categorical feature with target encoding
- Adds rolling window features (lags, volatility, momentum)
- Time-based validation split with RMSE and Spearman IC metrics
- Single unified model instead of per-industry splits
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("kaggle/input/stocks-return-prediction-v-2")
TRAIN_FILE = DATA_ROOT / "train_data.pkl"
TEST_FILE = DATA_ROOT / "test_data.pkl"
SAMPLE_SUBMISSION_FILE = DATA_ROOT / "sample_submission.csv"
OUTPUT_FILE = Path("lightgbm_enhanced_submission.csv")

RANDOM_STATE = 42
# 时间切片验证：使用最后20%时间作为验证集
TIME_VAL_RATIO = 0.2

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时序特征：滚动窗口统计、波动率、动量指标。"""
    df_feat = df.copy()
    
    # 确保 f_3 为数值类型（行业代码）
    df_feat["f_3"] = pd.to_numeric(df_feat["f_3"], errors="coerce")
    
    # 按 code 和 date 排序以便计算时序特征
    if "date" in df_feat.columns and "code" in df_feat.columns:
        df_feat = df_feat.sort_values(["code", "date"]).reset_index(drop=True)
        
        # 滚动窗口特征（按股票分组）
        for window in [5, 10, 20]:
            # f_2 (log return) 的滚动统计
            df_feat[f"f_2_roll_mean_{window}"] = df_feat.groupby("code")["f_2"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df_feat[f"f_2_roll_std_{window}"] = df_feat.groupby("code")["f_2"].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            
            # f_4 (traded value) 的滚动统计
            df_feat[f"f_4_roll_mean_{window}"] = df_feat.groupby("code")["f_4"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # 波动率特征（基于 f_2 的滚动标准差）
        df_feat["volatility_5d"] = df_feat.groupby("code")["f_2"].transform(
            lambda x: x.rolling(5, min_periods=1).std()
        )
        df_feat["volatility_20d"] = df_feat.groupby("code")["f_2"].transform(
            lambda x: x.rolling(20, min_periods=1).std()
        )
        
        # 动量指标（短期与长期均值比率）
        short_ma = df_feat.groupby("code")["f_2"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        long_ma = df_feat.groupby("code")["f_2"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df_feat["momentum_ratio"] = short_ma / (long_ma + 1e-8)
        
        # 滞后特征（避免前视偏差：使用 shift）
        for lag in [1, 2, 5]:
            df_feat[f"f_2_lag_{lag}"] = df_feat.groupby("code")["f_2"].shift(lag)
            df_feat[f"f_4_lag_{lag}"] = df_feat.groupby("code")["f_4"].shift(lag)
    
    # 基础特征工程（保留原有逻辑）
    # 行级统计
    base_features = ["f_0", "f_1", "f_2", "f_4", "f_5", "f_6"]
    valid_base = [c for c in base_features if c in df_feat.columns]
    df_feat["row_mean"] = df_feat[valid_base].mean(axis=1)
    df_feat["row_std"] = df_feat[valid_base].std(axis=1)
    df_feat["row_max"] = df_feat[valid_base].max(axis=1)
    df_feat["row_min"] = df_feat[valid_base].min(axis=1)
    
    # 安全比率（epsilon guard）
    if "f_0" in df_feat.columns:
        for col in ["f_1", "f_2", "f_4"]:
            if col in df_feat.columns:
                df_feat[f"f_0_div_{col}"] = df_feat["f_0"] / (df_feat[col].abs() + 1e-8)
    
    # 交互特征（top 3 features）
    if all(c in df_feat.columns for c in ["f_0", "f_1", "f_2"]):
        df_feat["f_0_x_f_1"] = df_feat["f_0"] * df_feat["f_1"]
        df_feat["f_0_x_f_2"] = df_feat["f_0"] * df_feat["f_2"]
        df_feat["f_1_x_f_2"] = df_feat["f_1"] * df_feat["f_2"]
    
    return df_feat


def compute_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_col: str = "f_3",
    target_col: str = "y",
    smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对分类变量进行 target encoding（带平滑避免过拟合）。"""
    # 计算全局均值
    global_mean = train_df[target_col].mean()
    
    # 计算每个行业的统计量
    agg = train_df.groupby(cat_col)[target_col].agg(["mean", "count"])
    
    # 平滑处理：加权平均（行业均值 vs 全局均值）
    agg["smoothed_mean"] = (
        agg["count"] * agg["mean"] + smoothing * global_mean
    ) / (agg["count"] + smoothing)
    
    # 映射到原始数据
    encoding_map = agg["smoothed_mean"].to_dict()
    train_df[f"{cat_col}_target_enc"] = train_df[cat_col].map(encoding_map).fillna(global_mean)
    test_df[f"{cat_col}_target_enc"] = test_df[cat_col].map(encoding_map).fillna(global_mean)
    
    return train_df, test_df


# ---------------------------------------------------------------------------
# Time-based validation
# ---------------------------------------------------------------------------


def time_based_split(
    df: pd.DataFrame, val_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按时间排序切分训练集和验证集。"""
    if "date" not in df.columns:
        raise ValueError("需要 date 列进行时间切片")
    
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_ratio))
    
    train_split = df_sorted.iloc[:split_idx]
    val_split = df_sorted.iloc[split_idx:]
    
    print(f"时间切片：训练集 {len(train_split)} 行，验证集 {len(val_split)} 行")
    print(f"训练集时间范围: {train_split['date'].min()} - {train_split['date'].max()}")
    print(f"验证集时间范围: {val_split['date'].min()} - {val_split['date'].max()}")
    
    return train_split, val_split


def evaluate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """计算 RMSE 和 Spearman IC（Rank Information Coefficient）。"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Spearman 相关系数（处理 NaN）
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    if mask.sum() > 10:
        ic, _ = spearmanr(y_true[mask], y_pred[mask])
    else:
        ic = np.nan
    
    return {"rmse": rmse, "spearman_ic": ic}


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_unified_model(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> Tuple[LGBMRegressor, dict]:
    """训练单一 LightGBM 模型并返回验证指标。"""
    # 时间切片验证
    train_split, val_split = time_based_split(train_df, val_ratio=TIME_VAL_RATIO)
    
    X_train = train_split[feature_cols]
    y_train = train_split["y"]
    X_val = val_split[feature_cols]
    y_val = val_split["y"]
    
    # 识别分类特征（f_3 原始 + target encoding）
    categorical_features = [c for c in feature_cols if c in ["f_3"]]
    
    model = LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_samples=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=categorical_features if categorical_features else "auto",
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100),
        ],
    )
    
    # 验证集评估
    val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    val_metrics = evaluate_metrics(y_val, val_pred)
    
    print(f"\n=== 验证集指标 ===")
    print(f"RMSE: {val_metrics['rmse']:.6f}")
    print(f"Spearman IC: {val_metrics['spearman_ic']:.6f}")
    print(f"Best iteration: {model.best_iteration_}")
    
    return model, val_metrics


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------


def generate_predictions(
    model: LGBMRegressor,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """生成测试集预测，使用 best_iteration。"""
    X_test = test_df[feature_cols]
    
    best_iter = getattr(model, "best_iteration_", None)
    preds = model.predict(X_test, num_iteration=best_iter)
    
    return pd.Series(preds, index=test_df.index)


def print_feature_importance(
    model: LGBMRegressor,
    feature_cols: list[str],
    top_n: int = 20,
) -> None:
    """打印特征重要性（Top N）。"""
    importance = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).head(top_n)
    
    print(f"\n=== Top {top_n} 特征重要性 ===")
    print(feat_imp_df.to_string(index=False))


# ----------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """主执行流程：数据加载 → 特征工程 → 模型训练 → 预测 → 提交文件生成。"""
    print("=" * 80)
    print("LightGBM Enhanced Pipeline - Time Series Feature Engineering")
    print("=" * 80)
    
    # ---------------------------------------------------------------------------
    # 1. 数据加载
    # ---------------------------------------------------------------------------
    print("\n[1/6] 加载数据...")
    train_raw = pd.read_pickle(TRAIN_FILE)
    test_raw = pd.read_pickle(TEST_FILE)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    
    print(f"训练集原始形状: {train_raw.shape}")
    print(f"测试集原始形状: {test_raw.shape}")
    print(f"时间范围: {train_raw['date'].min()} ~ {test_raw['date'].max()}")
    
    # ---------------------------------------------------------------------------
    # 2. 数据预处理
    # ---------------------------------------------------------------------------
    print("\n[2/6] 数据预处理...")
    # 确保 f_3 为数值类型
    train_raw["f_3"] = pd.to_numeric(train_raw["f_3"], errors="coerce")
    test_raw["f_3"] = pd.to_numeric(test_raw["f_3"], errors="coerce")
    
    # 中位数填充缺失值（避免前视偏差：仅使用训练集统计量）
    fill_values = train_raw.median(numeric_only=True)
    train_raw = train_raw.fillna(fill_values)
    test_raw = test_raw.fillna(fill_values)
    
    print(f"训练集缺失值: {train_raw.isnull().sum().sum()}")
    print(f"测试集缺失值: {test_raw.isnull().sum().sum()}")
    
    # ---------------------------------------------------------------------------
    # 3. 特征工程
    # ---------------------------------------------------------------------------
    print("\n[3/6] 特征工程（时序特征 + target encoding）...")
    
    # 时序特征
    train_feat = create_temporal_features(train_raw)
    test_feat = create_temporal_features(test_raw)
    
    # Target encoding（仅针对 f_3 行业分类）
    train_feat, test_feat = compute_target_encoding(
        train_feat, test_feat, cat_col="f_3", target_col="y", smoothing=10.0
    )
    
    # 填充时序特征产生的 NaN（滚动窗口/滞后特征在初始时期会产生 NaN）
    temporal_fill_values = train_feat.median(numeric_only=True)
    train_feat = train_feat.fillna(temporal_fill_values)
    test_feat = test_feat.fillna(temporal_fill_values)
    
    print(f"特征工程后训练集形状: {train_feat.shape}")
    print(f"特征工程后测试集形状: {test_feat.shape}")
    
    # 确定特征列（排除标识符和目标变量）
    exclude_cols = {"code", "date", "y"}
    feature_cols = [c for c in train_feat.columns if c not in exclude_cols]
    print(f"总特征数: {len(feature_cols)}")
    print(f"特征列示例: {feature_cols[:10]}")
    
    # ---------------------------------------------------------------------------
    # 4. 模型训练（时间切片验证）
    # ---------------------------------------------------------------------------
    print("\n[4/6] 训练 LightGBM 模型（时间切片验证）...")
    model, val_metrics = train_unified_model(train_feat, feature_cols)
    
    # ---------------------------------------------------------------------------
    # 5. 特征重要性分析
    # ---------------------------------------------------------------------------
    print_feature_importance(model, feature_cols, top_n=30)
    
    # ---------------------------------------------------------------------------
    # 6. 测试集预测 & 生成提交文件
    # ---------------------------------------------------------------------------
    print("\n[5/6] 生成测试集预测...")
    test_preds = generate_predictions(model, test_feat, feature_cols)
    
    # 构建提交文件
    submission = sample_sub.copy()
    submission["y"] = test_preds.values
    
    # 保存提交文件
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[6/6] 提交文件已保存: {OUTPUT_FILE}")
    print(f"提交文件形状: {submission.shape}")
    print(f"预测值统计:\n{submission['y'].describe()}")
    
    # ---------------------------------------------------------------------------
    # 最终总结
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("训练完成！关键指标总结：")
    print("=" * 80)
    print(f"验证集 RMSE:       {val_metrics['rmse']:.6f}")
    print(f"验证集 Spearman IC: {val_metrics['spearman_ic']:.6f}")
    print(f"Best Iteration:    {model.best_iteration_}")
    print(f"特征数量:          {len(feature_cols)}")
    print(f"提交文件:          {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()