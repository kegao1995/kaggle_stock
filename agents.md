# Agents Overview for `kaggle_stock`

## Project Snapshot
- **Objective**: Predict next-period stock returns (`y`) using tabular features `f_0`–`f_6` supplied by Kaggle dataset `stocks-return-prediction-v-2`.
- **Primary Script**: `sample_code.py` orchestrates the end-to-end workflow from data loading to submission artifact generation.
- **Runtime Context**: Designed for the Kaggle notebook environment (references `/kaggle/…` paths, expects matplotlib display, saves CSV outputs in working directory).

## Data Flow & Feature Engineering
- **Data Ingestion**: Uses `pd.read_pickle` for train/test features and `pd.read_csv` for the sample submission template from `kaggle/input/stocks-return-prediction-v-2`.
- **Preprocessing**: Coerces `f_3` to numeric, imputes residual nulls with the median.
- **Feature Engineering**:
  - Domain semantics: `f_2` captures daily log return `ln(P_t / P_{t-1})`, `f_3` encodes stock industry category (later coerced to numeric), `f_4` represents daily traded value.
  - Row-level statistics (mean, std, max, min across base features).
  - Limited pairwise interactions (`f_i * f_j` for top three features).
  - Safe ratios (`f_0 / f_i` with epsilon guard).
  - Stock-wise rolling context via group means per `code`.
- **Feature Set**: All engineered columns except identifiers (`code`, `date`) and target (`y`).

## Modeling Pipeline
1. **Split Strategy**: `train_test_split` with `test_size=0.2` and `random_state=42` to create validation holdout.
2. **Scaling**: Standardization using `StandardScaler` fit on training subset only.
3. **Models Trained**:
   - Linear Regression, Ridge Regression (`alpha=1.0`)
   - RandomForestRegressor (`n_estimators=30`, `max_depth=8`)
   - GradientBoostingRegressor (`n_estimators=30`, `max_depth=5`)
   - LightGBM regressor with early stopping (`num_boost_round=100`, `early_stopping_rounds=20`)
4. **Validation Metrics**: Reports RMSE and Rank Information Coefficient (Spearman correlation) for each model.
5. **Inference Artifacts**:
   - `submission.csv`: predictions from the best single model chosen by highest Rank IC.
   - `ensemble_submission.csv`: mean ensemble across all model predictions.

## Diagnostics & Visualization
- Produces summary statistics for `y` and detailed datatypes post-cleaning.
- Generates EDA plots (histograms, correlation heatmap, time-series snippets). Note: In non-notebook contexts, display requires backend configuration.
- Outputs model comparison table to stdout for quick inspection.

## Operational Considerations
- **Dependencies**: numpy, pandas, matplotlib, seaborn, scikit-learn (model_selection, preprocessing, ensemble, linear_model, metrics), lightgbm, scipy.
- **Reproducibility**: Controlled random seed (`np.random.seed(42)`), deterministic split parameters, LightGBM early stopping for stability.
- **File Expectations**: Requires Kaggle dataset mirrored locally under `kaggle/input/stocks-return-prediction-v-2`; writes outputs to project root.
- **Runtime Warnings**: Suppresses warnings globally; reconsider if debugging numerical stability.

## Extension Ideas
1. **Temporal Validation**: Replace random split with time-based or purged K-fold split to respect market chronology.
2. **Feature Enrichment**: Add lagged returns, volatility measures, sector embeddings if available.
3. **Hyperparameter Search**: Introduce cross-validated tuning (Optuna, GridSearchCV) targeting Rank IC.
4. **Model Diversity**: Explore CatBoost/XGBoost, neural tabular models, or stacking frameworks.
5. **Risk Controls**: Add post-processing steps (e.g., clipping, exposure limits) before submission files.

## Agent Task Surface
- **Data Agent**: Verify dataset availability, refresh cached inputs, and monitor schema drift.
- **Feature Agent**: Maintain and evolve `create_features`, ensuring no leakage and consistent transforms across train/test.
- **Model Agent**: Automate training loop, gather metrics, manage model registry keyed by validation Rank IC.
- **Evaluation Agent**: Track performance trends, run backtests or walk-forward validation, surface anomalies.
- **Deployment Agent**: Package best submission, handle ensemble strategies, and sync outputs to Kaggle submissions or downstream consumers.
