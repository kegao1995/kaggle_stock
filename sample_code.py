# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
# Stock Return Prediction Pipeline - Complete Working Solution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
import os 
os.getcwd()

warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=== Stock Return Prediction Pipeline ===\n")

# 1. Load Data
print("1. Loading data...")
train_data = pd.read_pickle('kaggle/input/stocks-return-prediction-v-2/train_data.pkl')
test_data = pd.read_pickle('kaggle/input/stocks-return-prediction-v-2/test_data.pkl')
sample_submission = pd.read_csv('kaggle/input/stocks-return-prediction-v-2/sample_submission.csv')

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# %%

# 2. Data Preprocessing
print("\n2. Data Preprocessing...")

# Convert f_3 from object to numeric
train_data['f_3'] = pd.to_numeric(train_data['f_3'], errors='coerce')
test_data['f_3'] = pd.to_numeric(test_data['f_3'], errors='coerce')

# Check for any remaining NaN values in f_3 and fill with median
if train_data['f_3'].isnull().any():
    median_f3 = train_data['f_3'].median()
    train_data['f_3'].fillna(median_f3, inplace=True)
    test_data['f_3'].fillna(median_f3, inplace=True)

print(f"Data types after preprocessing:")
print(train_data.dtypes)


# %%

# 3. Basic EDA
print("\n3. Exploratory Data Analysis...")

# Basic statistics
print("\nTarget variable statistics:")
print(train_data['y'].describe())

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Target distribution
axes[0, 0].hist(train_data['y'], bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Target Distribution')
axes[0, 0].set_xlabel('Return (y)')

# Sample stocks over time
sample_stocks = train_data['code'].unique()[:3]
for stock in sample_stocks:
    stock_data = train_data[train_data['code'] == stock]
    axes[0, 1].plot(stock_data['date'], stock_data['y'], label=stock, alpha=0.7)
axes[0, 1].set_title('Sample Stock Returns Over Time')
axes[0, 1].set_xlabel('Date')
axes[0, 1].legend()

# Feature correlations
feature_cols = [f'f_{i}' for i in range(7)]
corr_matrix = train_data[feature_cols + ['y']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 2])
axes[0, 2].set_title('Feature Correlations')

# Feature distributions
for i, feat in enumerate(['f_0', 'f_1', 'f_2']):
    axes[1, i].hist(train_data[feat], bins=50, alpha=0.7)
    axes[1, i].set_title(f'{feat} Distribution')
    axes[1, i].set_xlabel(feat)

plt.tight_layout()
plt.show()


# %%

# 4. Feature Engineering
print("\n4. Feature Engineering...")

def create_features(df):
    """Create features ensuring no NaN values"""
    df_feat = df.copy()
    
    # Basic features
    feature_cols = [f'f_{i}' for i in range(7)]
    
    # Row statistics
    df_feat['feat_mean'] = df_feat[feature_cols].mean(axis=1)
    df_feat['feat_std'] = df_feat[feature_cols].std(axis=1)
    df_feat['feat_max'] = df_feat[feature_cols].max(axis=1)
    df_feat['feat_min'] = df_feat[feature_cols].min(axis=1)
    
    # Interaction features (top 3 only to reduce complexity)
    for i in range(3):
        for j in range(i+1, 3):
            df_feat[f'f_{i}_x_f_{j}'] = df_feat[f'f_{i}'] * df_feat[f'f_{j}']
    
    # Ratio features with safety for division
    for i in range(1, 4):
        df_feat[f'f_0_div_f_{i}'] = df_feat['f_0'] / (df_feat[f'f_{i}'] + 1e-8)
    
    # Stock-level means
    for i in range(3):
        df_feat[f'f_{i}_stock_mean'] = df_feat.groupby('code')[f'f_{i}'].transform('mean')
    
    return df_feat

# Create features
train_feat = create_features(train_data)
test_feat = create_features(test_data)

# Get feature columns
feature_columns = [col for col in train_feat.columns if col not in ['code', 'date', 'y']]
print(f"Total features: {len(feature_columns)}")
print(f"Features: {feature_columns}")

# %%

# 5. Prepare Data for Modeling
print("\n5. Preparing data for modeling...")

X_train = train_feat[feature_columns].values
y_train = train_feat['y'].values
X_test = test_feat[feature_columns].values

# Check for NaN values and handle them
if np.isnan(X_train).any():
    print("Found NaN values in training features, replacing with 0...")
    X_train = np.nan_to_num(X_train, nan=0.0)
    
if np.isnan(X_test).any():
    print("Found NaN values in test features, replacing with 0...")
    X_test = np.nan_to_num(X_test, nan=0.0)

# Split data
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_tr_scaled.shape}")
print(f"Validation set shape: {X_val_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")


# %%

# 6. Model Training
print("\n6. Training models...")

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RF': RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1),
    'GBM': GradientBoostingRegressor(n_estimators=30, max_depth=5, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_tr_scaled, y_tr)
    
    # Validate
    y_pred_val = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rank_ic = spearmanr(y_val, y_pred_val)[0]
    
    results[name] = {'RMSE': rmse, 'Rank IC': rank_ic}
    print(f"{name}: RMSE={rmse:.4f}, Rank IC={rank_ic:.4f}")
    
    # Test predictions
    predictions[name] = model.predict(X_test_scaled)

# LightGBM
print("\nTraining LightGBM...")
lgb_train = lgb.Dataset(X_tr_scaled, y_tr)
lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'verbose': -1
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

# LightGBM evaluation
y_pred_val_lgb = lgb_model.predict(X_val_scaled, num_iteration=lgb_model.best_iteration)
lgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_lgb))
lgb_rank_ic = spearmanr(y_val, y_pred_val_lgb)[0]

results['LightGBM'] = {'RMSE': lgb_rmse, 'Rank IC': lgb_rank_ic}
print(f"LightGBM: RMSE={lgb_rmse:.4f}, Rank IC={lgb_rank_ic:.4f}")

predictions['LightGBM'] = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)


# %%

# 7. Results Summary
print("\n7. Model Comparison:")
results_df = pd.DataFrame(results).T
print(results_df)


# %%

# 8. Create Submission
print("\n8. Creating submission...")

# Best model by Rank IC
best_model = results_df['Rank IC'].idxmax()
print(f"\nBest model: {best_model}")

# Create submission
submission = sample_submission.copy()
submission['y_pred'] = predictions[best_model]

# Save
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")


# %%

# 9. Ensemble submission
print("\n9. Creating ensemble submission...")

# Simple average of all models
ensemble_pred = np.mean(list(predictions.values()), axis=0)
ensemble_submission = sample_submission.copy()
ensemble_submission['y_pred'] = ensemble_pred

ensemble_submission.to_csv('ensemble_submission.csv', index=False)
print("Saved ensemble_submission.csv")


# %%

# 10. Final Summary
print("\n=== Pipeline Complete ===")
print(f"Best single model: {best_model} (Rank IC: {results_df.loc[best_model, 'Rank IC']:.4f})")
print("\nSubmission preview:")
print(submission.head())
print(f"\nPrediction range: [{submission['y_pred'].min():.2f}, {submission['y_pred'].max():.2f}]")
print(f"Original target range: [{y_train.min():.2f}, {y_train.max():.2f}]")


