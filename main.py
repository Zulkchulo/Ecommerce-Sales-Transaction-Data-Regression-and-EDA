# ===================== Importing Libraries =====================
import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from joblib import dump
from tqdm import tqdm
import math

# ===================== Environment Setup =====================
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', category=FutureWarning)
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(color_codes=True)

print("Current working directory:", os.getcwd())

# ===================== Load the Dataset =====================
df = pd.read_csv('ecommerce_sales_34500.csv')
pd.set_option("display.max_columns", None)
print("Dataset loaded successfully.")
print(df.head(5))
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# ===================== Feature Engineering (before dropping leakage columns) =====================
if set(['price', 'quantity', 'discount']).issubset(df.columns):
    df['net_price'] = df['price'] * df['quantity']
    if df['discount'].max() > 1:
        df['discount_frac'] = df['discount'] / 100.0
    else:
        df['discount_frac'] = df['discount']
    df['discounted_price'] = df['net_price'] * (1 - df['discount_frac'])
    if 'shipping_cost' in df.columns:
        df['final_price_est'] = df['discounted_price'] + df['shipping_cost']
    else:
        df['final_price_est'] = df['discounted_price']
    df['qty_discount_interaction'] = df['quantity'] * df['discount_frac']
    df['age_qty_interaction'] = df['quantity'] * df.get('customer_age', 0)

if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    if df['order_date'].notna().any():
        df['order_year'] = df['order_date'].dt.year
        df['order_month'] = df['order_date'].dt.month
        df['order_day'] = df['order_date'].dt.day
        df['order_weekday'] = df['order_date'].dt.weekday
        df['order_is_weekend'] = df['order_weekday'].isin([5,6]).astype(int)
        min_date = df['order_date'].min()
        if pd.notna(min_date):
            df['order_days_from_start'] = (df['order_date'] - min_date).dt.days

# Frequency encoding for object columns
cat_cols_initial = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols_initial:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq_enc'] = df[col].map(freq)

# ===================== Preserve original numeric list for EDA boxplots =====================
numeric_cols_initial = df.select_dtypes(include=[np.number]).columns.tolist()
if 'total_amount' in numeric_cols_initial:
    numeric_cols_initial.remove('total_amount')

# ===================== Data Cleaning (drop identifiers/leakage but after engineered features) =====================
to_drop = ['order_id', 'customer_id', 'product_id', 'price', 'profit_margin', 'shipping_cost']
to_drop = [c for c in to_drop if c in df.columns]
if to_drop:
    df.drop(columns=to_drop, inplace=True)

if 'order_date' in df.columns:
    df.drop(columns=['order_date'], inplace=True)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# ===================== EDA FIXED: Boxplots before and after outlier removal =====================
# Boxplots BEFORE IQR-based outlier removal (use numeric_cols_initial)
os.makedirs('plots', exist_ok=True)
for col in numeric_cols_initial:
    try:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col} (before outlier removal)')
        plt.tight_layout()
        plt.savefig(f'plots/boxplot_{col}_before_outlier_removal.png', dpi=200)
        plt.close()
    except Exception:
        plt.close()

# Compute IQR and remove outliers (original method: 1.5*IQR) and save boxplots AFTER removal
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'total_amount' in numerical_cols:
    numerical_cols.remove('total_amount')

Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
mask = ~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[mask].copy()
print(f"Shape after IQR-based outlier removal: {df.shape}")

# Boxplots AFTER IQR removal
numerical_cols_after = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'total_amount' in numerical_cols_after:
    numerical_cols_after.remove('total_amount')

for col in numerical_cols_after:
    try:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col} (after IQR removal)')
        plt.tight_layout()
        plt.savefig(f'plots/boxplot_{col}_after_outlier_removal.png', dpi=200)
        plt.close()
    except Exception:
        plt.close()

# ===================== EDA: Heatmaps and Pairplot (constrained to subsets) =====================
# Heatmap of missing values (small and clear)
plt.figure(figsize=(12, 3))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values')
plt.tight_layout()
plt.savefig('plots/heatmap_missing_values.png', dpi=200)
plt.close()

# Correlation heatmap but limited to top numeric features by absolute correlation with target
if 'total_amount' in df.columns:
    corr = df.select_dtypes(include=[np.number]).corr()
    target_corr = corr['total_amount'].abs().sort_values(ascending=False)
    top_features = target_corr.index.tolist()[:12]  # include target and top 11 features
    corr_sub = df[top_features].corr()
    mask = np.triu(np.ones_like(corr_sub, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, cbar_kws={"shrink": .6})
    plt.title('Heatmap of Correlation Matrix (top features)')
    plt.tight_layout()
    plt.savefig('plots/heatmap_correlation_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

# Pairplot: sample and reduce columns to top correlated numeric features plus a couple of engineered interactions
try:
    pairplot_cols = target_corr.index.tolist()[1:7]  # top 6 predictors
    # include target
    pairplot_cols = ['total_amount'] + pairplot_cols
    sample_df = df[pairplot_cols].sample(n=min(500, len(df)), random_state=42)
    g = sns.pairplot(sample_df, corner=True, plot_kws={'s': 20, 'alpha': 0.6})
    g.fig.suptitle('Pairplot (sampled, top features)', y=1.02)
    g.fig.savefig('plots/pairplot_dataset.png', dpi=200, bbox_inches='tight')
    plt.close()
except Exception:
    pass

# Countplots: show top categories only (top 10)
categorical_cols_present = [c for c in cat_cols_initial if c in df.columns]
for col in categorical_cols_present:
    try:
        top = df[col].value_counts().nlargest(10).index
        plt.figure(figsize=(8, 5))
        sns.countplot(y=df[col].where(df[col].isin(top)))
        plt.title(f'Countplot of top 10 categories in {col}')
        plt.tight_layout()
        plt.savefig(f'plots/countplot_{col}_top10.png', dpi=200)
        plt.close()
    except Exception:
        plt.close()

# Distribution and boxplot for target
if 'total_amount' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df['total_amount'], kde=True)
    plt.title('Distribution of Total Amount')
    plt.tight_layout()
    plt.savefig('plots/distribution_total_amount.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['total_amount'])
    plt.title('Boxplot of Total Amount')
    plt.tight_layout()
    plt.savefig('plots/boxplot_total_amount.png', dpi=200)
    plt.close()

# ===================== Feature Selection & Encoding (preserve earlier logic) =====================
categorical_cols_now = [c for c in cat_cols_initial if c in df.columns]
small_cardinality_cols = [c for c in categorical_cols_now if df[c].nunique() <= 8]

if small_cardinality_cols:
    df = pd.get_dummies(df, columns=small_cardinality_cols, drop_first=True)

# Remove any remaining object dtype columns to avoid model errors
remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
if remaining_object_cols:
    df.drop(columns=remaining_object_cols, inplace=True)

# Numeric scaling
num_cols_for_scaling = df.select_dtypes(include=[np.number]).columns.tolist()
if 'total_amount' in num_cols_for_scaling:
    num_cols_for_scaling.remove('total_amount')
if num_cols_for_scaling:
    scaler = preprocessing.StandardScaler()
    df[num_cols_for_scaling] = scaler.fit_transform(df[num_cols_for_scaling])

# Polynomial interaction features on top correlated numeric predictors (keep limited)
if 'total_amount' in df.columns:
    corr_abs = df.corr()['total_amount'].abs().sort_values(ascending=False)
    top_num = [c for c in corr_abs.index if c != 'total_amount'][:6]
    try:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df[top_num])
        poly_feature_names = poly.get_feature_names_out(top_num)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        single_poly_cols = [c for c in poly_df.columns if c in top_num]
        interaction_cols = [c for c in poly_df.columns if c not in single_poly_cols]
        if interaction_cols:
            corrs_poly = poly_df[interaction_cols].corrwith(df['total_amount']).abs().sort_values(ascending=False)
            keep = corrs_poly.index[:8].tolist()
            df = pd.concat([df, poly_df[keep]], axis=1)
    except Exception:
        pass

# ===================== Model Building =====================
X = df.drop('total_amount', axis=1)
y = df['total_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=98)
print(f"Shape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}, Shape of y_test: {y_test.shape}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(random_state=42, silent=True),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    "ElasticNet": ElasticNet(random_state=42)
}

param_grids = {
    "Ridge": {"regressor__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
    "Lasso": {"regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
    "ElasticNet": {"regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1], "regressor__l1_ratio": [0.1, 0.3, 0.5]},
    "Decision Tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ['auto', 'sqrt']},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5]},
    "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]},
    "LightGBM": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [31, 50]},
    "CatBoost": {"iterations": [200, 400], "learning_rate": [0.01, 0.05, 0.1], "depth": [4, 6, 8]}
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
model_performance = {}
best_model = None
best_score = -np.inf

for model_name, base_model in tqdm(models.items(), desc="Training Models"):
    print(f"\n----- {model_name} -----")
    if model_name in ["Linear Regression", "Ridge", "Lasso", "ElasticNet"]:
        pipe = Pipeline([("scaler", preprocessing.StandardScaler()), ("regressor", base_model)])
    else:
        pipe = base_model

    if model_name in ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Gradient Boosting"]:
        search_params = param_grids.get(model_name, {})
        if search_params:
            rnd = RandomizedSearchCV(pipe, search_params, n_iter=8, scoring="r2", cv=cv, n_jobs=-1, random_state=42, verbose=0)
            rnd.fit(X_train, y_train)
            final_model = rnd.best_estimator_
            print(f"Best Params (Randomized) for {model_name}: {rnd.best_params_}")
        else:
            pipe.fit(X_train, y_train)
            final_model = pipe
    elif model_name in param_grids:
        grid = GridSearchCV(pipe, param_grids[model_name], cv=cv, scoring="r2", n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        final_model = grid.best_estimator_
        print(f"Best Params (Grid) for {model_name}: {grid.best_params_}")
    else:
        pipe.fit(X_train, y_train)
        final_model = pipe

    try:
        cv_scores = cross_val_score(final_model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    except Exception:
        cv_mean, cv_std = float('nan'), float('nan')

    mae, mse, rmse, r2 = evaluate_model(final_model, X_test, y_test)
    model_performance[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2,
        'CV R2 Mean': cv_mean,
        'CV R2 Std': cv_std
    }
    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2 Test: {r2:.4f}, CV mean R2: {cv_mean:.4f} +/- {cv_std:.4f}")

    score_to_compare = cv_mean if not math.isnan(cv_mean) else r2
    if score_to_compare > best_score:
        best_score = score_to_compare
        best_model = final_model

# Stacking ensemble (top 3 by CV R2 Mean or test R2)
perf_df_tmp = pd.DataFrame(model_performance).T
perf_df_tmp['rank_metric'] = perf_df_tmp['CV R2 Mean'].fillna(perf_df_tmp['R2 Score'])
top_models = perf_df_tmp.sort_values('rank_metric', ascending=False).index.tolist()[:3]
estimators = []
for name in top_models:
    if name in models:
        estimators.append((name, models[name]))
if estimators:
    try:
        stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), n_jobs=-1)
        stack.fit(X_train, y_train)
        mae, mse, rmse, r2 = evaluate_model(stack, X_test, y_test)
        model_performance['StackingEnsemble'] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2 Score': r2,
            'CV R2 Mean': np.nan,
            'CV R2 Std': np.nan
        }
        print(f"StackingEnsemble - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2 Test: {r2:.4f}")
        if r2 > best_score:
            best_model = stack
    except Exception:
        pass

performance_df = pd.DataFrame(model_performance).T
print("\nModel Performance Comparison:")
print(performance_df.sort_values('R2 Score', ascending=False))

dump(best_model, 'best_model.pkl')
print(f"Best model saved as 'best_model.pkl' ({type(best_model).__name__}).")
  