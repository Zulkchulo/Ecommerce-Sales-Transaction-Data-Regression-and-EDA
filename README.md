# E-commerce Sales Prediction

This project builds a regression pipeline for predicting **total sales
amount** in an e-commerce dataset.\
It performs **EDA, feature engineering, outlier removal, visualization,
and model comparison** across multiple algorithms.

------------------------------------------------------------------------

## Dataset

-   Input file: `ecommerce_sales_34500.csv`
-   Target variable: `total_amount`
-   Size: \~34,500 rows
-   Source: Provided dataset (can be adapted for other sales datasets).

------------------------------------------------------------------------

## Features Engineered

-   **Net Price** = `price * quantity`
-   **Discount Fraction** (normalized %)
-   **Discounted Price** = `net_price * (1 - discount_frac)`
-   **Final Price Estimate** = discounted price + shipping cost
-   **Quantity × Discount Interaction**
-   **Quantity × Customer Age Interaction**
-   **Datetime Features:** year, month, day, weekday, weekend flag, days
    since first order
-   **Frequency Encoding** for categorical variables (customer, product,
    payment method, etc.)
-   **Polynomial Interactions** on top correlated numerical features

------------------------------------------------------------------------

## Exploratory Data Analysis (EDA)

1.  **Boxplots** before and after outlier removal (via IQR method).\
2.  **Correlation Heatmap** (top 12 correlated features).\
3.  **Pairplot** (sampled, top predictors only).\
4.  **Countplots** for categorical features (top 10 categories).\
5.  **Distribution plots** for target variable (`total_amount`).

All plots are saved in the `plots/` directory.

------------------------------------------------------------------------

## Outlier Removal

-   IQR method applied on numerical features (excluding target).\
-   Boxplots saved before and after removal.

------------------------------------------------------------------------

## Models Implemented

-   Linear Regression
-   Ridge, Lasso, ElasticNet
-   Decision Tree
-   Random Forest
-   Gradient Boosting
-   XGBoost
-   LightGBM
-   CatBoost
-   Stacking Ensemble (top 3 models)

------------------------------------------------------------------------

## Hyperparameter Tuning

-   `GridSearchCV` for linear models.\
-   `RandomizedSearchCV` for tree-based and boosting models.\
-   Cross-validation: 5-fold (`KFold`).

------------------------------------------------------------------------

## Evaluation Metrics

For each model: - **MAE** (Mean Absolute Error)\
- **MSE** (Mean Squared Error)\
- **RMSE** (Root Mean Squared Error)\
- **R² Score**\
- **Cross-Validation R² Mean & Std**

A comparison table is printed and best model is saved.

------------------------------------------------------------------------

## Best Model

-   The best-performing model (highest cross-validation R² or test R²)
    is saved as:

```{=html}
<!-- -->
```
    best_model.pkl

------------------------------------------------------------------------

## Requirements

Install dependencies using:

``` bash
pip install -r requirements.txt
```

### Core Libraries

-   pandas, numpy, seaborn, matplotlib
-   scikit-learn
-   xgboost
-   lightgbm
-   catboost
-   tqdm
-   joblib

------------------------------------------------------------------------

## Outputs

-   **Plots:** Saved under `plots/`
-   **Model Performance Table:** Printed in console
-   **Best Model File:** `best_model.pkl`

------------------------------------------------------------------------

## Usage

``` bash
python main.py
```

------------------------------------------------------------------------

## Author

Generated for **E-commerce Sales Prediction Project**
