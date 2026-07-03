import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier

def impute_numerical_columns(df: pd.DataFrame, cols: list, method: str = 'mice', n_neighbors: int = 5) -> tuple[pd.DataFrame, dict]:
    """
    Imputes missing values in numerical columns using Advanced ML Imputation.
    Methods:
      - 'mice': Multivariate Imputation by Chained Equations (IterativeImputer).
      - 'knn': K-Nearest Neighbors Imputer (KNNImputer).
      - 'median': Statistical Median Imputation.
    """
    df_imputed = df.copy()
    imputation_log = {}
    
    if not cols:
        return df_imputed, imputation_log
        
    cols_to_impute = [col for col in cols if df_imputed[col].isna().any()]
    
    if not cols_to_impute:
        return df_imputed, imputation_log
        
    for col in cols_to_impute:
        missing_count = df_imputed[col].isna().sum()
        imputation_log[col] = int(missing_count)
        
    if method == 'mice':
        # Use IterativeImputer (MICE)
        # Note: We use all numeric columns to build the relationships, not just the missing ones
        imputer = IterativeImputer(max_iter=10, random_state=42)
        df_imputed[cols] = imputer.fit_transform(df_imputed[cols])
        
    elif method == 'knn':
        # Use KNN Imputer
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed[cols] = imputer.fit_transform(df_imputed[cols])
        
    else:
        # Fallback to median
        for col in cols:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val if not pd.isna(median_val) else 0)
            
    return df_imputed, imputation_log

def impute_categorical_columns(df: pd.DataFrame, cat_cols: list, num_cols: list, 
                               missing_threshold: float = 0.4) -> tuple[pd.DataFrame, dict]:
    """
    Imputes missing values in categorical columns.
    For high missingness (> missing_threshold), it creates an 'Unknown' category.
    For moderate missingness, it trains a Random Forest Classifier to predict the category if numerical columns exist.
    Otherwise, it falls back to the mode.
    """
    df_imputed = df.copy()
    imputation_log = {}
    
    if not cat_cols:
        return df_imputed, imputation_log
        
    for col in cat_cols:
        missing_count = df_imputed[col].isna().sum()
        if missing_count == 0:
            continue
            
        imputation_log[col] = int(missing_count)
        missing_ratio = missing_count / len(df)
        
        # High missingness: Label as 'Unknown'
        if missing_ratio >= missing_threshold:
            # First, convert to string category or object
            df_imputed[col] = df_imputed[col].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
            df_imputed[col] = df_imputed[col].fillna('Unknown')
            continue
            
        # Moderate missingness: Try to predict using Random Forest
        # We need numerical columns and at least some categories to train
        non_null_mask = df_imputed[col].notna()
        null_mask = df_imputed[col].isna()
        
        if len(num_cols) >= 1 and non_null_mask.sum() > 10 and null_mask.sum() > 0:
            try:
                # Prepare training data
                X_train = df_imputed.loc[non_null_mask, num_cols].copy()
                y_train = df_imputed.loc[non_null_mask, col].astype(str)
                
                # Fill temporary NaNs in X_train for classifier (should be clean from numerical imputer already)
                X_train = X_train.fillna(X_train.median())
                
                # Predictors
                X_predict = df_imputed.loc[null_mask, num_cols].copy()
                X_predict = X_predict.fillna(X_train.median())
                
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X_train, y_train)
                
                predictions = clf.predict(X_predict)
                df_imputed.loc[null_mask, col] = predictions
                continue
            except Exception:
                # If training fails, fall back to Mode
                pass
                
        # Fallback: Mode Imputation
        mode_series = df_imputed[col].mode()
        mode_val = mode_series[0] if not mode_series.empty else 'Unknown'
        df_imputed[col] = df_imputed[col].fillna(mode_val)
        
    return df_imputed, imputation_log
