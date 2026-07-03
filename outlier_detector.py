import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import median_abs_deviation

def detect_univariate_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Detects outliers in a 1D numerical series using the Robust Z-score method (using Median Absolute Deviation).
    Formula: Z = 0.6745 * (x - median) / MAD
    Returns a boolean mask where True indicates an outlier.
    """
    non_nulls = series.dropna()
    if len(non_nulls) < 3:
        return pd.Series(False, index=series.index)
        
    median = non_nulls.median()
    mad = median_abs_deviation(non_nulls, scale='normal') # scale='normal' multiplies by ~1.4826
    
    # Handle zero MAD to prevent division by zero
    if mad == 0:
        # Fallback to standard deviation or simple IQR if MAD is 0 (e.g. constant values)
        std = non_nulls.std()
        if std == 0:
            return pd.Series(False, index=series.index)
        z_scores = (series - median) / std
    else:
        # Robust Z-score
        z_scores = (series - median) / mad
        
    return z_scores.abs() > threshold

def detect_multivariate_outliers(df: pd.DataFrame, numerical_cols: list, contamination: float = 0.05) -> pd.Series:
    """
    Detects multivariate outliers using an Isolation Forest on the specified numerical columns.
    Returns a boolean mask where True indicates an outlier.
    """
    if not numerical_cols or len(df) < 10:
        return pd.Series(False, index=df.index)
        
    # Isolate numerical columns and impute temporarily for isolation forest
    sub_df = df[numerical_cols].copy()
    
    # Simple median imputation just for running Isolation Forest (does not modify source data)
    for col in sub_df.columns:
        if sub_df[col].isna().any():
            sub_df[col] = sub_df[col].fillna(sub_df[col].median())
            
    # If still empty or constant, return false
    if sub_df.dropna().empty:
        return pd.Series(False, index=df.index)
        
    clf = IsolationForest(contamination=contamination, random_state=42)
    # Fit and predict (-1 for outlier, 1 for inlier)
    preds = clf.fit_predict(sub_df)
    
    return pd.Series(preds == -1, index=df.index)

def handle_outliers(df: pd.DataFrame, numerical_cols: list, method: str = 'nullify', 
                    univariate_threshold: float = 3.5, multivariate_contamination: float = 0.02) -> tuple[pd.DataFrame, dict]:
    """
    Detects and handles outliers in the DataFrame.
    Methods:
      - 'nullify': Set outliers to NaN (so they can be imputed later by ML imputers).
      - 'winsorize': Cap outliers at 1.5x IQR or the MAD thresholds.
      - 'drop': Remove rows containing outliers.
    Returns:
      - Cleaned DataFrame.
      - A log dictionary detailing which outliers were caught.
    """
    df_clean = df.copy()
    outlier_log = {}
    
    # 1. Univariate Outlier Detection
    for col in numerical_cols:
        outlier_mask = detect_univariate_outliers(df_clean[col], threshold=univariate_threshold)
        num_outliers = outlier_mask.sum()
        
        if num_outliers > 0:
            outlier_log[f"{col} (univariate)"] = int(num_outliers)
            if method == 'nullify':
                df_clean.loc[outlier_mask, col] = np.nan
            elif method == 'winsorize':
                # Cap values
                median = df_clean[col].median()
                mad = median_abs_deviation(df_clean[col].dropna(), scale='normal')
                if mad > 0:
                    upper_bound = median + univariate_threshold * mad
                    lower_bound = median - univariate_threshold * mad
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'drop':
                df_clean = df_clean[~outlier_mask]
                
    # 2. Multivariate Outlier Detection
    if len(numerical_cols) >= 2:
        mv_mask = detect_multivariate_outliers(df_clean, numerical_cols, contamination=multivariate_contamination)
        num_mv_outliers = mv_mask.sum()
        
        if num_mv_outliers > 0:
            outlier_log["Multivariate (Isolation Forest)"] = int(num_mv_outliers)
            if method == 'nullify':
                # Set all numerical features of this row to NaN
                for col in numerical_cols:
                    df_clean.loc[mv_mask, col] = np.nan
            elif method == 'drop':
                df_clean = df_clean[~mv_mask]
            # Winsorization is not directly defined for multivariate outliers
            
    return df_clean, outlier_log
