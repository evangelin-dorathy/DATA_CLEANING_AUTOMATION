import pandas as pd
import numpy as np

def infer_column_types(df: pd.DataFrame, datetime_threshold: float = 0.7, categorical_threshold: float = 0.05, max_categorical_unique: int = 20) -> dict:
    """
    Intelligently infer semantic types of columns in a pandas DataFrame.
    Returns a dictionary mapping column names to inferred types:
    'numeric', 'categorical', 'datetime', 'boolean', 'text', 'id'
    """
    inferred_types = {}
    total_rows = len(df)
    
    if total_rows == 0:
        return {col: 'text' for col in df.columns}

    for col in df.columns:
        # Clean null values for inference
        series = df[col].dropna()
        if len(series) == 0:
            inferred_types[col] = 'text'  # Default for all-null columns
            continue
            
        unique_vals = series.unique()
        num_unique = len(unique_vals)
        unique_ratio = num_unique / len(series)
        
        col_name_lower = str(col).lower()
        
        # Check for Boolean
        if num_unique <= 2:
            # Check if values look like boolean values
            bool_likes = {'true', 'false', 'yes', 'no', 'y', 'n', 't', 'f', '1', '0', '1.0', '0.0'}
            str_vals = set(series.astype(str).str.strip().str.lower().unique())
            if str_vals.issubset(bool_likes) or series.dtype == bool:
                inferred_types[col] = 'boolean'
                continue

        # Check for ID columns (e.g., high uniqueness, sequential numbers, or ID-like names)
        id_keywords = {'id', 'key', 'code', 'pk', 'uuid', 'guid', 'index'}
        is_id_name = any(kw in col_name_lower or col_name_lower.endswith('_' + kw) or col_name_lower.startswith(kw + '_') for kw in id_keywords)
        
        if is_id_name and (unique_ratio > 0.95 or (pd.api.types.is_integer_dtype(series.dtype) and unique_ratio > 0.5)):
            inferred_types[col] = 'id'
            continue
            
        # Check Numeric (float or int)
        if pd.api.types.is_numeric_dtype(series.dtype):
            # Sometimes numerical columns with very few integers are actually categorical (like code or rating)
            if pd.api.types.is_integer_dtype(series.dtype) and num_unique <= max_categorical_unique and unique_ratio < categorical_threshold:
                inferred_types[col] = 'categorical'
            else:
                inferred_types[col] = 'numeric'
            continue

        # Check Datetime
        # Try to parse string columns as dates
        try:
            # We try to convert to datetime. If a high percentage converts successfully, it's datetime.
            parsed_dates = pd.to_datetime(series, errors='coerce')
            success_ratio = parsed_dates.notna().sum() / len(series)
            if success_ratio >= datetime_threshold:
                inferred_types[col] = 'datetime'
                continue
        except Exception:
            pass

        # String / Object columns
        # Differentiate between 'categorical', 'text', and 'id'
        avg_word_count = series.astype(str).str.split().apply(len).mean()
        avg_char_length = series.astype(str).str.len().mean()
        
        if avg_word_count > 3.5:
            inferred_types[col] = 'text'
        elif unique_ratio > 0.9 and avg_char_length > 5 and num_unique > max_categorical_unique:
            # Likely an ID-like code (e.g., SKU, email, order ref)
            inferred_types[col] = 'id'
        else:
            inferred_types[col] = 'categorical'
            
    return inferred_types

if __name__ == "__main__":
    # Quick verification code
    data = {
        'user_id': ['USR001', 'USR002', 'USR003', 'USR004', 'USR005'],
        'age': [25, 40, np.nan, 35, 50],
        'signup_date': ['2023-01-01', '02/15/2023', '2023/06/10', '2023-11-20', 'Mar 12, 2023'],
        'city': ['New York', 'Los Angeles', 'New York', 'Boston', 'Chicago'],
        'feedback': ['Great app, loved it!', 'Too many bugs', 'Okay experience', 'Excellent support!', 'Will use again'],
        'active': ['yes', 'no', 'yes', 'yes', 'no']
    }
    df = pd.DataFrame(data)
    types = infer_column_types(df)
    print("Inferred Types:")
    for col, t in types.items():
        print(f"  {col}: {t}")
