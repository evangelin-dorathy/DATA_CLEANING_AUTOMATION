import pandas as pd
import numpy as np
import os
import time

from .type_inference import infer_column_types
from .text_cleaner import resolve_text_inconsistencies
from .outlier_detector import handle_outliers
from .imputers import impute_numerical_columns, impute_categorical_columns
from .report_generator import generate_report

class AutoCleanerPipeline:
    """
    Automated data cleaning orchestrator.
    Executes a high-difficulty cleaning flow incorporating ML-based type inference,
    outlier removal, MICE/KNN missing imputation, and TF-IDF category merging.
    """
    def __init__(self, outlier_method: str = 'nullify', imputation_method: str = 'mice', 
                 similarity_threshold: float = 0.8, univariate_threshold: float = 3.5,
                 multivariate_contamination: float = 0.02):
        self.outlier_method = outlier_method
        self.imputation_method = imputation_method
        self.similarity_threshold = similarity_threshold
        self.univariate_threshold = univariate_threshold
        self.multivariate_contamination = multivariate_contamination
        
    def clean(self, df_input: pd.DataFrame, report_output_path: str = None) -> pd.DataFrame:
        start_time = time.time()
        df_before = df_input.copy()
        df = df_input.copy()
        
        print(f"Starting data cleaning on dataset with shape: {df.shape}")
        
        # 1. Type Inference
        inferred_types = infer_column_types(df)
        print("Inferred column categories:")
        for col, t in inferred_types.items():
            print(f"  - {col}: {t}")
            
        # Group columns by inferred type
        num_cols = [c for c, t in inferred_types.items() if t == 'numeric']
        cat_cols = [c for c, t in inferred_types.items() if t == 'categorical']
        date_cols = [c for c, t in inferred_types.items() if t == 'datetime']
        text_cols = [c for c, t in inferred_types.items() if t == 'text']
        bool_cols = [c for c, t in inferred_types.items() if t == 'boolean']
        id_cols = [c for c, t in inferred_types.items() if t == 'id']
        
        # 2. Datetime Normalization
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        # 3. Categorical Text Clustering & Typo Resolution
        fuzzy_mappings = {}
        for col in cat_cols:
            cleaned_series, mappings = resolve_text_inconsistencies(df[col], similarity_threshold=self.similarity_threshold)
            df[col] = cleaned_series
            fuzzy_mappings[col] = mappings
            if mappings:
                print(f"Resolved {len(mappings)} typos in column '{col}'")
                
        # 4. Outlier Handling (Numerical)
        df, outlier_log = handle_outliers(
            df, 
            numerical_cols=num_cols, 
            method=self.outlier_method,
            univariate_threshold=self.univariate_threshold,
            multivariate_contamination=self.multivariate_contamination
        )
        if outlier_log:
            print(f"Detected outliers: {outlier_log}")
            
        # 5. Missing Value Imputation
        # 5a. Numerical Imputation (uses MICE or KNN)
        df, missing_log_num = impute_numerical_columns(
            df, 
            cols=num_cols, 
            method=self.imputation_method
        )
        if missing_log_num:
            print(f"Imputed numerical NaNs: {missing_log_num}")
            
        # 5b. Categorical Imputation (uses Mode or Random Forest classifier)
        df, missing_log_cat = impute_categorical_columns(
            df, 
            cat_cols=cat_cols, 
            num_cols=num_cols
        )
        if missing_log_cat:
            print(f"Imputed categorical NaNs: {missing_log_cat}")
            
        # 6. Boolean & IDs basic handling
        for col in bool_cols:
            # Map yes/no, true/false to actual 1/0 or Boolean
            # If standard boolean values exist
            s = df[col].astype(str).str.strip().str.lower()
            val_map = {
                'true': True, '1': True, 'yes': True, 'y': True, 't': True, '1.0': True,
                'false': False, '0': False, 'no': False, 'n': False, 'f': False, '0.0': False
            }
            # Only map recognized values, keep NaN if any
            df[col] = s.map(val_map)
            # Impute booleans with mode
            if df[col].isna().any():
                bool_mode = df[col].mode()
                df[col] = df[col].fillna(bool_mode[0] if not bool_mode.empty else False)
                
        for col in id_cols:
            # Drop rows with duplicate or empty critical IDs, or fill with unique string if needed
            # For simplicity, we fill missing ID values with a generated sequence ID
            if df[col].isna().any():
                null_indices = df[df[col].isna()].index
                for i, idx in enumerate(null_indices):
                    df.loc[idx, col] = f"GEN_ID_{int(time.time())}_{idx}"
                    
        # 7. Convert types for cleaner data export
        for col in num_cols:
            # If there are no longer any NaNs, and the column values are all integers, convert to int
            if not df[col].isna().any():
                if np.array_equal(df[col], df[col].astype(int)):
                    df[col] = df[col].astype(int)
                    
        # 8. Report Generation
        if report_output_path:
            os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
            generate_report(
                df_before=df_before,
                df_after=df,
                inferred_types=inferred_types,
                missing_log_num=missing_log_num,
                missing_log_cat=missing_log_cat,
                outlier_log=outlier_log,
                fuzzy_mappings=fuzzy_mappings,
                output_filepath=report_output_path
            )
            
        print(f"Finished data cleaning pipeline in {time.time() - start_time:.2f} seconds.")
        return df
