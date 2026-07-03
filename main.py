import os
import argparse
import pandas as pd
import numpy as np
import random
from src.pipeline import AutoCleanerPipeline

def generate_synthetic_dirty_data(filepath: str, n_rows: int = 500):
    """
    Generates a highly synthetic dirty dataset containing:
    - Missing numerical, categorical, ID, boolean, and date values
    - String typos and casing anomalies (e.g. "new york", "New Yrok", "New York City")
    - Outliers (e.g. age = 999, age = -5, salary = 9,999,999)
    - Mixed date formats (ISO, US format, month text)
    - Unparseable date strings
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    np.random.seed(42)
    random.seed(42)
    
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco"]
    city_variants = {
        "New York": ["New York", "new york", "New York City", "New Yrok", "N.Y."],
        "Los Angeles": ["Los Angeles", "los angeles", "L.A.", "Los Angeles!"],
        "Chicago": ["Chicago", "chicago", "Chicagoo"],
        "Houston": ["Houston", "houston"],
        "San Francisco": ["San Francisco", "san francisco", "SF", "San Fran"]
    }
    
    dates = [
        "2023-01-15", "12/25/2022", "2024/05/01", "Oct 12, 2021", 
        "2020-11-30", "07-04-2023", "Feb 28, 2024", "invalid_date_xyz"
    ]
    
    data = []
    for i in range(n_rows):
        # 1. ID Column (introduce rare NaNs)
        cust_id = f"CUST-{1000 + i:04d}" if random.random() > 0.02 else np.nan
        
        # 2. Age (numerical - introduce NaNs and extreme outliers)
        r = random.random()
        if r < 0.10: # Missing
            age = np.nan
        elif r < 0.12: # Extreme high outlier
            age = random.choice([150, 250, 999])
        elif r < 0.13: # Extreme low outlier
            age = random.choice([-5, -50])
        else: # Normal distribution
            age = int(np.random.normal(38, 12))
            age = max(18, min(age, 90)) # clip standard range
            
        # 3. Income (numerical - linked to age, introduce NaNs, univariate & multivariate outliers)
        r = random.random()
        if r < 0.08: # Missing
            income = np.nan
        elif r < 0.10: # Extreme high outlier
            income = random.choice([5000000, 10000000])
        elif age is not None and not np.isnan(age) and age < 22 and r < 0.15: 
            # Multivariate Outlier: Very young age (e.g. 18-20) with very high income
            income = random.randint(350000, 500000)
        else: # Standard income roughly scaled by age
            base_income = 30000 + (age - 18) * 1500 if (age is not None and not np.isnan(age)) else 60000
            income = int(np.random.normal(base_income, 15000))
            income = max(15000, income)
            
        # 4. City (categorical with spelling/casing typos)
        base_city = random.choice(cities)
        city = random.choice(city_variants[base_city])
        if random.random() < 0.06: # Missing category
            city = np.nan
            
        # 5. Signup Date (mixed formats)
        r = random.random()
        if r < 0.05:
            date_str = np.nan
        else:
            date_str = random.choice(dates)
            
        # 6. Newsletter Subscription (boolean with mixed representations and missingness)
        r = random.random()
        if r < 0.06:
            subscribed = np.nan
        else:
            subscribed = random.choice(["yes", "no", "YES", "NO", "y", "n", "True", "False"])
            
        # 7. Customer Feedback (text column - longer string)
        feedbacks = [
            "Excellent user experience! Will buy again.",
            "Customer support was incredibly slow and unhelpful.",
            "Average product, does the job but nothing special.",
            "Worst service ever. App crashes continuously on startup.",
            "I really like the new dark mode design, it is much easier on the eyes.",
            "The shipment arrived 3 days late. Not happy.",
            "Absolutely amazing value for the price!"
        ]
        feedback = random.choice(feedbacks) if random.random() > 0.15 else np.nan
        
        data.append({
            'customer_id': cust_id,
            'age': age,
            'annual_income': income,
            'city': city,
            'signup_date': date_str,
            'newsletter_subscriber': subscribed,
            'customer_feedback': feedback
        })
        
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic dirty dataset with {n_rows} rows at: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="AutoSanitizer: High-difficulty Automated Data Cleaning Tool")
    parser.add_argument('--input', type=str, help="Path to input dirty CSV file")
    parser.add_argument('--output', type=str, default="data/cleaned_dataset.csv", help="Path to save cleaned CSV")
    parser.add_argument('--report', type=str, default="data/cleaning_report.html", help="Path to save HTML report")
    parser.add_argument('--imputer', type=str, choices=['mice', 'knn', 'median'], default='mice', help="ML Imputation method")
    parser.add_argument('--outlier-method', type=str, choices=['nullify', 'winsorize', 'drop'], default='nullify', help="Outlier handling method")
    parser.add_argument('--similarity', type=float, default=0.75, help="Fuzzy string similarity threshold (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    input_path = args.input
    if not input_path:
        # If no input path is provided, generate the synthetic dirty dataset automatically
        input_path = "data/sample_dirty.csv"
        generate_synthetic_dirty_data(input_path, n_rows=500)
        
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return
        
    df_dirty = pd.read_csv(input_path)
    
    # Initialize pipeline
    pipeline = AutoCleanerPipeline(
        outlier_method=args.outlier_method,
        imputation_method=args.imputer,
        similarity_threshold=args.similarity
    )
    
    # Run the pipeline
    df_clean = pipeline.clean(df_dirty, report_output_path=args.report)
    
    # Save the output CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_clean.to_csv(args.output, index=False)
    print(f"Saved clean dataset to: {args.output}")
    print("=" * 60)
    print("CLEANING COMPLETED SUCCESSFULLY!")
    print(f"Before cleaning null cells: {df_dirty.isna().sum().sum()}")
    print(f"After cleaning null cells:  {df_clean.isna().sum().sum()}")
    print(f"Audited changes saved to HTML report: {args.report}")
    print("=" * 60)

if __name__ == "__main__":
    main()
