import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from jinja2 import Template

def generate_base64_plot(fig) -> str:
    """Converts a matplotlib figure to a base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def create_missingness_chart(df_before: pd.DataFrame, df_after: pd.DataFrame) -> str:
    """Generates a comparison bar chart of missing values before and after cleaning."""
    missing_before = df_before.isna().sum()
    missing_after = df_after.isna().sum()
    
    cols = [col for col in df_before.columns if missing_before[col] > 0 or missing_after[col] > 0]
    if not cols:
        # Generate placeholder indicating no missing values
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No missing values detected in either dataset.", 
                ha='center', va='center', fontsize=12, color='#64748b')
        ax.axis('off')
        return generate_base64_plot(fig)
        
    x = np.arange(len(cols))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.2), 4))
    
    # Modern styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#475569')
    
    ax.bar(x - width/2, [missing_before[c] for c in cols], width, label='Before Cleaning', color='#f43f5e', alpha=0.85)
    ax.bar(x + width/2, [missing_after[c] for c in cols], width, label='After Cleaning', color='#10b981', alpha=0.85)
    
    ax.set_ylabel('Count of Missing Values', color='#475569', fontweight='bold')
    ax.set_title('Missing Values Profile (Before vs. After)', fontsize=14, pad=15, color='#1e293b', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.legend(frameon=False)
    
    fig.tight_layout()
    return generate_base64_plot(fig)

def create_distribution_chart(df_before: pd.DataFrame, df_after: pd.DataFrame, col: str) -> str:
    """Generates distribution comparison for a numerical column."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#475569')
    
    # Drop NaNs and clip for display representation
    b_data = df_before[col].dropna()
    a_data = df_after[col].dropna()
    
    # Draw densities/histograms
    ax.hist(b_data, bins=20, alpha=0.5, label='Original (with outliers)', color='#ef4444', density=True)
    ax.hist(a_data, bins=20, alpha=0.5, label='Cleaned (imputed/winsorized)', color='#06b6d4', density=True)
    
    ax.set_title(f'Distribution of {col}', fontsize=12, pad=10, color='#1e293b', fontweight='bold')
    ax.legend(frameon=False)
    
    fig.tight_layout()
    return generate_base64_plot(fig)

# HTML Template with Jinja2
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoSanitizer Cleaning Audit Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #4f46e5;
            --primary-light: #e0e7ff;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --border: #e2e8f0;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            line-height: 1.5;
            padding: 2.5rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand h1 {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-main);
            background: linear-gradient(135deg, var(--primary), #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand p {
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .timestamp {
            font-size: 0.875rem;
            background-color: var(--border);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            color: var(--text-muted);
            font-weight: 500;
        }

        /* Metric Grid */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .metric-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background-color: var(--primary);
        }

        .metric-card.success::before { background-color: var(--success); }
        .metric-card.danger::before { background-color: var(--danger); }
        .metric-card.warning::before { background-color: var(--warning); }

        .metric-card h3 {
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .metric-card .subtext {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        /* Section Layout */
        .section-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
            margin-bottom: 2.5rem;
        }

        .section-card h2 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--text-main);
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Table Design */
        .table-wrapper {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.95rem;
        }

        th {
            background-color: #f1f5f9;
            color: #475569;
            padding: 0.75rem 1rem;
            font-weight: 600;
            border-bottom: 2px solid var(--border);
        }

        td {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            color: #334155;
            vertical-align: middle;
        }

        tr:hover td {
            background-color: #f8fafc;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.375rem;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-type { background-color: var(--primary-light); color: var(--primary); }
        .badge-imputed { background-color: #fef3c7; color: #d97706; }
        .badge-clean { background-color: #d1fae5; color: #059669; }

        /* Chart grids */
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }

        .chart-container {
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            text-align: center;
            background-color: #fafafa;
        }

        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 0.5rem;
        }

        /* Fuzzy Clustering List */
        .fuzzy-list {
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }

        .fuzzy-item {
            background-color: #f8fafc;
            border: 1px dashed var(--border);
            padding: 1rem;
            border-radius: 0.5rem;
        }

        .fuzzy-item strong {
            color: var(--danger);
        }
        
        .fuzzy-item span {
            color: var(--success);
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <h1>AutoSanitizer Audit Report</h1>
                <p>Advanced statistical and machine-learning data cleaning execution logs</p>
            </div>
            <div class="timestamp">Generated: {{ timestamp }}</div>
        </header>

        <!-- Summary Widgets -->
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Initial Shape</h3>
                <div class="value">{{ shape_before[0] }} x {{ shape_before[1] }}</div>
                <div class="subtext">Rows x Columns initially</div>
            </div>
            <div class="metric-card success">
                <h3>Final Shape</h3>
                <div class="value">{{ shape_after[0] }} x {{ shape_after[1] }}</div>
                <div class="subtext">Rows x Columns after cleaning</div>
            </div>
            <div class="metric-card danger">
                <h3>Total Missing Imputed</h3>
                <div class="value">{{ total_imputed }}</div>
                <div class="subtext">NaNs resolved using ML imputers</div>
            </div>
            <div class="metric-card warning">
                <h3>Outliers Addressed</h3>
                <div class="value">{{ total_outliers }}</div>
                <div class="subtext">Multivariate & Univariate points</div>
            </div>
        </div>

        <!-- Schema and Cleaning Decisions -->
        <div class="section-card">
            <h2>Column Schema & Operation Summary</h2>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Column Name</th>
                            <th>Inferred Semantic Type</th>
                            <th>Initial NaNs</th>
                            <th>Final NaNs</th>
                            <th>Cleaning Operations Done</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for col, details in column_details.items() %}
                        <tr>
                            <td><strong>{{ col }}</strong></td>
                            <td><span class="badge badge-type">{{ details.type }}</span></td>
                            <td>{{ details.initial_nans }}</td>
                            <td>{{ details.final_nans }}</td>
                            <td>
                                {% if details.actions %}
                                    <ul style="list-style-type: none;">
                                    {% for action in details.actions %}
                                        <li style="margin-bottom: 0.25rem;">📝 {{ action }}</li>
                                    {% endfor %}
                                    </ul>
                                {% else %}
                                    <span class="badge badge-clean">Verified Clean</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Text Resolution Map -->
        {% if fuzzy_mappings %}
        <div class="section-card">
            <h2>Text Standardization & Typo Resolution</h2>
            <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
                The following slightly misspelled labels were clustered via TF-IDF + Cosine Similarity and mapped to their most frequent spelling (centroid):
            </p>
            <ul class="fuzzy-list">
                {% for target_col, mapping in fuzzy_mappings.items() %}
                    {% for orig, target in mapping.items() %}
                    <li class="fuzzy-item">
                        <code>[{{ target_col }}]</code>: 
                        <strong>"{{ orig }}"</strong> &rarr; <span>"{{ target }}"</span>
                    </li>
                    {% endfor %}
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <!-- Missing Values Visual Map -->
        <div class="section-card">
            <h2>Missing Values Distribution</h2>
            <div style="display: flex; justify-content: center; margin-top: 1rem;">
                <img src="data:image/png;base64,{{ missingness_plot }}" style="max-width: 100%; height: auto;" alt="Missingness Chart">
            </div>
        </div>

        <!-- Distribution plots before vs after -->
        {% if distribution_plots %}
        <div class="section-card">
            <h2>Numerical Feature Distributions (Before vs After)</h2>
            <div class="chart-grid">
                {% for col, plot_base64 in distribution_plots.items() %}
                <div class="chart-container">
                    <img src="data:image/png;base64,{{ plot_base64 }}" alt="Distribution of {{ col }}">
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

def generate_report(df_before: pd.DataFrame, df_after: pd.DataFrame, 
                    inferred_types: dict, missing_log_num: dict, missing_log_cat: dict, 
                    outlier_log: dict, fuzzy_mappings: dict, output_filepath: str):
    """
    Assembles the cleaning logs, computes graphs, and writes a beautiful HTML report.
    """
    import datetime
    
    # 1. Calculate shapes and metrics
    shape_before = df_before.shape
    shape_after = df_after.shape
    
    total_imputed = sum(missing_log_num.values()) + sum(missing_log_cat.values())
    total_outliers = sum([count for key, count in outlier_log.items()])
    
    # 2. Build details per column
    column_details = {}
    for col in df_before.columns:
        initial_nans = int(df_before[col].isna().sum())
        final_nans = int(df_after[col].isna().sum())
        
        actions = []
        
        # Check type
        col_type = inferred_types.get(col, 'unknown')
        
        # Check if outliers were detected in this column
        univariate_key = f"{col} (univariate)"
        if univariate_key in outlier_log:
            actions.append(f"Identified {outlier_log[univariate_key]} univariate outliers using Robust Z-score.")
            
        # Check if multivariate outlier was detected (it affects numeric columns)
        if "Multivariate (Isolation Forest)" in outlier_log and col_type == 'numeric':
            actions.append("Addressed in multivariate Isolation Forest outlier sweep.")
            
        # Check if missing values were imputed
        if col in missing_log_num:
            actions.append(f"Imputed {missing_log_num[col]} missing numerical cells using iterative ML imputation.")
        elif col in missing_log_cat:
            # Check if it was imputed as Mode or Unknown
            imputed_val_ratio = df_before[col].isna().sum() / len(df_before)
            if imputed_val_ratio >= 0.4:
                actions.append(f"Classified {missing_log_cat[col]} heavily missing rows to 'Unknown'.")
            else:
                actions.append(f"Imputed {missing_log_cat[col]} rows using Random Forest classifier/Mode.")
                
        # Check if fuzzy string correction was applied
        if col in fuzzy_mappings and fuzzy_mappings[col]:
            actions.append(f"Standardized {len(fuzzy_mappings[col])} variant text entries via TF-IDF character clustering.")
            
        # If type changed in representation
        if df_before[col].dtype != df_after[col].dtype:
            actions.append(f"Cast representation from {df_before[col].dtype} to {df_after[col].dtype}.")
            
        column_details[col] = {
            'type': col_type,
            'initial_nans': initial_nans,
            'final_nans': final_nans,
            'actions': actions
        }
        
    # 3. Generate missingness chart
    missingness_plot = create_missingness_chart(df_before, df_after)
    
    # 4. Generate distribution charts for key numerical columns
    distribution_plots = {}
    numeric_cols = [c for c, t in inferred_types.items() if t == 'numeric' and c in df_before.columns]
    for col in numeric_cols[:4]: # Limit to first 4 numerical columns to avoid bloated HTML files
        # Check if there were either outliers or missing values to show difference
        if df_before[col].isna().sum() > 0 or col in outlier_log or f"{col} (univariate)" in outlier_log:
            distribution_plots[col] = create_distribution_chart(df_before, df_after, col)
            
    # 5. Compile HTML
    template = Template(HTML_TEMPLATE)
    html_content = template.render(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        shape_before=shape_before,
        shape_after=shape_after,
        total_imputed=total_imputed,
        total_outliers=total_outliers,
        column_details=column_details,
        fuzzy_mappings={k: v for k, v in fuzzy_mappings.items() if v}, # Only non-empty
        missingness_plot=missingness_plot,
        distribution_plots=distribution_plots
    )
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"HTML Audit Report generated successfully at: {output_filepath}")
