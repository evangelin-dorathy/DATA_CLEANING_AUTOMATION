# INTERN ID:CITS5387
# DATA_CLEANING_AUTOMATION

## Overview

Data Cleaning Automation is a Python-based project that automates common data preprocessing tasks required before data analysis and machine learning. It intelligently identifies data types, handles missing values, standardizes inconsistent text, detects outliers, and generates a comprehensive HTML report summarizing the cleaning process.

The project is modular, making it easy to extend or integrate into other data processing workflows.

---

## Features

- Automatic column type inference
- Intelligent text standardization and typo correction
- Missing value imputation for numerical and categorical data
- Univariate and multivariate outlier detection
- Automated HTML report generation
- Synthetic dataset generation for testing the complete pipeline

---

## Project Structure

```
Data-Cleaning-Automation/
│
├── main.py
├── type_inference.py
├── text_cleaner.py
├── imputers.py
├── outlier_detector.py
├── report_generator.py
├── requirements.txt
└── README.md
```

---

## Module Description

### type_inference.py

Automatically identifies the type of each dataset column using statistical analysis and heuristic rules.

Supported types include:

- Numeric
- Categorical
- Datetime
- Boolean
- Text
- Identifier (ID)

---

### text_cleaner.py

Detects and standardizes inconsistent text values such as spelling mistakes and formatting differences using:

- Character N-gram TF-IDF
- Cosine Similarity

Example:

```
new york
New Yrok
NEW YORK

↓

New York
```

---

### outlier_detector.py

Detects abnormal observations using two approaches.

**Univariate Detection**

- Robust Z-Score
- Median Absolute Deviation (MAD)

**Multivariate Detection**

- Isolation Forest

---

### imputers.py

Handles missing values using machine learning techniques.

**Numerical Columns**

- MICE (Multivariate Imputation by Chained Equations)

**Categorical Columns**

- Random Forest Classifier

---

### report_generator.py

Creates an HTML report containing:

- Dataset overview
- Missing value summary
- Column type information
- Cleaning operations performed
- Distribution plots
- Base64 embedded visualizations

---

### main.py

Acts as the command-line interface for the project.

It automatically:

- Generates a synthetic dirty dataset
- Executes the entire cleaning pipeline
- Produces the cleaned dataset
- Generates the HTML report

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/Data-Cleaning-Automation.git
```

Move into the project directory

```bash
cd Data-Cleaning-Automation
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project using

```bash
python main.py
```

The pipeline will:

1. Generate a synthetic dataset
2. Detect column types
3. Clean inconsistent text
4. Impute missing values
5. Detect outliers
6. Generate an HTML report

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- HTML

---

## Future Improvements

- Interactive dashboard for report visualization
- Additional imputation algorithms
- Duplicate record detection
- Automated feature engineering
- Support for larger datasets
- Export cleaned datasets in multiple formats

---

## License

This project is intended for educational and research purposes.

---

## Author

Evangelin Dorathy
