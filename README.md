# Customer Churn Data Preprocessing and Feature Engineering

## Overview

This project prepares a customer churn dataset for downstream analysis and machine learning. The workflow focuses on data preprocessing, feature engineering, train-test splitting, and feature scaling.

The raw dataset is loaded from `data/raw/Dataset_ATS_v2.csv` and preserved unchanged.

---

## Project Structure

```text
telecom-churn-data-engineering/
├── data/
│   ├── raw/
│   │   └── Dataset_ATS_v2.csv
│   └── processed/
├── notebooks/
├── src/
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── full_preprocessing_pipeline.py
```

---

## Requirements

### Core dependencies

These packages are required to run the preprocessing pipeline:

```txt
pandas==3.0.2
numpy==2.4.4
scikit-learn==1.8.0
matplotlib==3.10.8
openpyxl==3.1.5
pyarrow==23.0.1
```

### Development dependencies

These packages are used for Jupyter Notebook and exploratory work:

```txt
-r requirements.txt
jupyter==1.1.1
ipykernel==7.2.0
seaborn==0.13.2
```

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

### 4. Install development dependencies

```bash
pip install -r requirements-dev.txt
```

---

## How to Run

Run the full preprocessing pipeline from the project root:

```bash
python final_preprocessing.py
```

---

## Data Preprocessing and Feature Engineering

The raw dataset was loaded from `data/raw/Dataset_ATS_v2.csv` and preserved unchanged.

### Preprocessing steps

* inspected dataset structure, missing values, duplicates, and data types
* standardized column names to lowercase with underscores
* standardized categorical text values to lowercase
* retained duplicate rows because no customer ID was available to verify whether they were true duplicates
* identified numeric and categorical features

### Feature engineering steps

* separated input features and target variable (`churn`)
* converted target variable from `yes/no` to `1/0`
* applied one-hot encoding to categorical features using `pd.get_dummies(..., drop_first=True)`
* converted boolean dummy columns to integer
* saved processed dataset to `data/processed/churn_processed.csv`

### Additional preparation steps

* split the processed dataset into training and testing sets using an 80:20 stratified split
* applied feature scaling to `tenure` and `monthlycharges` using `StandardScaler`
* fitted the scaler on the training set only and applied it to the test set
* saved train, test, and scaled output files to the `data/processed/` directory

### Output files generated

* `data/processed/churn_cleaned.csv`
* `data/processed/churn_processed.csv`
* `data/processed/churn_train.csv`
* `data/processed/churn_test.csv`
* `data/processed/churn_train_scaled.csv`
* `data/processed/churn_test_scaled.csv`

### Pipeline summary

This program prepares the customer churn dataset for downstream analysis and machine learning by cleaning the raw data, encoding categorical variables, splitting the dataset into training and testing sets, and standardizing selected continuous features.

---

## Notes

* No missing values were found in the dataset, so no missing-data imputation was required.
* Duplicate rows were detected but retained because the dataset does not contain a unique customer identifier.
* The term **standardisation** is more accurate than **normalisation** here because the workflow uses `StandardScaler`.
* The processed outputs are ready for downstream analysis, clustering, and machine learning tasks such as customer segmentation and churn prediction.
