## Data Preprocessing and Feature Engineering

The raw dataset was loaded from `data/raw/Dataset_ATS_v2.csv` and preserved unchanged.

Preprocessing steps:
- inspected dataset structure, missing values, duplicates, and data types
- standardized column names to lowercase with underscores
- standardized categorical text values to lowercase
- retained duplicate rows because no customer ID was available to verify whether they were true duplicates
- identified numeric and categorical features

Feature engineering steps:
- separated input features and target variable (`churn`)
- converted target variable from `yes/no` to `1/0`
- applied one-hot encoding to categorical features using `pd.get_dummies(..., drop_first=True)`
- converted boolean dummy columns to integer
- saved processed dataset to `data/processed/churn_processed.csv`