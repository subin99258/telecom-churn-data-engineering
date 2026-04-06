import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Dataset_ATS_v2.csv")
CLEAN_PATH = Path("data/processed/churn_cleaned.csv")


def clean_data():
    df = pd.read_csv(RAW_PATH)

    # clean column names
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )

    # standardize text values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()

    # create processed folder if missing
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    # save cleaned data
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_PATH}")


if __name__ == "__main__":
    clean_data()