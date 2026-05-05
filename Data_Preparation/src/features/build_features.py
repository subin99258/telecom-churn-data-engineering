import pandas as pd
from pathlib import Path

CLEAN_PATH = Path("data/processed/churn_cleaned.csv")
FINAL_PATH = Path("data/processed/churn_processed.csv")


def build_features():
    df = pd.read_csv(CLEAN_PATH)

    X = df.drop("churn", axis=1)
    y = df["churn"].map({"no": 0, "yes": 1})

    categorical_features = [
        "gender",
        "dependents",
        "phoneservice",
        "multiplelines",
        "internetservice",
        "contract"
    ]

    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    final_df = X_encoded.copy()
    final_df["churn"] = y

    # convert boolean columns to int
    bool_cols = final_df.select_dtypes(include="bool").columns
    final_df[bool_cols] = final_df[bool_cols].astype(int)

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_PATH, index=False)

    print(f"Feature-engineered data saved to {FINAL_PATH}")
    print("Final shape:", final_df.shape)


if __name__ == "__main__":
    build_features()