from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# File paths
# -----------------------------
RAW_PATH = Path("data/raw/Dataset_ATS_v2.csv")
PROCESSED_DIR = Path("data/processed")

CLEANED_PATH = PROCESSED_DIR / "churn_cleaned.csv"
PROCESSED_PATH = PROCESSED_DIR / "churn_processed.csv"
TRAIN_PATH = PROCESSED_DIR / "churn_train.csv"
TEST_PATH = PROCESSED_DIR / "churn_test.csv"
TRAIN_SCALED_PATH = PROCESSED_DIR / "churn_train_scaled.csv"
TEST_SCALED_PATH = PROCESSED_DIR / "churn_test_scaled.csv"


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    return pd.read_csv(path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )
    return df


def standardize_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces and lowercase text columns."""
    df = df.copy()
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print key dataset inspection results."""
    print("\n--- DATA INSPECTION ---")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    duplicate_count = df.duplicated().sum()
    print("\nDuplicate rows:", duplicate_count)

    if duplicate_count > 0:
        print(
            "Note: Duplicates were detected but retained because there is no unique customer ID "
            "to verify whether they are true duplicate records."
        )


def build_processed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables and prepare final processed dataset."""
    df = df.copy()

    if "churn" not in df.columns:
        raise ValueError("Target column 'churn' not found in dataset.")

    # Convert target variable
    df["churn"] = df["churn"].map({"no": 0, "yes": 1})

    if df["churn"].isnull().any():
        raise ValueError("Target column 'churn' contains unexpected values after mapping.")

    categorical_features = [
        "gender",
        "dependents",
        "phoneservice",
        "multiplelines",
        "internetservice",
        "contract",
    ]

    missing_cat_cols = [col for col in categorical_features if col not in df.columns]
    if missing_cat_cols:
        raise ValueError(f"Missing expected categorical columns: {missing_cat_cols}")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    final_df = X_encoded.copy()
    final_df["churn"] = y

    # Convert bool columns to int
    bool_cols = final_df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        final_df[bool_cols] = final_df[bool_cols].astype(int)

    return final_df


def split_dataset(final_df: pd.DataFrame):
    """Split processed dataset into train and test sets."""
    X = final_df.drop("churn", axis=1)
    y = final_df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale selected continuous numeric features using StandardScaler."""
    scale_cols = ["tenure", "monthlycharges"]

    missing_scale_cols = [col for col in scale_cols if col not in X_train.columns]
    if missing_scale_cols:
        raise ValueError(f"Missing columns for scaling: {missing_scale_cols}")

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    scaler = StandardScaler()
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train_scaled, X_test_scaled


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw dataset
    raw_df = load_raw_data(RAW_PATH)

    # 2. Clean and standardize
    cleaned_df = clean_column_names(raw_df)
    cleaned_df = standardize_text_values(cleaned_df)

    # 3. Inspect
    inspect_data(cleaned_df)

    # 4. Save cleaned dataset
    save_dataframe(cleaned_df, CLEANED_PATH)
    print(f"\nCleaned dataset saved to: {CLEANED_PATH}")

    # 5. Build processed dataset
    final_df = build_processed_dataset(cleaned_df)
    save_dataframe(final_df, PROCESSED_PATH)
    print(f"Processed dataset saved to: {PROCESSED_PATH}")
    print("Processed dataset shape:", final_df.shape)

    # 6. Split into train and test
    X_train, X_test, y_train, y_test = split_dataset(final_df)

    train_df = X_train.copy()
    train_df["churn"] = y_train

    test_df = X_test.copy()
    test_df["churn"] = y_test

    save_dataframe(train_df, TRAIN_PATH)
    save_dataframe(test_df, TEST_PATH)

    print(f"Train dataset saved to: {TRAIN_PATH}")
    print(f"Test dataset saved to: {TEST_PATH}")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nOverall churn distribution:")
    print(final_df["churn"].value_counts(normalize=True))

    print("\nTraining churn distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting churn distribution:")
    print(y_test.value_counts(normalize=True))

    # 7. Scale selected features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    train_scaled_df = X_train_scaled.copy()
    train_scaled_df["churn"] = y_train

    test_scaled_df = X_test_scaled.copy()
    test_scaled_df["churn"] = y_test

    save_dataframe(train_scaled_df, TRAIN_SCALED_PATH)
    save_dataframe(test_scaled_df, TEST_SCALED_PATH)

    print(f"\nScaled train dataset saved to: {TRAIN_SCALED_PATH}")
    print(f"Scaled test dataset saved to: {TEST_SCALED_PATH}")

    print("\nBefore scaling:")
    print(X_train[["tenure", "monthlycharges"]].describe())

    print("\nAfter scaling:")
    print(X_train_scaled[["tenure", "monthlycharges"]].describe())

    print("\n--- PIPELINE COMPLETE ---")
    print("All preprocessing, train-test split, and scaling tasks finished successfully.")


if __name__ == "__main__":
    main()