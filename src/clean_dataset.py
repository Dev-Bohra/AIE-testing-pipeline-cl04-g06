import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset.csv"
CLEANED_DATA_PATH = ROOT / "data" / "cleaned" / "bridge_cleaned.csv"


def clean_dataset():
    df = pd.read_csv(RAW_DATA_PATH)

    print("Original shape:", df.shape)

    # Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Remove duplicate sensor records
    df = df.drop_duplicates()

    # Filling missing numeric values with median values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing categorical values with most common value
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove extreme outliers using IQR method
    rows_before_outliers = len(df)

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    rows_after_outliers = len(df)

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)

    print("Cleaned shape:", df.shape)
    print("Rows removed as outliers:", rows_before_outliers - rows_after_outliers)
    print("Saved cleaned dataset to:", CLEANED_DATA_PATH)


if __name__ == "__main__":
    clean_dataset()