"""
Bridge Structural Health Monitoring — Data Preprocessing Pipeline

This script:
1. Loads raw bridge dataset
2. Removes duplicates
3. Drops leakage / unusable columns
4. Handles missing values
5. Handles outliers
6. Encodes categorical variables
7. Scales numeric features
8. Saves cleaned and processed datasets
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# ─────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset.csv"
NEW_DATA_PATH = ROOT / "data" / "new_data.csv"

CLEANED_OUTPUT_PATH = ROOT / "data" / "cleaned" / "bridge_cleaned.csv"
PROCESSED_OUTPUT_PATH = ROOT / "data" / "processed" / "bridge_processed.csv"

SCALER_OUTPUT_PATH = ROOT / "artifacts" / "preprocessing" / "scaler.pkl"


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TARGET = "Structural_Health_Index_SHI"

COLS_DROP_MISSING = [
    "Vibration_Anomaly_Location"
]

COLS_DROP_LEAKAGE = [
    # Future SHI / target-derived columns
    "SHI_Predicted_24h_Ahead",
    "SHI_Predicted_7d_Ahead",
    "SHI_Predicted_30d_Ahead",
    "Bridge_Mood_Meter",
    # Output / decision columns that would leak target information
    "Probability_of_Failure_PoF",
    "Maintenance_Alert",
    "Estimated_Repair_Cost_USD_incremental",
    "Carbon_Footprint_tCO2e_incremental",
    "Energy_Harvesting_Potential_W",
    # Highly derived / redundant simulated outputs
    "Simulated_Water_Flow_m3s",
    "Axle_Counts_pmin",
    "Air_Quality_Index_AQI",
]

Z_THRESHOLD = 3.0
IQR_FACTOR = 1.5


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_dataset(path: Path) -> pd.DataFrame:
    print("=" * 60)
    print("LOAD DATASET")
    print("=" * 60)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        print(f"Date range: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")

    print(f"Loaded dataset: {path}")
    print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print()
    return df


# ─────────────────────────────────────────────
# REMOVE DUPLICATES
# ─────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("REMOVE DUPLICATES")
    print("=" * 60)

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"Duplicates removed: {before - after}")
    print(f"Shape after duplicate removal: {df.shape}")
    print()
    return df


# ─────────────────────────────────────────────
# DROP COLUMNS
# ─────────────────────────────────────────────

def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("DROP UNUSABLE / LEAKAGE COLUMNS")
    print("=" * 60)

    columns_to_drop = COLS_DROP_MISSING + COLS_DROP_LEAKAGE
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    df = df.drop(columns=existing_columns)

    print(f"Columns dropped: {len(existing_columns)}")
    for col in existing_columns:
        print(f"- {col}")

    print(f"Shape after column drop: {df.shape}")
    print()
    return df


# ─────────────────────────────────────────────
# HANDLE MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("HANDLE MISSING VALUES")
    print("=" * 60)

    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != "Timestamp"]

    if numeric_cols:
        df[numeric_cols] = (
            df[numeric_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    missing_after = df.isnull().sum().sum()
    print(f"Missing values after: {missing_after}")
    print()
    return df


# ─────────────────────────────────────────────
# HANDLE OUTLIERS
# ─────────────────────────────────────────────

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("HANDLE OUTLIERS")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    skip_cols = [
        TARGET,
        "Flood_Event_Flag",
        "Maintenance_Alert",
        "Anomaly_Detection_Score",
        "Localized_Strain_Hotspot",
        "High_Winds_Storms",
        "Abnormal_Traffic_Load_Surges",
        "Landslide_Ground_Movement",
        "Impact_Events_g",
        "Seismic_Activity_ms2",
    ]

    treat_cols = [col for col in numeric_cols if col not in skip_cols]

    total_fixed = 0

    for col in treat_cols:
        if df[col].dropna().shape[0] < 30:
            continue

        z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
        z_mask = z_scores > Z_THRESHOLD

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        iqr_mask = (df[col] < q1 - IQR_FACTOR * iqr) | (
            df[col] > q3 + IQR_FACTOR * iqr
        )

        combined_mask = z_mask & iqr_mask
        flagged_count = int(combined_mask.sum())

        if flagged_count > 0:
            df.loc[combined_mask, col] = np.nan
            total_fixed += flagged_count

    if treat_cols:
        df[treat_cols] = (
            df[treat_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    print(f"Total outlier values corrected: {total_fixed}")
    print()
    return df


# ─────────────────────────────────────────────
# ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("ENCODE CATEGORICAL VARIABLES")
    print("=" * 60)

    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != "Timestamp"]

    if not categorical_cols:
        print("No categorical columns to encode.")
        print()
        return df

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col + "_enc"] = encoder.fit_transform(df[col].astype(str))
        df = df.drop(columns=[col])
        print(f"Encoded: {col} -> {col}_enc")

    print()
    return df


# ─────────────────────────────────────────────
# SCALE FEATURES
# ─────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 60)
    print("SCALE FEATURES")
    print("=" * 60)

    df_unscaled = df.copy()
    df_scaled = df.copy()

    no_scale_cols = [
        "Timestamp",
        TARGET,
        "Flood_Event_Flag",
        "Maintenance_Alert",
        "High_Winds_Storms",
        "Abnormal_Traffic_Load_Surges",
        "Landslide_Ground_Movement",
    ]

    no_scale_cols = [col for col in no_scale_cols if col in df_scaled.columns]

    scale_cols = [
        col for col in df_scaled.select_dtypes(include="number").columns
        if col not in no_scale_cols
    ]

    scaler = StandardScaler()

    if scale_cols:
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])

        SCALER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_OUTPUT_PATH)

    print(f"Scaled columns: {len(scale_cols)}")
    print(f"Scaler saved to: {SCALER_OUTPUT_PATH}")
    print()
    return df_scaled, df_unscaled


# ─────────────────────────────────────────────
# VALIDATE OUTPUT
# ─────────────────────────────────────────────

def validate_output(df_clean: pd.DataFrame, raw_shape: tuple) -> None:
    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print(f"Original shape: {raw_shape}")
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    print(f"Duplicate rows: {df_clean.duplicated().sum()}")

    if TARGET in df_clean.columns:
        print("\nTarget summary:")
        print(df_clean[TARGET].describe())

    print()


# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_outputs(df_scaled: pd.DataFrame, df_unscaled: pd.DataFrame) -> None:
    print("=" * 60)
    print("SAVE OUTPUTS")
    print("=" * 60)

    CLEANED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_unscaled.to_csv(CLEANED_OUTPUT_PATH, index=False)
    df_scaled.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    print(f"Cleaned unscaled dataset saved to: {CLEANED_OUTPUT_PATH}")
    print(f"Processed scaled dataset saved to: {PROCESSED_OUTPUT_PATH}")
    print()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(input_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("BRIDGE DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")

    df = load_dataset(input_path)
    raw_shape = df.shape

    df = remove_duplicates(df)
    df = drop_unusable_columns(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)

    df_scaled, df_unscaled = scale_features(df)

    validate_output(df_unscaled, raw_shape)
    save_outputs(df_scaled, df_unscaled)

    print("Preprocessing pipeline complete.")
    return df_scaled, df_unscaled


if __name__ == "__main__":
    run_pipeline()