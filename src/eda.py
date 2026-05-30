# Bridge Structural Health Monitoring — Exploratory Data Analysis (EDA)
# This script is similar to portfolio task 1 which helps us understand data better

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset.csv"
FIGURES_DIR = ROOT / "reports" / "eda_figures"

TARGET = "Structural_Health_Index_SHI"


# Task 1 - Dataset Loading & Initial Exploration
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!\n")

print("Dataset Shape (rows, columns):", df.shape, "\n")

print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Data types before conversion:")
print(df.dtypes, "\n")

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    print("Data types after Timestamp conversion:")
    print(df.dtypes, "\n")

print("Statistical Summary:")
print(df.describe(), "\n")

print("Missing values in each column:")
print(df.isnull().sum(), "\n")

print("Dataset Info:")
print(df.info())


# Task 2 - Data Quality Assessment
missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing_count,
    "Percentage (%)": missing_percent
})

print("\nMissing values table:")
print(missing_table, "\n")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "missing_values_heatmap.png")
plt.close()

duplicate_count = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_count, "\n")

if duplicate_count > 0:
    print("Examples of duplicate rows:")
    print(df[df.duplicated()].head(), "\n")
else:
    print("No duplicate rows found.\n")

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"boxplot_{col}.png")
    plt.close()


# Task 3 - Target Variable Preparation / Analysis
print("\nTarget Variable:", TARGET)
print("Target variable summary:")
print(df[TARGET].describe(), "\n")

df["SHI_Class"] = pd.qcut(
    df[TARGET],
    q=3,
    labels=["Low", "Medium", "High"]
)

print("SHI class distribution:")
print(df["SHI_Class"].value_counts(), "\n")

plt.figure(figsize=(7, 5))
df["SHI_Class"].value_counts().plot(kind="bar")
plt.title("SHI Class Distribution")
plt.xlabel("SHI Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shi_class_distribution.png")
plt.close()


# Task 4 - Univariate Analysis
selected_features = [
    "Strain_microstrain",
    "Deflection_mm",
    "Vibration_ms2",
    "Tilt_deg",
    "Temperature_C",
    "Humidity_percent",
    "Wind_Speed_ms",
    "Vehicle_Load_tons",
    TARGET
]

selected_features = [col for col in selected_features if col in df.columns]

print("Selected predictor variables for univariate analysis:")
print(selected_features, "\n")

for col in selected_features:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"histogram_{col}.png")
    plt.close()

print("Summary statistics for selected predictors:")
print(df[selected_features].describe(), "\n")


# Task 5 - Multivariate Analysis
numerical_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(14, 10))
sns.heatmap(numerical_df.corr(), cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
plt.close()

correlation_with_target = numerical_df.corr()[TARGET].sort_values(ascending=False)

print("Correlation with target:")
print(correlation_with_target, "\n")

exclude = [
    TARGET,
    "SHI_Predicted_24h_Ahead",
    "SHI_Predicted_7d_Ahead",
    "SHI_Predicted_30d_Ahead",
    "Bridge_Mood_Meter"
]

exclude = [col for col in exclude if col in correlation_with_target.index]

top_5 = correlation_with_target.drop(exclude, errors="ignore").head(5)

print("Top 5 features correlated with SHI:")
print(top_5, "\n")

corr_matrix = numerical_df.corr()
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_value = corr_matrix.iloc[i, j]

        if abs(corr_value) > 0.8:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_value
            ))

print("Highly correlated feature pairs (|corr| > 0.8):")
for pair in high_corr_pairs:
    print(pair)

key_features = list(top_5.index) + [TARGET]
key_features = [col for col in key_features if col in df.columns]

if len(key_features) > 1:
    sample_df = df[key_features].dropna().sample(
        min(500, len(df)),
        random_state=42
    )

    sns.pairplot(sample_df)
    plt.savefig(FIGURES_DIR / "pairplot_key_features.png")
    plt.close()

print("\nEDA complete.")
print(f"Figures saved in: {FIGURES_DIR}")

'''
# Final EDA insight:
# The main indicators affecting bridge health are structural/load-related variables.
# SHI has strong negative relationships with Strain_microstrain (-0.93), Tilt_deg (-0.85),
# Deflection_mm (-0.77), Cable_Member_Tension_kN (-0.76), Traffic_Volume_vph (-0.75),
# Vehicle_Load_tons (-0.75), and Probability_of_Failure_PoF (-0.75). This makes engineering
# sense because higher strain, tilt, deflection, traffic load, and failure risk usually indicate
# lower bridge health.

# The modelling approach in model.py is appropriate because SHI is a continuous value, so
# regression is the correct task. Random Forest is suitable because the dataset has nonlinear
# sensor relationships and highly correlated features. The MLP neural network is also useful as
# a lightweight comparison model after scaling. Comparing both models and selecting the one
# with the lowest RMSE is a valid approach for reliable model selection.
#
# Leakage columns such as future SHI predictions and output-derived fields will be removed
# before training because they would make the model look unrealistically accurate.
'''