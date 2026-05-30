# Bridge Structural Health Monitoring — Trend Analysis
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset.csv"
REPORTS_DIR = ROOT / "reports" / "trend_figures"

TIMESTAMP = "Timestamp"
TARGET = "Structural_Health_Index_SHI"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    df = df.sort_values(TIMESTAMP)

    return df


def plot_shi_over_time(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df[TIMESTAMP], df[TARGET])
    plt.title("SHI Trend Over Time")
    plt.xlabel("Time")
    plt.ylabel("SHI")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shi_trend_over_time.png")
    plt.close()


def plot_daily_average_shi(df):
    daily_df = (
        df.set_index(TIMESTAMP)[TARGET]
        .resample("D")
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        daily_df[TIMESTAMP],
        daily_df[TARGET],
        marker="o"
    )

    plt.title("Daily Average SHI")
    plt.xlabel("Date")
    plt.ylabel("Average SHI")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "daily_average_shi.png")
    plt.close()


def plot_strain_trend(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df[TIMESTAMP], df["Strain_microstrain"])
    plt.title("Strain Trend Over Time")
    plt.xlabel("Time")
    plt.ylabel("Strain (microstrain)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "strain_trend.png")
    plt.close()


def plot_traffic_trend(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df[TIMESTAMP], df["Traffic_Volume_vph"])
    plt.title("Traffic Volume Trend")
    plt.xlabel("Time")
    plt.ylabel("Traffic Volume (vph)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "traffic_volume_trend.png")
    plt.close()


def plot_shi_vs_strain(df):
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["Strain_microstrain"],
        df[TARGET],
        alpha=0.3
    )

    plt.title("SHI vs Strain")
    plt.xlabel("Strain (microstrain)")
    plt.ylabel("SHI")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shi_vs_strain.png")
    plt.close()


def plot_shi_vs_vehicle_load(df):
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["Vehicle_Load_tons"],
        df[TARGET],
        alpha=0.3
    )

    plt.title("SHI vs Vehicle Load")
    plt.xlabel("Vehicle Load (tons)")
    plt.ylabel("SHI")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shi_vs_vehicle_load.png")
    plt.close()


def main():
    print("=" * 60)
    print("BRIDGE TREND ANALYSIS")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    plot_shi_over_time(df)
    plot_daily_average_shi(df)
    plot_strain_trend(df)
    plot_traffic_trend(df)
    plot_shi_vs_strain(df)
    plot_shi_vs_vehicle_load(df)

    print("Trend analysis complete.")
    print(f"Figures saved to: {REPORTS_DIR}")
    
if __name__ == "__main__":
    main()