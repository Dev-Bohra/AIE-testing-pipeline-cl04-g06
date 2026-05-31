
# Bridge Structural Health Monitoring — Monitoring Script (for Data Quality, Drift & Monitoring Alerts) 
from pathlib import Path
import json
import logging
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset.csv"
CURRENT_DATA_PATH = ROOT / "data" / "processed" / "bridge_processed.csv"

METRICS_PATH = ROOT / "artifacts" / "metrics" / "monitoring_metrics.json"

MONITORING_DIR = ROOT / "monitoring"
LOGS_DIR = MONITORING_DIR / "logs"
REPORTS_DIR = MONITORING_DIR / "reports"
ALERTS_DIR = MONITORING_DIR / "alerts"

MONITORING_LOG_PATH = LOGS_DIR / "monitoring.log"
DRIFT_LOG_PATH = LOGS_DIR / "drift_detection.log"

DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
MONITORING_DASHBOARD_PATH = REPORTS_DIR / "monitoring_dashboard.html"
ALERTS_PATH = ALERTS_DIR / "monitoring_alerts.txt"

TARGET = "Structural_Health_Index_SHI"

DRIFT_P_VALUE_THRESHOLD = 0.05


LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=MONITORING_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


drift_logger = logging.getLogger("drift_detection")
drift_logger.setLevel(logging.INFO)

drift_handler = logging.FileHandler(DRIFT_LOG_PATH)
drift_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

if not drift_logger.handlers:
    drift_logger.addHandler(drift_handler)


def log(message):
    print(message)
    logging.info(message)


def load_data():
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_DATA_PATH}")

    if not CURRENT_DATA_PATH.exists():
        raise FileNotFoundError(f"Current processed data not found: {CURRENT_DATA_PATH}")

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    current_df = pd.read_csv(CURRENT_DATA_PATH)

    return reference_df, current_df


def data_quality_check(df):
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    status = "PASS"

    if missing_values > 0 or duplicate_rows > 0:
        status = "WARNING"

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "status": status,
    }


def detect_drift(reference_df, current_df):
    drift_results = {}

    reference_numeric = reference_df.select_dtypes(include="number")
    current_numeric = current_df.select_dtypes(include="number")

    common_cols = [
        col for col in reference_numeric.columns
        if col in current_numeric.columns and col != TARGET
    ]

    for col in common_cols:
        ref_values = reference_numeric[col].dropna()
        current_values = current_numeric[col].dropna()

        if len(ref_values) < 30 or len(current_values) < 30:
            continue

        statistic, p_value = ks_2samp(ref_values, current_values)

        drift_detected = p_value < DRIFT_P_VALUE_THRESHOLD

        drift_results[col] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": bool(drift_detected),
        }

        drift_logger.info(
            f"{col}: KS={statistic:.4f}, p={p_value:.6f}, drift={drift_detected}"
        )

    return drift_results


def generate_alerts(data_quality, drift_results):
    alerts = []

    if data_quality["status"] == "WARNING":
        alerts.append(
            "Data quality warning: missing values or duplicate rows detected."
        )

    drifted_features = [
        feature for feature, result in drift_results.items()
        if result["drift_detected"]
    ]

    if drifted_features:
        alerts.append(
            f"Data drift detected in {len(drifted_features)} feature(s): "
            + ", ".join(drifted_features)
        )
    else:
        alerts.append("No significant feature drift detected.")

    return alerts


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def save_alerts(alerts):
    with open(ALERTS_PATH, "w") as file:
        for alert in alerts:
            file.write(alert + "\n")




def save_monitoring_dashboard(report, path):
    data_quality = report.get("data_quality", {})
    drift_results = report.get("drift_results", {})
    alerts = report.get("alerts", [])

    drifted_features = [
        feature for feature, result in drift_results.items()
        if result.get("drift_detected") is True
    ]

    total_features = len(drift_results)
    drift_count = len(drifted_features)
    drift_detected = drift_count > 0

    status_label = "Drift Detected" if drift_detected else "Stable"
    status_class = "danger" if drift_detected else "success"

    rows = data_quality.get("rows", "N/A")
    columns = data_quality.get("columns", "N/A")
    missing_values = data_quality.get("missing_values", "N/A")
    duplicate_rows = data_quality.get("duplicate_rows", "N/A")
    quality_status = data_quality.get("status", "N/A")

    drift_items = "".join(
        f"<li>{feature} <span class='muted'>(p={drift_results[feature].get('p_value')}, KS={drift_results[feature].get('ks_statistic')})</span></li>"
        for feature in drifted_features
    )

    if not drift_items:
        drift_items = "<li>No major drifted features detected.</li>"

    alert_items = "".join(f"<li>{alert}</li>" for alert in alerts)
    if not alert_items:
        alert_items = "<li>No active alerts.</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bridge Health Monitoring Dashboard</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      background: #f4f6f8;
      color: #1f2937;
    }}
    header {{
      background: #111827;
      color: white;
      padding: 24px 40px;
    }}
    header h1 {{
      margin: 0;
      font-size: 28px;
    }}
    header p {{
      margin: 8px 0 0;
      color: #d1d5db;
    }}
    .container {{
      padding: 30px 40px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 20px;
      margin-bottom: 28px;
    }}
    .card {{
      background: white;
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 16px;
      color: #6b7280;
    }}
    .value {{
      font-size: 30px;
      font-weight: bold;
      margin: 0;
    }}
    .success {{
      color: #059669;
    }}
    .danger {{
      color: #dc2626;
    }}
    .section {{
      background: white;
      border-radius: 14px;
      padding: 24px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.08);
      margin-bottom: 24px;
    }}
    ul {{
      columns: 2;
      line-height: 1.8;
    }}
    .muted {{
      color: #6b7280;
      font-size: 13px;
    }}
    .footer {{
      color: #6b7280;
      font-size: 13px;
      margin-top: 20px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Bridge Structural Health Monitoring Dashboard</h1>
    <p>Generated automatically from the DVC monitoring stage.</p>
  </header>

  <div class="container">
    <div class="cards">
      <div class="card">
        <h2>Monitoring Status</h2>
        <p class="value {status_class}">{status_label}</p>
      </div>
      <div class="card">
        <h2>Features Checked</h2>
        <p class="value">{total_features}</p>
      </div>
      <div class="card">
        <h2>Drifted Features</h2>
        <p class="value danger">{drift_count}</p>
      </div>
      <div class="card">
        <h2>Data Quality</h2>
        <p class="value {status_class if quality_status != 'PASS' else 'success'}">{quality_status}</p>
      </div>
    </div>

    <div class="cards">
      <div class="card">
        <h2>Rows</h2>
        <p class="value">{rows}</p>
      </div>
      <div class="card">
        <h2>Columns</h2>
        <p class="value">{columns}</p>
      </div>
      <div class="card">
        <h2>Missing Values</h2>
        <p class="value">{missing_values}</p>
      </div>
      <div class="card">
        <h2>Duplicate Rows</h2>
        <p class="value">{duplicate_rows}</p>
      </div>
    </div>

    <div class="section">
      <h2>Detected Drift Features</h2>
      <ul>
        {drift_items}
      </ul>
    </div>

    <div class="section">
      <h2>Monitoring Alerts</h2>
      <ul>
        {alert_items}
      </ul>
    </div>

    <div class="section">
      <h2>Pipeline Summary</h2>
      <p>This dashboard summarises the monitoring output for the AI-based bridge structural health monitoring pipeline.</p>
      <p>The pipeline includes preprocessing, model training, evaluation, drift monitoring, and artefact tracking through DVC.</p>
      <p class="footer">Dashboard file: monitoring/reports/monitoring_dashboard.html</p>
    </div>
  </div>
</body>
</html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main():
    log("=" * 60)
    log("BRIDGE MONITORING")
    log("=" * 60)

    reference_df, current_df = load_data()

    data_quality = data_quality_check(current_df)
    drift_results = detect_drift(reference_df, current_df)
    alerts = generate_alerts(data_quality, drift_results)

    monitoring_report = {
        "monitoring_type": "data_quality_and_drift_detection",
        "reference_dataset": str(REFERENCE_DATA_PATH),
        "current_dataset": str(CURRENT_DATA_PATH),
        "drift_test": "Kolmogorov-Smirnov two-sample test",
        "drift_p_value_threshold": DRIFT_P_VALUE_THRESHOLD,
        "data_quality": data_quality,
        "drift_results": drift_results,
        "alerts": alerts,
    }

    save_json(monitoring_report, DRIFT_REPORT_PATH)
    save_json(monitoring_report, METRICS_PATH)
    save_monitoring_dashboard(monitoring_report, MONITORING_DASHBOARD_PATH)
    save_alerts(alerts)

    log(f"Monitoring metrics saved to: {METRICS_PATH}")
    log(f"Drift report saved to: {DRIFT_REPORT_PATH}")
    log(f"Monitoring dashboard saved to: {MONITORING_DASHBOARD_PATH}")
    log(f"Alerts saved to: {ALERTS_PATH}")

    for alert in alerts:
        log(f"ALERT: {alert}")

    log("Monitoring complete.")


if __name__ == "__main__":
    main()