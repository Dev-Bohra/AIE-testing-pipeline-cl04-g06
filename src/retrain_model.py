"""
Bridge Structural Health Monitoring — Automated Retraining Script

Expected new data location:
data/new_data.csv
"""

from pathlib import Path
from datetime import datetime
import subprocess
import logging
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset.csv"
NEW_DATA_PATH = ROOT / "data" / "new_data.csv"
UPDATED_DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset_updated.csv"

PREPROCESSING_SCRIPT = ROOT / "src" / "data_preprocessing.py"
TRAINING_SCRIPT = ROOT / "src" / "model.py"

LOGS_DIR = ROOT / "monitoring" / "logs"
REPORTS_DIR = ROOT / "monitoring" / "reports"
METADATA_DIR = ROOT / "artifacts" / "metadata"

RETRAINING_LOG_PATH = LOGS_DIR / "retraining.log"
RETRAINING_REPORT_PATH = REPORTS_DIR / "retraining_report.txt"
LAST_RETRAIN_PATH = METADATA_DIR / "last_retrain.txt"


LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=RETRAINING_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def log(message):
    print(message)
    logging.info(message)


def check_required_files():
    required_files = [
        RAW_DATA_PATH,
        NEW_DATA_PATH,
        PREPROCESSING_SCRIPT,
        TRAINING_SCRIPT,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")

    log("All required files found.")


def merge_datasets():
    log("=" * 60)
    log("MERGING ORIGINAL DATASET WITH NEW DATA")
    log("=" * 60)

    raw_df = pd.read_csv(RAW_DATA_PATH)
    new_df = pd.read_csv(NEW_DATA_PATH)

    original_rows = len(raw_df)
    new_rows = len(new_df)

    combined_df = pd.concat(
        [raw_df, new_df],
        ignore_index=True
    )

    combined_df = combined_df.drop_duplicates()

    final_rows = len(combined_df)
    duplicates_removed = original_rows + new_rows - final_rows

    UPDATED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(UPDATED_DATA_PATH, index=False)

    log(f"Original rows: {original_rows}")
    log(f"New rows added: {new_rows}")
    log(f"Final rows after merge: {final_rows}")
    log(f"Duplicates removed: {duplicates_removed}")
    log(f"Updated dataset saved to: {UPDATED_DATA_PATH}")

    return {
        "original_rows": original_rows,
        "new_rows_added": new_rows,
        "final_rows_after_merge": final_rows,
        "duplicates_removed": duplicates_removed,
    }


def run_script(script_path, extra_args=None):
    command = ["python", str(script_path)]

    if extra_args:
        command.extend(extra_args)

    log("=" * 60)
    log(f"RUNNING: {' '.join(command)}")
    log("=" * 60)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True
    )

    log(f"Completed: {script_path.name}")


def update_last_retrain():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LAST_RETRAIN_PATH, "w") as file:
        file.write(timestamp)

    log(f"Last retrain timestamp saved to: {LAST_RETRAIN_PATH}")


def save_retraining_report(merge_report):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_text = f"""
Bridge SHI Retraining Report
Generated: {timestamp}

New data source:
- data/new_data.csv

Original rows: {merge_report['original_rows']}
New rows added: {merge_report['new_rows_added']}
Final rows after merge: {merge_report['final_rows_after_merge']}
Duplicates removed: {merge_report['duplicates_removed']}

Pipeline steps completed:
1. Checked for new data
2. Merged original dataset with data/new_data.csv
3. Reran preprocessing on merged dataset
4. Retrained Random Forest and MLP models
5. Reselected best model using lowest RMSE
6. Updated model artefacts, metrics, metadata, reports, and logs
"""

    with open(RETRAINING_REPORT_PATH, "w") as file:
        file.write(report_text)

    log(f"Retraining report saved to: {RETRAINING_REPORT_PATH}")


def main():
    log("=" * 60)
    log("AUTOMATED MODEL RETRAINING")
    log("=" * 60)

    try:
        check_required_files()
        merge_report = merge_datasets()

        run_script(PREPROCESSING_SCRIPT, extra_args=[str(UPDATED_DATA_PATH)])
        run_script(TRAINING_SCRIPT)

        update_last_retrain()
        save_retraining_report(merge_report)

        log("Retraining complete.")

    except Exception as error:
        logging.exception("Retraining failed.")
        print(f"Retraining failed: {error}")
        raise


if __name__ == "__main__":
    main()