from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
PRED_PATH = ROOT / "predictions" / "test" / "ensemble_v1.csv"

OUTPUT_DIR = ROOT / "submissions"
OUTPUT_PATH = OUTPUT_DIR / "submission_ens_v1.csv"


def main():

    print("=" * 60)
    print("Create Kaggle Submission - ENS_V1")
    print("=" * 60)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    sample = pd.read_csv(SAMPLE_PATH)
    pred = pd.read_csv(PRED_PATH)

    print("\nSample submission:")
    print("Shape:", sample.shape)
    print("Columns:", sample.columns.tolist())

    print("\nPrediction:")
    print("Shape:", pred.shape)
    print("Columns:", pred.columns.tolist())

    # -----------------------------------------------------
    # Detect columns from sample_submission
    # -----------------------------------------------------

    if len(sample.columns) != 2:
        raise ValueError(
            f"Expected 2 columns in sample_submission, "
            f"but found {len(sample.columns)}"
        )

    id_col = sample.columns[0]
    target_col = sample.columns[1]

    print("\nDetected:")
    print("ID column    :", id_col)
    print("Target column:", target_col)

    # -----------------------------------------------------
    # Basic checks
    # -----------------------------------------------------

    if len(sample) != len(pred):
        raise ValueError(
            f"Row count mismatch: "
            f"sample={len(sample)}, prediction={len(pred)}"
        )

    if id_col not in pred.columns:
        raise ValueError(
            f"Prediction file does not contain ID column: {id_col}"
        )

    if "prediction" not in pred.columns:
        raise ValueError(
            "Prediction file does not contain 'prediction' column"
        )

    # ID 顺序必须完全一致
    if not sample[id_col].equals(pred[id_col]):
        raise ValueError(
            "ID order does not match sample_submission!"
        )

    values = pred["prediction"].to_numpy()

    if np.isnan(values).any():
        raise ValueError("Predictions contain NaN!")

    if np.isinf(values).any():
        raise ValueError("Predictions contain inf!")

    if (values < 0).any() or (values > 1).any():
        raise ValueError(
            "Predictions are outside [0, 1]!"
        )

    # -----------------------------------------------------
    # Create submission
    # -----------------------------------------------------

    submission = sample.copy()

    submission[target_col] = values

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    assert submission.shape == sample.shape
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission[id_col].equals(sample[id_col])
    assert submission[target_col].notna().all()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("VALIDATION PASSED")
    print("=" * 60)

    print("Rows       :", len(submission))
    print("Columns    :", submission.columns.tolist())
    print("NaN        :", submission.isna().sum().sum())
    print(
        "Pred range :",
        f"{submission[target_col].min():.6f}",
        "->",
        f"{submission[target_col].max():.6f}"
    )

    print("\nPreview:")
    print(submission.head())

    print("\nSubmission saved:")
    print(OUTPUT_PATH)

    print("=" * 60)


if __name__ == "__main__":
    main()