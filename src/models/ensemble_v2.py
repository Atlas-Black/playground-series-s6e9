from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

OOF_DIR = ROOT / "predictions" / "oof"
TEST_DIR = ROOT / "predictions" / "test"

TARGET = "Will_Buy_EV"

MODEL_NAMES = ["lgb", "cat", "xgb"]


# =========================================================
# Load predictions
# =========================================================

def load_predictions():

    model_versions = {
        "lgb": "v1",
        "cat": "v6",
        "xgb": "v1",
    }

    oof = {}
    test = {}

    for model, version in model_versions.items():

        oof[model] = pd.read_csv(
            OOF_DIR / f"{model}_{version}.csv"
        )

        test[model] = pd.read_csv(
            TEST_DIR / f"{model}_{version}.csv"
        )

    return oof, test


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 65)
    print("Ensemble V2 - LGB_V1 + CAT_V6 + XGB_V1")
    print("=" * 65)

    oof, test = load_predictions()

    # -----------------------------------------------------
    # Basic checks
    # -----------------------------------------------------

    n = len(oof["lgb"])

    for model in MODEL_NAMES:
        assert len(oof[model]) == n

    y = oof["lgb"][TARGET].to_numpy()

    # 确认三个 OOF 的标签完全一致
    for model in ["cat", "xgb"]:
        assert np.array_equal(
            y,
            oof[model][TARGET].to_numpy()
        )

    preds = {
        model: oof[model]["prediction"].to_numpy()
        for model in MODEL_NAMES
    }

    # -----------------------------------------------------
    # Individual model scores
    # -----------------------------------------------------

    print("\nIndividual OOF AUC:")

    for model in MODEL_NAMES:

        score = roc_auc_score(
            y,
            preds[model]
        )

        print(f"{model.upper():>4}: {score:.6f}")

    # -----------------------------------------------------
    # Prediction correlation
    # -----------------------------------------------------

    corr_df = pd.DataFrame({
        model: preds[model]
        for model in MODEL_NAMES
    })

    print("\nPrediction correlation:")
    print(corr_df.corr().round(6))

    # -----------------------------------------------------
    # Simple average
    # -----------------------------------------------------

    simple_pred = (
        preds["lgb"]
        + preds["cat"]
        + preds["xgb"]
    ) / 3

    simple_auc = roc_auc_score(
        y,
        simple_pred
    )

    print(f"\nEqual-weight ensemble AUC: {simple_auc:.6f}")

    # -----------------------------------------------------
    # Weight search
    #
    # 权重步长 = 0.02
    # w_lgb + w_cat + w_xgb = 1
    # -----------------------------------------------------

    best_auc = -1
    best_weights = None

    results = []

    step = 0.02

    weights = np.arange(
        0,
        1 + step / 2,
        step
    )

    for w_lgb in weights:

        for w_cat in weights:

            w_xgb = 1.0 - w_lgb - w_cat

            if w_xgb < -1e-9:
                continue

            if w_xgb > 1:
                continue

            w_xgb = max(0.0, w_xgb)

            ensemble_pred = (
                w_lgb * preds["lgb"]
                + w_cat * preds["cat"]
                + w_xgb * preds["xgb"]
            )

            auc = roc_auc_score(
                y,
                ensemble_pred
            )

            results.append(
                (
                    auc,
                    w_lgb,
                    w_cat,
                    w_xgb
                )
            )

            if auc > best_auc:

                best_auc = auc

                best_weights = (
                    w_lgb,
                    w_cat,
                    w_xgb
                )

    # -----------------------------------------------------
    # Best results
    # -----------------------------------------------------

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print("\n" + "=" * 65)
    print("Best Ensemble")
    print("=" * 65)

    print(f"Best OOF AUC: {best_auc:.6f}")

    print(
        "Weights:"
        f" LGB={best_weights[0]:.2f},"
        f" CAT={best_weights[1]:.2f},"
        f" XGB={best_weights[2]:.2f}"
    )

    print("\nTop 10 combinations:")

    for auc, w_lgb, w_cat, w_xgb in results[:10]:

        print(
            f"AUC={auc:.6f} | "
            f"LGB={w_lgb:.2f} "
            f"CAT={w_cat:.2f} "
            f"XGB={w_xgb:.2f}"
        )

    # -----------------------------------------------------
    # Save best ensemble OOF
    # -----------------------------------------------------

    best_oof_pred = (
        best_weights[0] * preds["lgb"]
        + best_weights[1] * preds["cat"]
        + best_weights[2] * preds["xgb"]
    )

    ensemble_oof = pd.DataFrame({
        TARGET: y,
        "prediction": best_oof_pred
    })

    ensemble_oof.to_csv(
        OOF_DIR / "ensemble_v2.csv",
        index=False
    )

    # -----------------------------------------------------
    # Test ensemble
    # -----------------------------------------------------

    test_preds = {
        model: test[model]["prediction"].to_numpy()
        for model in MODEL_NAMES
    }

    ensemble_test_pred = (
        best_weights[0] * test_preds["lgb"]
        + best_weights[1] * test_preds["cat"]
        + best_weights[2] * test_preds["xgb"]
    )

    # 用 LGB test 文件里的 ID，保持原始顺序
    id_col = [
        col
        for col in test["lgb"].columns
        if col != "prediction"
    ][0]

    ensemble_test = pd.DataFrame({
        id_col: test["lgb"][id_col],
        "prediction": ensemble_test_pred
    })

    ensemble_test.to_csv(
        TEST_DIR / "ensemble_v2.csv",
        index=False
    )

    print("\nSaved:")
    print(OOF_DIR / "ensemble_v2.csv")
    print(TEST_DIR / "ensemble_v2.csv")

    print("=" * 65)


if __name__ == "__main__":
    main()