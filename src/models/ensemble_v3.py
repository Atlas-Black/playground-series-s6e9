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

# 模型名称 -> prediction 文件名
MODEL_FILES = {
    "lgb": "lgb_v1.csv",
    "xgb": "xgb_v1.csv",
    "cat_v6": "cat_v6.csv",
    "cat_feat_v2": "cat_feat_v2.csv",
}

MODEL_NAMES = list(MODEL_FILES.keys())


# =========================================================
# Load predictions
# =========================================================

def load_predictions():

    oof = {}
    test = {}

    for model, filename in MODEL_FILES.items():

        oof_path = OOF_DIR / filename
        test_path = TEST_DIR / filename

        if not oof_path.exists():
            raise FileNotFoundError(
                f"OOF file not found: {oof_path}"
            )

        if not test_path.exists():
            raise FileNotFoundError(
                f"Test file not found: {test_path}"
            )

        oof[model] = pd.read_csv(oof_path)
        test[model] = pd.read_csv(test_path)

    return oof, test


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("Ensemble V3 - 4 Model OOF Weight Search")
    print("=" * 70)

    oof, test = load_predictions()

    # -----------------------------------------------------
    # Basic checks
    # -----------------------------------------------------

    n = len(oof["lgb"])

    for model in MODEL_NAMES:
        assert len(oof[model]) == n

    y = oof["lgb"][TARGET].to_numpy()

    # 检查所有模型标签一致
    for model in MODEL_NAMES:

        if model == "lgb":
            continue

        assert np.array_equal(
            y,
            oof[model][TARGET].to_numpy()
        ), f"Target mismatch: {model}"

    preds = {
        model: oof[model]["prediction"].to_numpy()
        for model in MODEL_NAMES
    }

    # -----------------------------------------------------
    # Individual scores
    # -----------------------------------------------------

    print("\nIndividual OOF AUC:")

    individual_scores = {}

    for model in MODEL_NAMES:

        score = roc_auc_score(
            y,
            preds[model]
        )

        individual_scores[model] = score

        print(
            f"{model.upper():>12}: "
            f"{score:.6f}"
        )

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
    # Equal weight
    # -----------------------------------------------------

    equal_pred = np.mean(
        [
            preds[model]
            for model in MODEL_NAMES
        ],
        axis=0
    )

    equal_auc = roc_auc_score(
        y,
        equal_pred
    )

    print(
        f"\nEqual-weight ensemble AUC: "
        f"{equal_auc:.6f}"
    )

    # =====================================================
    # Local Weight Search
    #
    # 根据 V3 粗搜索结果：
    # LGB = 0
    # XGB ≈ 0.15
    # CAT_V6 ≈ 0.20
    # CAT_FEAT_V2 ≈ 0.65
    #
    # 现在只在附近进行 0.01 精细搜索
    # =====================================================

    print("\nSearching local weights...")

    best_auc = -1
    best_weights = None

    results = []

    # XGB: 0.05 ~ 0.25
    xgb_weights = np.arange(
        0.05,
        0.251,
        0.01
    )

    # CAT_V6: 0.10 ~ 0.35
    cat6_weights = np.arange(
        0.10,
        0.351,
        0.01
    )

    for w_xgb in xgb_weights:

        for w_cat6 in cat6_weights:

            # LGB 已经确定设为 0
            w_lgb = 0.0

            # 剩余权重全部给 CAT_FEAT_V2
            w_cat_v2 = (
                    1.0
                    - w_xgb
                    - w_cat6
            )

            # 安全检查
            if w_cat_v2 < 0:
                continue

            ensemble_pred = (
                    w_xgb * preds["xgb"]
                    + w_cat6 * preds["cat_v6"]
                    + w_cat_v2 * preds["cat_feat_v2"]
            )

            auc = roc_auc_score(
                y,
                ensemble_pred
            )

            current_weights = (
                w_lgb,
                w_xgb,
                w_cat6,
                w_cat_v2,
            )

            results.append(
                (
                    auc,
                    *current_weights
                )
            )

            if auc > best_auc:
                best_auc = auc
                best_weights = current_weights

    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # =====================================================
    # Results
    # =====================================================

    print("\n" + "=" * 70)
    print("Best Ensemble")
    print("=" * 70)

    print(
        f"Best OOF AUC: "
        f"{best_auc:.6f}"
    )

    print(
        "Weights:"
        f" LGB={best_weights[0]:.2f},"
        f" XGB={best_weights[1]:.2f},"
        f" CAT_V6={best_weights[2]:.2f},"
        f" CAT_FEAT_V2={best_weights[3]:.2f}"
    )

    best_single_model = max(
        individual_scores,
        key=individual_scores.get
    )

    best_single_auc = individual_scores[
        best_single_model
    ]

    print(
        f"\nBest single model: "
        f"{best_single_model.upper()} "
        f"({best_single_auc:.6f})"
    )

    print(
        "Ensemble improvement: "
        f"{best_auc - best_single_auc:+.6f}"
    )

    # -----------------------------------------------------
    # Top 15
    # -----------------------------------------------------

    print("\nTop 15 combinations:")

    for (
        auc,
        w_lgb,
        w_xgb,
        w_cat6,
        w_cat_v2
    ) in results[:15]:

        print(
            f"AUC={auc:.6f} | "
            f"LGB={w_lgb:.2f} "
            f"XGB={w_xgb:.2f} "
            f"CAT6={w_cat6:.2f} "
            f"CAT_V2={w_cat_v2:.2f}"
        )

    # =====================================================
    # Save best OOF
    # =====================================================

    best_oof_pred = (
        best_weights[0] * preds["lgb"]
        + best_weights[1] * preds["xgb"]
        + best_weights[2] * preds["cat_v6"]
        + best_weights[3] * preds["cat_feat_v2"]
    )

    # 保留 ID
    id_col = "id"

    ensemble_oof = pd.DataFrame({
        id_col: oof["lgb"][id_col],
        TARGET: y,
        "prediction": best_oof_pred
    })

    oof_path = (
        OOF_DIR
        / "ensemble_v3.csv"
    )

    ensemble_oof.to_csv(
        oof_path,
        index=False
    )

    # =====================================================
    # Test ensemble
    # =====================================================

    test_preds = {
        model: test[model]["prediction"].to_numpy()
        for model in MODEL_NAMES
    }

    # 检查 test 长度
    test_n = len(test["lgb"])

    for model in MODEL_NAMES:
        assert len(test[model]) == test_n

    # 检查 ID 顺序
    for model in MODEL_NAMES:

        assert np.array_equal(
            test["lgb"][id_col].to_numpy(),
            test[model][id_col].to_numpy()
        ), f"Test ID mismatch: {model}"

    ensemble_test_pred = (
        best_weights[0] * test_preds["lgb"]
        + best_weights[1] * test_preds["xgb"]
        + best_weights[2] * test_preds["cat_v6"]
        + best_weights[3] * test_preds["cat_feat_v2"]
    )

    ensemble_test = pd.DataFrame({
        id_col: test["lgb"][id_col],
        "prediction": ensemble_test_pred
    })

    test_path = (
        TEST_DIR
        / "ensemble_v3.csv"
    )

    ensemble_test.to_csv(
        test_path,
        index=False
    )

    # =====================================================
    # Done
    # =====================================================

    print("\nSaved:")
    print(oof_path)
    print(test_path)

    print("=" * 70)


if __name__ == "__main__":
    main()