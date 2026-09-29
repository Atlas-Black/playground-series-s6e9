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
ID_COL = "id"
PRED_COL = "prediction"


# =========================================================
# Models
# =========================================================
#
# LGB_V5 = 当前最强模型
#
# 如果某个文件不存在，程序会自动跳过，
# 不会因为一个旧模型文件缺失而直接报错。
# =========================================================

MODEL_FILES = {
    "LGB_V5": "lgb_v5.csv",
    "LGB_V4": "lgb_v4.csv",
    "XGB_V4_TE": "xgb_v4_te.csv",
    "XGB_V3": "xgb_v3.csv",
    "CAT_FEAT_V5": "cat_feat_v5.csv",
}


# =========================================================
# Load OOF
# =========================================================

def load_oof_models():

    models = {}
    y_true = None

    print("=" * 80)
    print("Loading OOF predictions")
    print("=" * 80)

    for name, filename in MODEL_FILES.items():

        path = OOF_DIR / filename

        if not path.exists():

            print(
                f"SKIP {name:<12} "
                f"-> file not found: {filename}"
            )

            continue

        df = pd.read_csv(path)

        if PRED_COL not in df.columns:

            print(
                f"SKIP {name:<12} "
                f"-> missing '{PRED_COL}'"
            )

            continue

        if TARGET not in df.columns:

            print(
                f"SKIP {name:<12} "
                f"-> missing '{TARGET}'"
            )

            continue

        if y_true is None:

            y_true = (
                df[TARGET]
                .to_numpy()
            )

        else:

            current_y = (
                df[TARGET]
                .to_numpy()
            )

            if len(current_y) != len(y_true):

                raise ValueError(
                    f"{name}: OOF length mismatch."
                )

            if not np.array_equal(
                current_y,
                y_true,
            ):

                raise ValueError(
                    f"{name}: target order mismatch."
                )

        models[name] = (
            df[PRED_COL]
            .to_numpy(
                dtype=np.float64
            )
        )

        print(
            f"Loaded {name:<12} "
            f"-> {filename}"
        )

    if y_true is None:

        raise RuntimeError(
            "No valid OOF prediction files found."
        )

    if "LGB_V5" not in models:

        raise RuntimeError(
            "LGB_V5 is required."
        )

    return y_true, models


# =========================================================
# Load test
# =========================================================

def load_test_models(
    selected_models,
):

    models = {}
    test_ids = None

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Loading TEST predictions"
    )

    print(
        "=" * 80
    )

    for name in selected_models:

        filename = MODEL_FILES[name]

        path = TEST_DIR / filename

        if not path.exists():

            raise FileNotFoundError(
                f"Missing TEST prediction: "
                f"{path}"
            )

        df = pd.read_csv(path)

        if PRED_COL not in df.columns:

            raise ValueError(
                f"{name}: missing "
                f"'{PRED_COL}' in test file."
            )

        if ID_COL not in df.columns:

            raise ValueError(
                f"{name}: missing "
                f"'{ID_COL}' in test file."
            )

        current_ids = (
            df[ID_COL]
            .to_numpy()
        )

        if test_ids is None:

            test_ids = current_ids

        else:

            if not np.array_equal(
                current_ids,
                test_ids,
            ):

                raise ValueError(
                    f"{name}: test ID order mismatch."
                )

        models[name] = (
            df[PRED_COL]
            .to_numpy(
                dtype=np.float64
            )
        )

        print(
            f"Loaded {name:<12} "
            f"-> {filename}"
        )

    return test_ids, models


# =========================================================
# AUC helper
# =========================================================

def auc(
    y_true,
    prediction,
):

    return roc_auc_score(
        y_true,
        prediction,
    )


# =========================================================
# Individual scores
# =========================================================

def print_individual_scores(
    y_true,
    models,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Individual OOF AUC"
    )

    print(
        "=" * 80
    )

    scores = {}

    for name, pred in models.items():

        score = auc(
            y_true,
            pred,
        )

        scores[name] = score

    scores = dict(
        sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    for name, score in scores.items():

        print(
            f"{name:<12}: "
            f"{score:.6f}"
        )

    return scores


# =========================================================
# Correlation
# =========================================================

def print_correlation(
    models,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OOF Prediction Correlation"
    )

    print(
        "=" * 80
    )

    df = pd.DataFrame(
        models
    )

    print(
        df.corr()
        .round(6)
        .to_string()
    )


# =========================================================
# Two-model search
# =========================================================

def two_model_search(
    y_true,
    base_pred,
    other_pred,
    base_name,
    other_name,
):

    results = []

    # other model weight:
    # 0.00 -> 0.50
    #
    # 0.01 step = only 51 AUC evaluations.
    for other_weight in np.arange(
        0.0,
        0.5001,
        0.01,
    ):

        base_weight = (
            1.0
            - other_weight
        )

        pred = (
            base_weight
            * base_pred
            + other_weight
            * other_pred
        )

        score = auc(
            y_true,
            pred,
        )

        results.append({
            "base_model": base_name,
            "other_model": other_name,
            "base_weight": base_weight,
            "other_weight": other_weight,
            "auc": score,
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return results_df


# =========================================================
# Three-model local search
# =========================================================

def three_model_search(
    y_true,
    pred_a,
    pred_b,
    pred_c,
    name_a,
    name_b,
    name_c,
):

    results = []

    # Core model A keeps at least 50%.
    #
    # 0.02 step keeps runtime reasonable:
    # only a few hundred combinations.
    weights = np.arange(
        0.0,
        0.5001,
        0.02,
    )

    for wb in weights:

        for wc in weights:

            wa = (
                1.0
                - wb
                - wc
            )

            if wa < 0.50:
                continue

            if wa < 0:
                continue

            pred = (
                wa * pred_a
                + wb * pred_b
                + wc * pred_c
            )

            score = auc(
                y_true,
                pred,
            )

            results.append({
                name_a: wa,
                name_b: wb,
                name_c: wc,
                "auc": score,
            })

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return results_df


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=" * 80
    )

    print(
        "ENSEMBLE V5"
    )

    print(
        "LGB V5 centered ensemble search"
    )

    print(
        "=" * 80
    )

    # =====================================================
    # Load OOF
    # =====================================================

    y_true, models = (
        load_oof_models()
    )

    # =====================================================
    # Individual AUC
    # =====================================================

    individual_scores = (
        print_individual_scores(
            y_true,
            models,
        )
    )

    # =====================================================
    # Correlation
    # =====================================================

    print_correlation(
        models
    )

    # =====================================================
    # Base
    # =====================================================

    base_name = "LGB_V5"

    base_pred = (
        models[base_name]
    )

    base_auc = auc(
        y_true,
        base_pred,
    )

    # =====================================================
    # Two-model searches
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Stage 1 - Two-model searches"
    )

    print(
        "=" * 80
    )

    two_model_results = []

    for other_name in models:

        if other_name == base_name:
            continue

        result_df = two_model_search(
            y_true=y_true,
            base_pred=base_pred,
            other_pred=models[
                other_name
            ],
            base_name=base_name,
            other_name=other_name,
        )

        best = (
            result_df.iloc[0]
        )

        improvement = (
            best["auc"]
            - base_auc
        )

        print(
            f"\n{base_name} + "
            f"{other_name}"
        )

        print(
            f"Best AUC     : "
            f"{best['auc']:.6f}"
        )

        print(
            f"{base_name} weight: "
            f"{best['base_weight']:.2f}"
        )

        print(
            f"{other_name} weight: "
            f"{best['other_weight']:.2f}"
        )

        print(
            f"Delta        : "
            f"{improvement:+.6f}"
        )

        two_model_results.append({
            "other_model": other_name,
            "auc": best["auc"],
            "base_weight": (
                best["base_weight"]
            ),
            "other_weight": (
                best["other_weight"]
            ),
            "delta": improvement,
        })

    two_df = pd.DataFrame(
        two_model_results
    )

    two_df = (
        two_df
        .sort_values(
            "auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Two-model ranking"
    )

    print(
        "-" * 80
    )

    print(
        two_df.to_string(
            index=False
        )
    )

    # =====================================================
    # Choose two best complementary models
    # =====================================================

    complementary = (
        two_df[
            two_df["other_weight"] > 0
        ]
        ["other_model"]
        .tolist()
    )

    # Need at least 2 auxiliary models
    if len(complementary) >= 2:

        second_name = (
            complementary[0]
        )

        third_name = (
            complementary[1]
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "Stage 2 - Three-model search"
        )

        print(
            "=" * 80
        )

        print(
            "Models:"
        )

        print(
            f"  {base_name}"
        )

        print(
            f"  {second_name}"
        )

        print(
            f"  {third_name}"
        )

        three_df = (
            three_model_search(
                y_true=y_true,

                pred_a=models[
                    base_name
                ],

                pred_b=models[
                    second_name
                ],

                pred_c=models[
                    third_name
                ],

                name_a=base_name,
                name_b=second_name,
                name_c=third_name,
            )
        )

        print(
            "\nTop 15:"
        )

        print(
            three_df
            .head(15)
            .to_string(
                index=False
            )
        )

        best_three = (
            three_df.iloc[0]
        )

        best_auc = (
            best_three["auc"]
        )

        final_weights = {
            base_name:
                best_three[
                    base_name
                ],

            second_name:
                best_three[
                    second_name
                ],

            third_name:
                best_three[
                    third_name
                ],
        }

    else:

        # Fallback:
        # best two-model blend
        best_two = (
            two_df.iloc[0]
        )

        second_name = (
            best_two[
                "other_model"
            ]
        )

        best_auc = (
            best_two["auc"]
        )

        final_weights = {
            base_name:
                best_two[
                    "base_weight"
                ],

            second_name:
                best_two[
                    "other_weight"
                ],
        }

        three_df = None

    # =====================================================
    # Final OOF
    # =====================================================

    final_oof = np.zeros(
        len(y_true),
        dtype=np.float64,
    )

    for name, weight in (
        final_weights.items()
    ):

        final_oof += (
            weight
            * models[name]
        )

    final_auc = auc(
        y_true,
        final_oof,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL ENSEMBLE V5"
    )

    print(
        "=" * 80
    )

    print(
        f"Best single LGB V5: "
        f"{base_auc:.6f}"
    )

    print(
        f"Ensemble OOF AUC   : "
        f"{final_auc:.6f}"
    )

    print(
        f"Improvement        : "
        f"{final_auc - base_auc:+.6f}"
    )

    print(
        "\nWeights:"
    )

    for name, weight in (
        final_weights.items()
    ):

        print(
            f"{name:<12}: "
            f"{weight:.4f}"
        )

    # =====================================================
    # Test ensemble
    # =====================================================

    selected_models = list(
        final_weights.keys()
    )

    test_ids, test_models = (
        load_test_models(
            selected_models
        )
    )

    final_test = np.zeros(
        len(test_ids),
        dtype=np.float64,
    )

    for name, weight in (
        final_weights.items()
    ):

        final_test += (
            weight
            * test_models[name]
        )

    # =====================================================
    # Save
    # =====================================================

    OOF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof_path = (
        OOF_DIR
        / "ensemble_v5.csv"
    )

    test_path = (
        TEST_DIR
        / "ensemble_v5.csv"
    )

    search_path = (
        ROOT
        / "predictions"
        / "ensemble_v5_search.csv"
    )

    pd.DataFrame({
        TARGET: y_true,
        PRED_COL: final_oof,
    }).to_csv(
        oof_path,
        index=False,
    )

    pd.DataFrame({
        ID_COL: test_ids,
        PRED_COL: final_test,
    }).to_csv(
        test_path,
        index=False,
    )

    two_df.to_csv(
        search_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Saved"
    )

    print(
        "=" * 80
    )

    print(
        "OOF :",
        oof_path
    )

    print(
        "TEST:",
        test_path
    )

    print(
        "Search:",
        search_path
    )


if __name__ == "__main__":
    main()