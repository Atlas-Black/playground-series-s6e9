#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ensemble V4

目标：
1. 保留现有所有模型结果，不覆盖 V1/V2/V3。
2. 加入最新 CAT_FEAT_V5。
3. 分析模型 OOF AUC 和 prediction correlation。
4. 使用两阶段权重搜索：
   - Stage 1: step = 0.05
   - Stage 2: step = 0.01，在最优解附近精搜
5. 自动生成新的 OOF/Test ensemble prediction。

Current important models:
    XGB_V1
    CAT_V6
    CAT_FEAT_V2
    CAT_FEAT_V5

说明：
LGB 在之前 ensemble 搜索中权重已经降到 0，
因此本版本不再浪费搜索维度在 LGB 上。
"""

from pathlib import Path
from itertools import product

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


# =========================================================
# Model files
# =========================================================

MODEL_FILES = {
    "XGB": "xgb_v1.csv",
    "CAT_V6": "cat_v6.csv",
    "CAT_FEAT_V2": "cat_feat_v2.csv",
    "CAT_FEAT_V5": "cat_feat_v5.csv",
}


# =========================================================
# Utilities
# =========================================================

def load_prediction_file(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"\nPrediction file not found:\n{path}\n"
        )

    df = pd.read_csv(path)

    if "prediction" not in df.columns:
        raise ValueError(
            f"{path.name} does not contain 'prediction'. "
            f"Columns = {df.columns.tolist()}"
        )

    return df


def check_length(
    name: str,
    df: pd.DataFrame,
    expected_length: int,
):

    if len(df) != expected_length:
        raise ValueError(
            f"{name}: row count mismatch. "
            f"Expected {expected_length}, got {len(df)}."
        )


def normalize_weights(weights):

    weights = np.asarray(
        weights,
        dtype=float
    )

    total = weights.sum()

    if total <= 0:
        raise ValueError(
            "Weight sum must be > 0."
        )

    return weights / total


def blend(pred_matrix, weights):

    weights = normalize_weights(weights)

    return np.average(
        pred_matrix,
        axis=1,
        weights=weights
    )


# =========================================================
# Search
# =========================================================

def coarse_search(
    y,
    pred_matrix,
    step=0.05,
):

    """
    4-model coarse search.

    Search weights whose sum = 1.

    step=0.05:
    only a few thousand valid combinations,
    so this should finish quickly.
    """

    print("\n" + "=" * 70)
    print("Stage 1 - Coarse weight search")
    print(f"Step = {step}")
    print("=" * 70)

    values = np.arange(
        0.0,
        1.0 + step / 2,
        step
    )

    best_auc = -np.inf
    best_weights = None

    results = []

    count = 0

    for w_xgb in values:

        for w_cat6 in values:

            for w_v2 in values:

                w_v5 = (
                    1.0
                    - w_xgb
                    - w_cat6
                    - w_v2
                )

                if w_v5 < -1e-9:
                    continue

                if w_v5 > 1.0 + 1e-9:
                    continue

                w_v5 = max(
                    0.0,
                    min(1.0, w_v5)
                )

                weights = np.array([
                    w_xgb,
                    w_cat6,
                    w_v2,
                    w_v5,
                ])

                pred = (
                    pred_matrix
                    @ weights
                )

                auc = roc_auc_score(
                    y,
                    pred
                )

                results.append(
                    (
                        auc,
                        *weights
                    )
                )

                count += 1

                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights.copy()

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print(
        f"\nEvaluated combinations: {count}"
    )

    return (
        best_auc,
        best_weights,
        results
    )


def fine_search(
    y,
    pred_matrix,
    center_weights,
    radius=0.06,
    step=0.01,
):

    """
    Fine search around coarse best weights.

    Instead of searching the whole simplex at step=0.01,
    only search around the best coarse solution.
    """

    print("\n" + "=" * 70)
    print("Stage 2 - Fine weight search")
    print(f"Radius = ±{radius}")
    print(f"Step   = {step}")
    print("=" * 70)

    candidate_ranges = []

    for w in center_weights:

        low = max(
            0.0,
            w - radius
        )

        high = min(
            1.0,
            w + radius
        )

        values = np.arange(
            low,
            high + step / 2,
            step
        )

        candidate_ranges.append(
            values
        )

    best_auc = -np.inf
    best_weights = None

    results = []

    count = 0

    # 前三个权重枚举
    # 第四个由 1 - sum 自动计算
    for w_xgb, w_cat6, w_v2 in product(
        candidate_ranges[0],
        candidate_ranges[1],
        candidate_ranges[2],
    ):

        w_v5 = (
            1.0
            - w_xgb
            - w_cat6
            - w_v2
        )

        # 必须在 V5 搜索范围内
        if (
            w_v5
            < candidate_ranges[3][0] - 1e-9
            or
            w_v5
            > candidate_ranges[3][-1] + 1e-9
        ):
            continue

        if w_v5 < 0 or w_v5 > 1:
            continue

        weights = np.array([
            w_xgb,
            w_cat6,
            w_v2,
            w_v5,
        ])

        # 防止浮点误差
        weights = normalize_weights(
            weights
        )

        pred = (
            pred_matrix
            @ weights
        )

        auc = roc_auc_score(
            y,
            pred
        )

        results.append(
            (
                auc,
                *weights
            )
        )

        count += 1

        if auc > best_auc:
            best_auc = auc
            best_weights = weights.copy()

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print(
        f"\nEvaluated combinations: {count}"
    )

    return (
        best_auc,
        best_weights,
        results
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("Ensemble V4")
    print("XGB + CAT_V6 + CAT_FEAT_V2 + CAT_FEAT_V5")
    print("=" * 70)

    # -----------------------------------------------------
    # Load OOF
    # -----------------------------------------------------

    oof_data = {}

    for name, filename in MODEL_FILES.items():

        path = OOF_DIR / filename

        print(
            f"Loading OOF: {name:<12} -> {filename}"
        )

        oof_data[name] = (
            load_prediction_file(path)
        )

    # -----------------------------------------------------
    # Reference
    # -----------------------------------------------------

    reference_name = "CAT_FEAT_V2"
    reference = oof_data[reference_name]

    n_train = len(reference)

    if TARGET not in reference.columns:
        raise ValueError(
            f"{MODEL_FILES[reference_name]} "
            f"does not contain target column '{TARGET}'."
        )

    y = (
        reference[TARGET]
        .to_numpy()
    )

    # -----------------------------------------------------
    # Length checks
    # -----------------------------------------------------

    for name, df in oof_data.items():

        check_length(
            name,
            df,
            n_train
        )

    # -----------------------------------------------------
    # ID consistency
    # -----------------------------------------------------

    reference_ids = None

    if ID_COL in reference.columns:

        reference_ids = (
            reference[ID_COL]
            .to_numpy()
        )

        for name, df in oof_data.items():

            if ID_COL not in df.columns:
                continue

            ids = (
                df[ID_COL]
                .to_numpy()
            )

            if not np.array_equal(
                reference_ids,
                ids
            ):
                raise ValueError(
                    f"OOF ID order mismatch: {name}"
                )

    # -----------------------------------------------------
    # Prediction matrix
    # -----------------------------------------------------

    model_names = list(
        MODEL_FILES.keys()
    )

    pred_matrix = np.column_stack([
        oof_data[name]["prediction"].to_numpy()
        for name in model_names
    ])

    # -----------------------------------------------------
    # Individual AUC
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("Individual OOF AUC")
    print("=" * 70)

    individual_scores = {}

    for i, name in enumerate(model_names):

        auc = roc_auc_score(
            y,
            pred_matrix[:, i]
        )

        individual_scores[name] = auc

        print(
            f"{name:<12}: {auc:.6f}"
        )

    best_single_name = max(
        individual_scores,
        key=individual_scores.get
    )

    best_single_auc = (
        individual_scores[
            best_single_name
        ]
    )

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("Prediction Correlation")
    print("=" * 70)

    correlation = pd.DataFrame(
        pred_matrix,
        columns=model_names
    ).corr()

    print(
        correlation.round(6)
    )

    print("\nCorrelation with CAT_FEAT_V5:")

    for name in model_names:

        if name == "CAT_FEAT_V5":
            continue

        corr = correlation.loc[
            "CAT_FEAT_V5",
            name
        ]

        print(
            f"{name:<12}: {corr:.6f}"
        )

    # -----------------------------------------------------
    # Old ensemble V3
    # -----------------------------------------------------

    old_weights = np.array([
        0.13,   # XGB
        0.20,   # CAT_V6
        0.67,   # CAT_FEAT_V2
        0.00,   # CAT_FEAT_V5
    ])

    old_pred = blend(
        pred_matrix,
        old_weights
    )

    old_auc = roc_auc_score(
        y,
        old_pred
    )

    print("\n" + "=" * 70)
    print("Previous Ensemble V3")
    print("=" * 70)

    print(
        f"OOF AUC: {old_auc:.6f}"
    )

    # -----------------------------------------------------
    # Coarse search
    # -----------------------------------------------------

    (
        coarse_auc,
        coarse_weights,
        coarse_results
    ) = coarse_search(
        y,
        pred_matrix,
        step=0.05
    )

    print("\nBest coarse result:")

    print(
        f"AUC = {coarse_auc:.6f}"
    )

    for name, weight in zip(
        model_names,
        coarse_weights
    ):
        print(
            f"{name:<12}: {weight:.4f}"
        )

    # -----------------------------------------------------
    # Fine search
    # -----------------------------------------------------

    (
        best_auc,
        best_weights,
        fine_results
    ) = fine_search(
        y,
        pred_matrix,
        coarse_weights,
        radius=0.06,
        step=0.01
    )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("Best Ensemble V4")
    print("=" * 70)

    print(
        f"Best OOF AUC: {best_auc:.6f}"
    )

    print("\nWeights:")

    for name, weight in zip(
        model_names,
        best_weights
    ):

        print(
            f"{name:<12}: {weight:.4f}"
        )

    print("\nBest single model:")

    print(
        f"{best_single_name}: "
        f"{best_single_auc:.6f}"
    )

    print("\nImprovement:")

    print(
        "vs best single : "
        f"{best_auc - best_single_auc:+.6f}"
    )

    print(
        "vs Ensemble V3 : "
        f"{best_auc - old_auc:+.6f}"
    )

    # -----------------------------------------------------
    # Top combinations
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("Top 15 Fine Search Combinations")
    print("=" * 70)

    for row in fine_results[:15]:

        auc = row[0]
        weights = row[1:]

        text = (
            f"AUC={auc:.6f} | "
        )

        text += " ".join(
            f"{name}={weight:.2f}"
            for name, weight
            in zip(
                model_names,
                weights
            )
        )

        print(text)

    # -----------------------------------------------------
    # Load test predictions
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("Loading test predictions")
    print("=" * 70)

    test_data = {}

    for name, filename in MODEL_FILES.items():

        path = TEST_DIR / filename

        print(
            f"Loading TEST: {name:<12} -> {filename}"
        )

        test_data[name] = (
            load_prediction_file(path)
        )

    reference_test = (
        test_data["CAT_FEAT_V2"]
    )

    n_test = len(reference_test)

    for name, df in test_data.items():

        check_length(
            name,
            df,
            n_test
        )

    # -----------------------------------------------------
    # Test ID consistency
    # -----------------------------------------------------

    test_ids = None

    if ID_COL in reference_test.columns:

        test_ids = (
            reference_test[ID_COL]
            .to_numpy()
        )

        for name, df in test_data.items():

            if ID_COL not in df.columns:
                continue

            ids = (
                df[ID_COL]
                .to_numpy()
            )

            if not np.array_equal(
                test_ids,
                ids
            ):
                raise ValueError(
                    f"Test ID order mismatch: {name}"
                )

    # -----------------------------------------------------
    # Test blend
    # -----------------------------------------------------

    test_matrix = np.column_stack([
        test_data[name]["prediction"].to_numpy()
        for name in model_names
    ])

    ensemble_oof = blend(
        pred_matrix,
        best_weights
    )

    ensemble_test = blend(
        test_matrix,
        best_weights
    )

    # -----------------------------------------------------
    # Save OOF
    # -----------------------------------------------------

    oof_output = pd.DataFrame({
        TARGET: y,
        "prediction": ensemble_oof
    })

    if reference_ids is not None:

        oof_output.insert(
            0,
            ID_COL,
            reference_ids
        )

    oof_path = (
        OOF_DIR
        / "ensemble_v4.csv"
    )

    oof_output.to_csv(
        oof_path,
        index=False
    )

    # -----------------------------------------------------
    # Save test
    # -----------------------------------------------------

    test_output = pd.DataFrame({
        "prediction": ensemble_test
    })

    if test_ids is not None:

        test_output.insert(
            0,
            ID_COL,
            test_ids
        )

    test_path = (
        TEST_DIR
        / "ensemble_v4.csv"
    )

    test_output.to_csv(
        test_path,
        index=False
    )

    # -----------------------------------------------------
    # Save search result
    # -----------------------------------------------------

    result_rows = []

    for row in fine_results:

        result_rows.append({
            "auc": row[0],
            "xgb_weight": row[1],
            "cat_v6_weight": row[2],
            "cat_feat_v2_weight": row[3],
            "cat_feat_v5_weight": row[4],
        })

    result_df = pd.DataFrame(
        result_rows
    )

    search_path = (
        ROOT
        / "predictions"
        / "ensemble_v4_search.csv"
    )

    result_df.to_csv(
        search_path,
        index=False
    )

    # -----------------------------------------------------
    # Done
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("Saved")
    print("=" * 70)

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

    print("\nFinal:")

    print(
        f"OOF AUC = {best_auc:.6f}"
    )

    print(
        "Weights = "
        + ", ".join(
            f"{name}:{weight:.4f}"
            for name, weight
            in zip(
                model_names,
                best_weights
            )
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    main()