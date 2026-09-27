#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CatBoost Feature V5

实验目的
--------
在当前最佳 CAT_FEAT_V2 的基础上，只修改 CatBoost 的树深度：

    V2: depth = 6
        OOF AUC = 0.942020

    V3: depth = 6
        删除 Commute_Charging_Pressure
        OOF AUC = 0.942014

    V4: depth = 7
        恢复完整 V2 features
        OOF AUC = 0.941932

    V5: depth = 5
        使用完整 V2 features

本实验只测试：
    depth = 5

其余核心训练参数保持不变。

输出：
    predictions/oof/cat_feat_v5.csv
    predictions/test/cat_feat_v5.csv

不会覆盖 V2 / V3 / V4。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "train_engineered_v2.parquet"
)

TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "test_engineered_v2.parquet"
)

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_FOLDS = 5
SEED = 42


# =========================================================
# Previous experiment results
# =========================================================

V2_REFERENCE_AUC = 0.942020
V3_REFERENCE_AUC = 0.942014
V4_REFERENCE_AUC = 0.941932


# =========================================================
# Features
#
# 使用完整 V2 features
# =========================================================

FEATURES = [

    # -----------------------------------------------------
    # Original numerical
    # -----------------------------------------------------

    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",

    # -----------------------------------------------------
    # Original categorical
    # -----------------------------------------------------

    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",

    # -----------------------------------------------------
    # V1 engineered features
    # -----------------------------------------------------

    "Total_Charging_Stations",
    "Charging_Per_Commute",
    "Has_Home_Charging",
    "Income_Per_Age",
    "Range_Anxiety_Num",
    "Anxiety_Commute_Ratio",
    "Has_Multiple_Cars",
    "Subsidy_Flag",
    "Eco_Subsidy_Score",

    # -----------------------------------------------------
    # V2 engineered features
    # -----------------------------------------------------

    "Income_Per_Car",
    "Income_Subsidy_Interaction",
    "HomeCharging_Commute",
    "Charging_Anxiety_Interaction",
    "Commute_Charging_Pressure",
    "Eco_Income_Interaction",
    "Eco_HomeCharging_Score",
]


# =========================================================
# Categorical features
# =========================================================

CATEGORICAL_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]


# =========================================================
# Prepare categorical features
# =========================================================

def prepare_categorical(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train = train.copy()
    test = test.copy()

    for col in CATEGORICAL_COLS:

        if col not in train.columns:
            raise ValueError(
                f"Categorical column missing in train: {col}"
            )

        if col not in test.columns:
            raise ValueError(
                f"Categorical column missing in test: {col}"
            )

        train[col] = (
            train[col]
            .fillna("Missing")
            .astype(str)
        )

        test[col] = (
            test[col]
            .fillna("Missing")
            .astype(str)
        )

    return train, test


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("CatBoost FEAT_V5")
    print("Full V2 Features + depth=5")
    print("=" * 70)

    print("\nExperiment design:")
    print("V2 : full V2 features + depth=6")
    print("V3 : remove Commute_Charging_Pressure + depth=6")
    print("V4 : full V2 features + depth=7")
    print("V5 : full V2 features + depth=5")
    print("\nOnly tree depth is changed relative to V2.")

    # =====================================================
    # Load data
    # =====================================================

    print("\nLoading data...")

    train = pd.read_parquet(
        TRAIN_PATH
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    print(
        "Train shape:",
        train.shape
    )

    print(
        "Test shape :",
        test.shape
    )

    # =====================================================
    # Validate target / fold / test ID
    # =====================================================

    if TARGET not in train.columns:
        raise ValueError(
            f"Train missing target column: {TARGET}"
        )

    if "fold" not in train.columns:
        raise ValueError(
            "Train missing fold column."
        )

    if ID_COL not in test.columns:
        raise ValueError(
            f"Test missing ID column: {ID_COL}"
        )

    # =====================================================
    # Validate feature columns
    # =====================================================

    missing_train = [
        col
        for col in FEATURES
        if col not in train.columns
    ]

    missing_test = [
        col
        for col in FEATURES
        if col not in test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Train missing features: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Test missing features: {missing_test}"
        )

    # =====================================================
    # Check fold values
    # =====================================================

    expected_folds = set(
        range(N_FOLDS)
    )

    actual_folds = set(
        train["fold"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if actual_folds != expected_folds:
        raise ValueError(
            "Unexpected fold values. "
            f"Expected {sorted(expected_folds)}, "
            f"found {sorted(actual_folds)}"
        )

    # =====================================================
    # Information
    # =====================================================

    print(
        "\nFeature count:",
        len(FEATURES)
    )

    print(
        "\nFold distribution:"
    )

    print(
        train["fold"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nV2 engineered features:"
    )

    v2_new_features = [
        "Income_Per_Car",
        "Income_Subsidy_Interaction",
        "HomeCharging_Commute",
        "Charging_Anxiety_Interaction",
        "Commute_Charging_Pressure",
        "Eco_Income_Interaction",
        "Eco_HomeCharging_Score",
    ]

    for feature in v2_new_features:
        print(
            " -",
            feature
        )

    # =====================================================
    # Prepare categorical
    # =====================================================

    train, test = prepare_categorical(
        train,
        test,
    )

    # =====================================================
    # Build matrices
    # =====================================================

    X = train[
        FEATURES
    ].copy()

    y = train[
        TARGET
    ].copy()

    X_test = test[
        FEATURES
    ].copy()

    folds = (
        train["fold"]
        .to_numpy()
    )

    # =====================================================
    # Prediction arrays
    # =====================================================

    oof = np.zeros(
        len(train),
        dtype=np.float64,
    )

    test_pred = np.zeros(
        len(test),
        dtype=np.float64,
    )

    fold_scores = []
    best_iterations = []

    # =====================================================
    # 5-fold CV
    # =====================================================

    for fold in range(N_FOLDS):

        print(
            "\n"
            + "=" * 20
            + f" Fold {fold} "
            + "=" * 20
        )

        # -------------------------------------------------
        # Split
        # -------------------------------------------------

        train_idx = np.where(
            folds != fold
        )[0]

        valid_idx = np.where(
            folds == fold
        )[0]

        if len(train_idx) == 0:
            raise ValueError(
                f"Fold {fold}: empty training split."
            )

        if len(valid_idx) == 0:
            raise ValueError(
                f"Fold {fold}: empty validation split."
            )

        X_train = X.iloc[
            train_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        X_valid = X.iloc[
            valid_idx
        ]

        y_valid = y.iloc[
            valid_idx
        ]

        print(
            "Train rows:",
            len(train_idx)
        )

        print(
            "Valid rows:",
            len(valid_idx)
        )

        # =================================================
        # CatBoost V5
        #
        # 唯一核心修改：
        #
        # V2 depth = 6
        # V4 depth = 7
        # V5 depth = 5
        #
        # =================================================

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,

            # ---------------------------------------------
            # V5 experiment
            # ---------------------------------------------
            depth=5,

            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

        # =================================================
        # Train
        # =================================================

        model.fit(
            X_train,
            y_train,

            cat_features=CATEGORICAL_COLS,

            eval_set=(
                X_valid,
                y_valid,
            ),

            early_stopping_rounds=100,

            verbose=100,
        )

        # =================================================
        # Validation prediction
        # =================================================

        valid_pred = (
            model
            .predict_proba(
                X_valid
            )[:, 1]
        )

        oof[
            valid_idx
        ] = valid_pred

        # =================================================
        # Validation checks
        # =================================================

        if np.isnan(
            valid_pred
        ).any():

            raise ValueError(
                f"Fold {fold}: "
                "validation predictions contain NaN."
            )

        if np.isinf(
            valid_pred
        ).any():

            raise ValueError(
                f"Fold {fold}: "
                "validation predictions contain inf."
            )

        # =================================================
        # Fold AUC
        # =================================================

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred,
        )

        fold_scores.append(
            fold_auc
        )

        best_iteration = (
            model
            .get_best_iteration()
        )

        best_iterations.append(
            best_iteration
        )

        print(
            f"\nFold {fold} AUC: "
            f"{fold_auc:.6f}"
        )

        print(
            "Best iteration:",
            best_iteration
        )

        # =================================================
        # Test prediction
        # =================================================

        fold_test_pred = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )

        if np.isnan(
            fold_test_pred
        ).any():

            raise ValueError(
                f"Fold {fold}: "
                "test predictions contain NaN."
            )

        if np.isinf(
            fold_test_pred
        ).any():

            raise ValueError(
                f"Fold {fold}: "
                "test predictions contain inf."
            )

        test_pred += (
            fold_test_pred
            / N_FOLDS
        )

    # =====================================================
    # Final prediction validation
    # =====================================================

    if np.isnan(oof).any():
        raise ValueError(
            "OOF predictions contain NaN."
        )

    if np.isinf(oof).any():
        raise ValueError(
            "OOF predictions contain inf."
        )

    if np.isnan(test_pred).any():
        raise ValueError(
            "Final test predictions contain NaN."
        )

    if np.isinf(test_pred).any():
        raise ValueError(
            "Final test predictions contain inf."
        )

    # =====================================================
    # Overall OOF
    # =====================================================

    overall_auc = roc_auc_score(
        y,
        oof,
    )

    average_best_iteration = float(
        np.mean(
            best_iterations
        )
    )

    # =====================================================
    # Results
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CatBoost FEAT_V5 Results"
    )

    print(
        "=" * 70
    )

    print(
        "\nFold AUC:"
    )

    for i, score in enumerate(
        fold_scores
    ):

        print(
            f"Fold {i}: "
            f"{score:.6f}"
        )

    print(
        "-" * 70
    )

    print(
        f"OOF ROC-AUC: "
        f"{overall_auc:.6f}"
    )

    print(
        "Average best iteration:",
        round(
            average_best_iteration,
            1,
        )
    )

    # =====================================================
    # Experiment comparison
    # =====================================================

    difference_v2 = (
        overall_auc
        - V2_REFERENCE_AUC
    )

    difference_v3 = (
        overall_auc
        - V3_REFERENCE_AUC
    )

    difference_v4 = (
        overall_auc
        - V4_REFERENCE_AUC
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Experiment Comparison"
    )

    print(
        "=" * 70
    )

    print(
        f"FEAT_V2 "
        f"(depth=6, all V2 features) : "
        f"{V2_REFERENCE_AUC:.6f}"
    )

    print(
        f"FEAT_V3 "
        f"(remove pressure feature)  : "
        f"{V3_REFERENCE_AUC:.6f}"
    )

    print(
        f"FEAT_V4 "
        f"(depth=7, all V2 features) : "
        f"{V4_REFERENCE_AUC:.6f}"
    )

    print(
        f"FEAT_V5 "
        f"(depth=5, all V2 features) : "
        f"{overall_auc:.6f}"
    )

    print(
        "\nV5 - V2:",
        f"{difference_v2:+.6f}"
    )

    print(
        "V5 - V3:",
        f"{difference_v3:+.6f}"
    )

    print(
        "V5 - V4:",
        f"{difference_v4:+.6f}"
    )

    # =====================================================
    # Automatic interpretation
    # =====================================================

    if overall_auc > V2_REFERENCE_AUC:

        print(
            "\nResult: "
            "V5 improved over the current V2 baseline."
        )

    elif overall_auc < V2_REFERENCE_AUC:

        print(
            "\nResult: "
            "V5 is worse than the current V2 baseline."
        )

    else:

        print(
            "\nResult: "
            "V5 and V2 have equal OOF AUC."
        )

    # =====================================================
    # Save directories
    # =====================================================

    oof_dir = (
        ROOT
        / "predictions"
        / "oof"
    )

    test_dir = (
        ROOT
        / "predictions"
        / "test"
    )

    oof_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Save OOF
    #
    # train_engineered_v2.parquet 当前不依赖 id。
    # Ensemble 需要 TARGET + prediction 即可。
    # =====================================================

    oof_df = pd.DataFrame({
        TARGET: y.to_numpy(),
        "prediction": oof,
    })

    oof_path = (
        oof_dir
        / "cat_feat_v5.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    # =====================================================
    # Save test predictions
    # =====================================================

    test_df = pd.DataFrame({
        ID_COL:
            test[
                ID_COL
            ].to_numpy(),

        "prediction":
            test_pred,
    })

    test_path = (
        test_dir
        / "cat_feat_v5.csv"
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    # =====================================================
    # Saved-file validation
    # =====================================================

    if len(oof_df) != len(train):
        raise ValueError(
            "OOF output row count mismatch."
        )

    if len(test_df) != len(test):
        raise ValueError(
            "Test output row count mismatch."
        )

    if oof_df["prediction"].isna().any():
        raise ValueError(
            "OOF output contains NaN."
        )

    if test_df["prediction"].isna().any():
        raise ValueError(
            "Test output contains NaN."
        )

    # =====================================================
    # Done
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Predictions saved:"
    )

    print(
        oof_path
    )

    print(
        test_path
    )

    print(
        "\nV2 / V3 / V4 prediction files "
        "were NOT overwritten."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()