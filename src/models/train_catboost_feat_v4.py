#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CatBoost Feature V4

实验目的：
    在 CAT_FEAT_V2 基础上，仅修改 CatBoost depth。

V2:
    全部 V2 engineered features
    depth = 6
    OOF AUC = 0.942020

V3:
    删除 Commute_Charging_Pressure
    depth = 6
    OOF AUC = 0.942014
    Difference vs V2 = -0.000006
    -> 没有提升，因此不采用 V3 特征方案

V4:
    恢复全部 V2 engineered features
    唯一修改：
        depth = 7

目的：
    测试更深的树是否能够学习 V2 交互特征中的更复杂关系。

注意：
    不覆盖 V2 / V3 预测文件。
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

# 当前正式基线
V2_REFERENCE_AUC = 0.942020

# V3 实验结果，仅用于记录
V3_REFERENCE_AUC = 0.942014


# =========================================================
# Features
# =========================================================
#
# 这里恢复完整 V2 特征。
#
# 原始特征：
#   Age
#   Annual_Income_USD
#   Daily_Commute_km
#   Gender
#   City_Type
#   Current_Car_Type
#   Home_Charging_Possible
#   Subsidy_Available
#   Range_Anxiety_Level
#   Number_of_Cars_Owned
#   Charging_Stations_Near_Home
#   Charging_Stations_Near_Work
#   Environmental_Concern_Level
#
# V1 engineered：
#   Total_Charging_Stations
#   Charging_Per_Commute
#   Has_Home_Charging
#   Income_Per_Age
#   Range_Anxiety_Num
#   Anxiety_Commute_Ratio
#   Has_Multiple_Cars
#   Subsidy_Flag
#   Eco_Subsidy_Score
#
# V2 new：
#   Income_Per_Car
#   Income_Subsidy_Interaction
#   HomeCharging_Commute
#   Charging_Anxiety_Interaction
#   Commute_Charging_Pressure
#   Eco_Income_Interaction
#   Eco_HomeCharging_Score
#
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
    # V1 engineered
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
    # V2 engineered
    # -----------------------------------------------------

    "Income_Per_Car",
    "Income_Subsidy_Interaction",
    "HomeCharging_Commute",
    "Charging_Anxiety_Interaction",

    # V3 删除过它，但结果下降
    # 因此 V4 恢复
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
# Categorical preprocessing
# =========================================================

def prepare_categorical(
    train: pd.DataFrame,
    test: pd.DataFrame,
):

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
    print("CatBoost FEAT_V4")
    print("Full V2 Features + depth=7")
    print("=" * 70)

    print("\nExperiment:")
    print("V2 baseline : depth=6")
    print("V4          : depth=7")
    print("All V2 features are restored.")
    print(
        "Commute_Charging_Pressure is INCLUDED."
    )

    # =====================================================
    # Load
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
    # Basic checks
    # =====================================================

    print("\nChecking columns...")

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
            "Train missing features: "
            f"{missing_train}"
        )

    if missing_test:

        raise ValueError(
            "Test missing features: "
            f"{missing_test}"
        )

    if TARGET not in train.columns:

        raise ValueError(
            f"Train missing target: {TARGET}"
        )

    if "fold" not in train.columns:

        raise ValueError(
            "Train missing fold column."
        )

    if ID_COL not in test.columns:

        raise ValueError(
            f"Test missing ID column: {ID_COL}"
        )

    print(
        "Feature count:",
        len(FEATURES)
    )

    print(
        "\nV2 interaction features:"
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
    # Fold distribution
    # =====================================================

    print(
        "\nFold distribution:"
    )

    print(
        train["fold"]
        .value_counts()
        .sort_index()
    )

    # =====================================================
    # Prepare categorical
    # =====================================================

    train, test = (
        prepare_categorical(
            train,
            test,
        )
    )

    # =====================================================
    # Dataset
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
    # 5-Fold CV
    # =====================================================

    for fold in range(
        N_FOLDS
    ):

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

        X_train = (
            X.iloc[
                train_idx
            ]
        )

        y_train = (
            y.iloc[
                train_idx
            ]
        )

        X_valid = (
            X.iloc[
                valid_idx
            ]
        )

        y_valid = (
            y.iloc[
                valid_idx
            ]
        )

        print(
            "Train rows:",
            len(
                train_idx
            )
        )

        print(
            "Valid rows:",
            len(
                valid_idx
            )
        )

        # =================================================
        # Model
        # =================================================
        #
        # V2:
        #   depth = 6
        #
        # V4:
        #   depth = 7
        #
        # 其他参数保持一致。
        #
        # =================================================

        model = CatBoostClassifier(

            iterations=2000,

            learning_rate=0.05,

            # =============================================
            # V4 唯一核心修改
            # =============================================

            depth=7,

            # =============================================

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
        # Validation
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
        # Fold AUC
        # =================================================

        fold_auc = (
            roc_auc_score(
                y_valid,
                valid_pred,
            )
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

        test_pred += (
            fold_test_pred
            / N_FOLDS
        )

    # =====================================================
    # Overall OOF
    # =====================================================

    overall_auc = (
        roc_auc_score(
            y,
            oof,
        )
    )

    average_best_iteration = (
        np.mean(
            best_iterations
        )
    )

    # =====================================================
    # Print result
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CatBoost FEAT_V4 Results"
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
    # Comparison
    # =====================================================

    difference_v2 = (
        overall_auc
        - V2_REFERENCE_AUC
    )

    difference_v3 = (
        overall_auc
        - V3_REFERENCE_AUC
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
        f"{overall_auc:.6f}"
    )

    print(
        "\nV4 - V2:",
        f"{difference_v2:+.6f}"
    )

    print(
        "V4 - V3:",
        f"{difference_v3:+.6f}"
    )

    # -----------------------------------------------------
    # Automatic interpretation
    # -----------------------------------------------------

    if overall_auc > V2_REFERENCE_AUC:

        print(
            "\nResult: "
            "V4 improved over the current V2 baseline."
        )

    elif overall_auc < V2_REFERENCE_AUC:

        print(
            "\nResult: "
            "V4 is worse than the current V2 baseline."
        )

    else:

        print(
            "\nResult: "
            "V4 and V2 are equal."
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
    # =====================================================
    #
    # 这里故意不要求 train 中存在 id。
    #
    # 你刚才 V3 已经遇到过：
    #
    # ValueError:
    # Train missing ID column: id
    #
    # ensemble 阶段实际上只需要：
    #   target
    #   prediction
    #
    # 所以这里沿用修正后的方式。
    #
    # =====================================================

    oof_df = pd.DataFrame({

        TARGET:
            y.to_numpy(),

        "prediction":
            oof,
    })

    oof_path = (
        oof_dir
        / "cat_feat_v4.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    # =====================================================
    # Save Test
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
        / "cat_feat_v4.csv"
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    # =====================================================
    # Final
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
        "\nV2 and V3 prediction files "
        "were NOT overwritten."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()