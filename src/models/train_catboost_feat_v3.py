from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.features.build_features_v2 import V2_FEATURES


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "processed" / "train_engineered_v2.parquet"
)

TEST_PATH = (
    ROOT / "data" / "processed" / "test_engineered_v2.parquet"
)

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_FOLDS = 5
SEED = 42


# =========================================================
# Features
#
# V3:
# 在 V2 的基础上，只删除消融实验中表现较弱的
# Commute_Charging_Pressure
#
# 其他所有特征保持不变
# =========================================================

REMOVED_FEATURES = [
    "Commute_Charging_Pressure",
]

FEATURES = [
    feature
    for feature in V2_FEATURES
    if feature not in REMOVED_FEATURES
]


# V1 中原本作为 categorical 处理的列
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
# CatBoost categorical preprocessing
# =========================================================

def prepare_categorical(train, test):

    train = train.copy()
    test = test.copy()

    for col in CATEGORICAL_COLS:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

    return train, test


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("CatBoost FEAT_V3")
    print("V2 Features - Commute_Charging_Pressure")
    print("=" * 60)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    print("\nTrain shape:", train.shape)
    print("Test shape :", test.shape)

    print("\nOriginal V2 feature count:", len(V2_FEATURES))
    print("V3 feature count         :", len(FEATURES))

    print("\nRemoved features:")

    for feature in REMOVED_FEATURES:
        print(" -", feature)

    print("\nFold distribution:")

    print(
        train["fold"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

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

    if TARGET not in train.columns:
        raise ValueError(
            f"Train missing target: {TARGET}"
        )

    if "fold" not in train.columns:
        raise ValueError(
            "Train missing fold column"
        )

    # if ID_COL not in train.columns:
    #     raise ValueError(
    #         f"Train missing ID column: {ID_COL}"
    #     )

    if ID_COL not in test.columns:
        raise ValueError(
            f"Test missing ID column: {ID_COL}"
        )

    # 确认删除的特征确实不存在于 V3 FEATURES
    for feature in REMOVED_FEATURES:

        if feature in FEATURES:
            raise ValueError(
                f"{feature} was not removed!"
            )

    # -----------------------------------------------------
    # Categorical preprocessing
    # -----------------------------------------------------

    train, test = prepare_categorical(
        train,
        test
    )

    X = train[FEATURES]
    y = train[TARGET]

    X_test = test[FEATURES]

    folds = train["fold"].to_numpy()

    # -----------------------------------------------------
    # Prediction arrays
    # -----------------------------------------------------

    oof = np.zeros(
        len(train),
        dtype=float
    )

    test_pred = np.zeros(
        len(test),
        dtype=float
    )

    fold_scores = []
    best_iterations = []

    # -----------------------------------------------------
    # 5-fold CV
    # -----------------------------------------------------

    for fold in range(N_FOLDS):

        print(
            f"\n========== Fold {fold} =========="
        )

        train_idx = np.where(
            folds != fold
        )[0]

        valid_idx = np.where(
            folds == fold
        )[0]

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        # -------------------------------------------------
        # IMPORTANT
        #
        # 与 FEAT_V2 使用完全相同的 CatBoost 参数
        #
        # 唯一变化：
        # 删除 Commute_Charging_Pressure
        # -------------------------------------------------

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=CATEGORICAL_COLS,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=100,
            verbose=100,
        )

        # -------------------------------------------------
        # Validation prediction
        # -------------------------------------------------

        valid_pred = model.predict_proba(
            X_valid
        )[:, 1]

        oof[valid_idx] = valid_pred

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred
        )

        fold_scores.append(
            fold_auc
        )

        best_iteration = (
            model.get_best_iteration()
        )

        best_iterations.append(
            best_iteration
        )

        print(
            f"Fold {fold} AUC: "
            f"{fold_auc:.6f}"
        )

        print(
            f"Best iteration: "
            f"{best_iteration}"
        )

        # -------------------------------------------------
        # Test prediction
        # -------------------------------------------------

        test_pred += (
            model.predict_proba(
                X_test
            )[:, 1]
            / N_FOLDS
        )

    # -----------------------------------------------------
    # Overall OOF
    # -----------------------------------------------------

    overall_auc = roc_auc_score(
        y,
        oof
    )

    # =========================================================
    # Save predictions
    # =========================================================

    oof_dir = (
        ROOT / "predictions" / "oof"
    )

    test_dir = (
        ROOT / "predictions" / "test"
    )

    oof_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # OOF
    # -----------------------------------------------------

    oof_df = pd.DataFrame({
        TARGET: y.to_numpy(),
        "prediction": oof,
    })

    # 注意：
    # 使用新文件名，不覆盖 V2
    oof_path = (
        oof_dir / "cat_feat_v3.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    test_df = pd.DataFrame({
        ID_COL: test[ID_COL].to_numpy(),
        "prediction": test_pred,
    })

    # 注意：
    # 使用新文件名，不覆盖 V2
    test_path = (
        test_dir / "cat_feat_v3.csv"
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("CatBoost FEAT_V3 Results")
    print("=" * 60)

    print("\nFold AUC:")

    for i, score in enumerate(
        fold_scores
    ):
        print(
            f"Fold {i}: {score:.6f}"
        )

    print("-" * 60)

    print(
        f"OOF ROC-AUC: "
        f"{overall_auc:.6f}"
    )

    print(
        "Average best iteration:",
        round(
            np.mean(best_iterations),
            1
        )
    )

    # -----------------------------------------------------
    # Compare against known V2 result
    # -----------------------------------------------------

    V2_AUC = 0.942020

    improvement = (
        overall_auc - V2_AUC
    )

    print("\n" + "-" * 60)

    print(
        f"FEAT_V2 reference AUC: "
        f"{V2_AUC:.6f}"
    )

    print(
        f"FEAT_V3 AUC          : "
        f"{overall_auc:.6f}"
    )

    print(
        f"Difference           : "
        f"{improvement:+.6f}"
    )

    if improvement > 0:

        print(
            "\nResult: V3 improved over V2."
        )

    elif improvement < 0:

        print(
            "\nResult: V3 is worse than V2."
        )

    else:

        print(
            "\nResult: V3 and V2 are equal."
        )

    # -----------------------------------------------------å
    # Saved files
    # -----------------------------------------------------

    print("\nPredictions saved:")

    print(oof_path)
    print(test_path)

    print("\nV2 files were NOT overwritten.")

    print("=" * 60)


if __name__ == "__main__":
    main()