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
# =========================================================

FEATURES = V2_FEATURES

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
    print("CatBoost FEAT_V2 - V6 Params + V2 Features")
    print("=" * 60)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

    print("\nFeature count:", len(FEATURES))

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
        col for col in FEATURES
        if col not in train.columns
    ]

    missing_test = [
        col for col in FEATURES
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

    if ID_COL not in test.columns:
        raise ValueError(
            f"Test missing ID column: {ID_COL}"
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
        # IMPORTANT:
        # 与 CatBoost V6 保持完全相同的参数
        # 只改变 features
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
    #
    # V2 train 没有 id，因此这里只保存：
    # target + prediction
    # -----------------------------------------------------

    oof_df = pd.DataFrame({
        ID_COL: train[ID_COL],
        TARGET: y,
        "prediction": oof,
    })

    oof_path = (
        oof_dir / "cat_feat_v2.csv"
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

    test_path = (
        test_dir / "cat_feat_v2.csv"
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\nPredictions saved:")
    print(oof_path)
    print(test_path)

    print("\n" + "=" * 60)

    print("Fold AUC:")

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

    print("=" * 60)


if __name__ == "__main__":
    main()