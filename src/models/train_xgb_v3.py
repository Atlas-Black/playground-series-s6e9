from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.features.build_features_v3 import V3_FEATURES


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "train_engineered_v3.parquet"
)

TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "test_engineered_v3.parquet"
)

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_FOLDS = 5
SEED = 42


# =========================================================
# Categorical columns
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
# Align categorical dtypes
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
                f"Train missing categorical column: {col}"
            )

        if col not in test.columns:
            raise ValueError(
                f"Test missing categorical column: {col}"
            )

        train_col = (
            train[col]
            .astype(str)
            .str.strip()
        )

        test_col = (
            test[col]
            .astype(str)
            .str.strip()
        )

        # 保证 train / test 类别编码完全一致
        categories = sorted(
            set(train_col.unique())
            | set(test_col.unique())
        )

        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        train[col] = train_col.astype(dtype)
        test[col] = test_col.astype(dtype)

    return train, test


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 65)
    print("XGBoost V3 - V3 Feature Experiment")
    print("=" * 65)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    print("\nLoading V3 data...")

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

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    missing_train = [
        col
        for col in V3_FEATURES
        if col not in train.columns
    ]

    missing_test = [
        col
        for col in V3_FEATURES
        if col not in test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Train missing V3 features: "
            f"{missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Test missing V3 features: "
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
            f"Test missing ID: {ID_COL}"
        )

    print(
        "\nNumber of features:",
        len(V3_FEATURES)
    )

    print("\nFold distribution:")

    print(
        train["fold"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # Categorical preprocessing
    # -----------------------------------------------------

    print(
        "\nPreparing categorical features..."
    )

    train, test = prepare_categorical(
        train,
        test,
    )

    X = train[V3_FEATURES]
    y = train[TARGET]

    X_test = test[V3_FEATURES]

    folds = (
        train["fold"]
        .to_numpy()
    )

    # -----------------------------------------------------
    # Prediction arrays
    # -----------------------------------------------------

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
            f"\n========== Fold {fold} =========="
        )

        train_idx = np.where(
            folds != fold
        )[0]

        valid_idx = np.where(
            folds == fold
        )[0]

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

        # -------------------------------------------------
        # 与原 XGB V1 保持相同参数
        # -------------------------------------------------

        model = xgb.XGBClassifier(

            objective="binary:logistic",
            eval_metric="auc",

            n_estimators=2000,
            learning_rate=0.05,

            max_depth=7,
            min_child_weight=1,

            subsample=0.8,
            colsample_bytree=0.8,

            reg_alpha=0.0,
            reg_lambda=1.0,

            tree_method="hist",
            enable_categorical=True,

            random_state=SEED,
            n_jobs=-1,

            early_stopping_rounds=100,
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        model.fit(
            X_train,
            y_train,

            eval_set=[
                (
                    X_valid,
                    y_valid,
                )
            ],

            verbose=100,
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        valid_pred = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        oof[
            valid_idx
        ] = valid_pred

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred,
        )

        fold_scores.append(
            fold_auc
        )

        best_iteration = (
            model.best_iteration
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

        # -------------------------------------------------
        # Test prediction
        # -------------------------------------------------

        fold_test_pred = (
            model.predict_proba(
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

    overall_auc = roc_auc_score(
        y,
        oof,
    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        "XGBoost V3 Results"
    )

    print(
        "=" * 65
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
        "-" * 65
    )

    print(
        f"OOF ROC-AUC: "
        f"{overall_auc:.6f}"
    )

    print(
        "Mean fold AUC:",
        f"{np.mean(fold_scores):.6f}",
    )

    print(
        "Std fold AUC:",
        f"{np.std(fold_scores):.6f}",
    )

    print(
        "Average best iteration:",
        round(
            np.mean(
                best_iterations
            ),
            1,
        ),
    )

    # =====================================================
    # Compare against XGB V1
    # =====================================================

    old_oof_path = (
        ROOT
        / "predictions"
        / "oof"
        / "xgb_v1.csv"
    )

    if old_oof_path.exists():

        old_oof = pd.read_csv(
            old_oof_path
        )

        if (
            TARGET
            in old_oof.columns
            and "prediction"
            in old_oof.columns
            and len(old_oof)
            == len(train)
        ):

            old_auc = roc_auc_score(
                old_oof[TARGET],
                old_oof["prediction"],
            )

            improvement = (
                overall_auc
                - old_auc
            )

            print(
                "\nComparison with XGB V1:"
            )

            print(
                f"XGB V1 : "
                f"{old_auc:.6f}"
            )

            print(
                f"XGB V3 : "
                f"{overall_auc:.6f}"
            )

            print(
                f"Delta  : "
                f"{improvement:+.6f}"
            )

    # =====================================================
    # Save predictions
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

    # -----------------------------------------------------
    # V3 train 没有 id，所以 OOF 不强制保存 id
    # -----------------------------------------------------

    oof_df = pd.DataFrame({
        TARGET: y.to_numpy(),
        "prediction": oof,
    })

    oof_path = (
        oof_dir
        / "xgb_v3.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    # -----------------------------------------------------
    # Test 保留 id
    # -----------------------------------------------------

    test_df = pd.DataFrame({
        ID_COL: test[ID_COL],
        "prediction": test_pred,
    })

    test_path = (
        test_dir
        / "xgb_v3.csv"
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(
        "\nPredictions saved:"
    )

    print(
        oof_path
    )

    print(
        test_path
    )

    print(
        "\n"
        + "=" * 65
    )


if __name__ == "__main__":
    main()