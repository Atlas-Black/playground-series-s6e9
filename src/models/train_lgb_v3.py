from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.features.build_features_v3 import V3_FEATURES


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "processed" / "train_engineered_v3.parquet"
)

TEST_PATH = (
    ROOT / "data" / "processed" / "test_engineered_v3.parquet"
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
# Prepare categorical features
# =========================================================

def prepare_categorical(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Make train/test categorical codes identical.

    LightGBM can directly handle pandas category dtype.
    """

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

        train_values = (
            train[col]
            .astype(str)
            .str.strip()
        )

        test_values = (
            test[col]
            .astype(str)
            .str.strip()
        )

        categories = sorted(
            set(train_values.unique())
            | set(test_values.unique())
        )

        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        train[col] = train_values.astype(dtype)
        test[col] = test_values.astype(dtype)

    return train, test


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("LightGBM V3 - V3 Feature Experiment")
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
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
    # Checks
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
            f"Train missing V3 features: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Test missing V3 features: {missing_test}"
        )

    if TARGET not in train.columns:
        raise ValueError(
            f"Train missing target: {TARGET}"
        )

    if "fold" not in train.columns:
        raise ValueError(
            "Train missing fold."
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
    # Categorical preparation
    # -----------------------------------------------------

    print(
        "\nPreparing categorical features..."
    )

    train, test = prepare_categorical(
        train,
        test,
    )

    X = train[V3_FEATURES].copy()
    y = train[TARGET].copy()

    X_test = test[V3_FEATURES].copy()

    folds = train["fold"].to_numpy()

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
    # CV
    # =====================================================

    for fold in range(N_FOLDS):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Fold {fold}"
        )

        print(
            "=" * 70
        )

        train_idx = np.where(
            folds != fold
        )[0]

        valid_idx = np.where(
            folds == fold
        )[0]

        X_train = X.iloc[
            train_idx
        ].copy()

        y_train = y.iloc[
            train_idx
        ].copy()

        X_valid = X.iloc[
            valid_idx
        ].copy()

        y_valid = y.iloc[
            valid_idx
        ].copy()

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model = lgb.LGBMClassifier(

            objective="binary",

            n_estimators=5000,
            learning_rate=0.03,

            num_leaves=31,
            max_depth=-1,

            min_child_samples=40,

            subsample=0.85,
            subsample_freq=1,

            colsample_bytree=0.85,

            reg_alpha=0.1,
            reg_lambda=1.0,

            random_state=SEED,
            n_jobs=-1,

            verbosity=-1,
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

            eval_metric="auc",

            categorical_feature=CATEGORICAL_COLS,

            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=150,
                    verbose=False,
                ),

                lgb.log_evaluation(
                    period=100
                ),
            ],
        )

        # -------------------------------------------------
        # Validation prediction
        # -------------------------------------------------

        valid_pred = (
            model.predict_proba(
                X_valid,
                num_iteration=model.best_iteration_,
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

        best_iterations.append(
            model.best_iteration_
        )

        print(
            f"\nFold {fold} AUC: "
            f"{fold_auc:.6f}"
        )

        print(
            "Best iteration:",
            model.best_iteration_
        )

        # -------------------------------------------------
        # Test prediction
        # -------------------------------------------------

        fold_test_pred = (
            model.predict_proba(
                X_test,
                num_iteration=model.best_iteration_,
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
        + "=" * 70
    )

    print(
        "LightGBM V3 Results"
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
        "\n"
        + "-" * 70
    )

    print(
        f"OOF ROC-AUC: "
        f"{overall_auc:.6f}"
    )

    print(
        f"Mean fold AUC: "
        f"{np.mean(fold_scores):.6f}"
    )

    print(
        f"Std fold AUC: "
        f"{np.std(fold_scores):.6f}"
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
    # Compare XGB V3 / V4
    # =====================================================

    comparison_files = {
        "XGB_V3": (
            ROOT
            / "predictions"
            / "oof"
            / "xgb_v3.csv"
        ),

        "XGB_V4_TE": (
            ROOT
            / "predictions"
            / "oof"
            / "xgb_v4_te.csv"
        ),
    }

    comparison_predictions = {}

    print(
        "\nComparison:"
    )

    print(
        f"LGB_V3    : "
        f"{overall_auc:.6f}"
    )

    for name, path in comparison_files.items():

        if not path.exists():
            continue

        df = pd.read_csv(
            path
        )

        if (
            TARGET not in df.columns
            or "prediction" not in df.columns
            or len(df) != len(train)
        ):
            continue

        auc = roc_auc_score(
            df[TARGET],
            df["prediction"],
        )

        comparison_predictions[
            name
        ] = df["prediction"].to_numpy()

        print(
            f"{name:<10}: "
            f"{auc:.6f}"
        )

    # =====================================================
    # Correlation
    # =====================================================

    if comparison_predictions:

        print(
            "\nOOF prediction correlation "
            "with LGB V3:"
        )

        for (
            name,
            prediction
        ) in comparison_predictions.items():

            corr = np.corrcoef(
                oof,
                prediction,
            )[0, 1]

            print(
                f"{name:<10}: "
                f"{corr:.6f}"
            )

    # =====================================================
    # Save
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

    oof_path = (
        oof_dir
        / "lgb_v3.csv"
    )

    test_path = (
        test_dir
        / "lgb_v3.csv"
    )

    pd.DataFrame({
        TARGET: y.to_numpy(),
        "prediction": oof,
    }).to_csv(
        oof_path,
        index=False,
    )

    pd.DataFrame({
        ID_COL: test[ID_COL].to_numpy(),
        "prediction": test_pred,
    }).to_csv(
        test_path,
        index=False,
    )

    print(
        "\nSaved:"
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
        "=" * 70
    )


if __name__ == "__main__":
    main()