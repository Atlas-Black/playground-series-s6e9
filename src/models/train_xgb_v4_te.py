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
    ROOT / "data" / "processed" / "train_engineered_v3.parquet"
)

TEST_PATH = (
    ROOT / "data" / "processed" / "test_engineered_v3.parquet"
)

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_FOLDS = 5
SEED = 42

# Target Encoding smoothing
TE_SMOOTHING = 20.0


# =========================================================
# Original categorical columns
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
# Target Encoding definitions
# =========================================================

TE_SINGLE_COLS = [
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
    "Environmental_Concern_Level",
]

TE_INTERACTIONS = [
    (
        "Subsidy_Available",
        "Environmental_Concern_Level",
    ),
    (
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ),
    (
        "Home_Charging_Possible",
        "City_Type",
    ),
]


# =========================================================
# Helpers
# =========================================================

def prepare_categorical(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Make train/test categorical dtypes identical.
    """

    train = train.copy()
    test = test.copy()

    for col in CATEGORICAL_COLS:

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


def make_key(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.Series:
    """
    Create a stable string key for multi-column TE.
    """

    if len(cols) == 1:
        return (
            df[cols[0]]
            .astype(str)
            .fillna("__MISSING__")
        )

    key = (
        df[cols[0]]
        .astype(str)
        .fillna("__MISSING__")
    )

    for col in cols[1:]:

        key = (
            key
            + "||"
            + df[col]
            .astype(str)
            .fillna("__MISSING__")
        )

    return key


def fit_target_encoder(
    df: pd.DataFrame,
    y: pd.Series,
    cols: list[str],
    smoothing: float,
):
    """
    Fit smoothed target encoding using TRAINING PORTION ONLY.

    TE(category) =
        (sum_target + smoothing * global_mean)
        / (count + smoothing)
    """

    key = make_key(
        df,
        cols,
    )

    temp = pd.DataFrame({
        "_key": key.to_numpy(),
        "_target": y.to_numpy(),
    })

    stats = (
        temp
        .groupby(
            "_key",
            observed=True,
        )["_target"]
        .agg(
            ["sum", "count"]
        )
    )

    global_mean = float(
        y.mean()
    )

    stats["te"] = (
        stats["sum"]
        + smoothing * global_mean
    ) / (
        stats["count"]
        + smoothing
    )

    mapping = stats["te"]

    return mapping, global_mean


def apply_target_encoder(
    df: pd.DataFrame,
    cols: list[str],
    mapping: pd.Series,
    global_mean: float,
) -> np.ndarray:
    """
    Apply fitted TE mapping.
    Unknown categories -> training global mean.
    """

    key = make_key(
        df,
        cols,
    )

    encoded = (
        key
        .map(mapping)
        .fillna(global_mean)
        .astype("float32")
        .to_numpy()
    )

    return encoded


def add_fold_safe_te(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Fit every TE feature ONLY on current fold's training portion.

    Returns:
        X_train_te
        X_valid_te
        X_test_te
        te_feature_names
    """

    X_train_te = X_train.copy()
    X_valid_te = X_valid.copy()
    X_test_te = X_test.copy()

    te_feature_names = []

    # -----------------------------------------------------
    # Single-column TE
    # -----------------------------------------------------

    for col in TE_SINGLE_COLS:

        feature_name = (
            f"TE_{col}"
        )

        mapping, global_mean = (
            fit_target_encoder(
                X_train,
                y_train,
                [col],
                TE_SMOOTHING,
            )
        )

        X_train_te[feature_name] = (
            apply_target_encoder(
                X_train,
                [col],
                mapping,
                global_mean,
            )
        )

        X_valid_te[feature_name] = (
            apply_target_encoder(
                X_valid,
                [col],
                mapping,
                global_mean,
            )
        )

        X_test_te[feature_name] = (
            apply_target_encoder(
                X_test,
                [col],
                mapping,
                global_mean,
            )
        )

        te_feature_names.append(
            feature_name
        )

    # -----------------------------------------------------
    # Interaction TE
    # -----------------------------------------------------

    for col1, col2 in TE_INTERACTIONS:

        feature_name = (
            f"TE_{col1}__{col2}"
        )

        mapping, global_mean = (
            fit_target_encoder(
                X_train,
                y_train,
                [col1, col2],
                TE_SMOOTHING,
            )
        )

        X_train_te[feature_name] = (
            apply_target_encoder(
                X_train,
                [col1, col2],
                mapping,
                global_mean,
            )
        )

        X_valid_te[feature_name] = (
            apply_target_encoder(
                X_valid,
                [col1, col2],
                mapping,
                global_mean,
            )
        )

        X_test_te[feature_name] = (
            apply_target_encoder(
                X_test,
                [col1, col2],
                mapping,
                global_mean,
            )
        )

        te_feature_names.append(
            feature_name
        )

    return (
        X_train_te,
        X_valid_te,
        X_test_te,
        te_feature_names,
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("XGBoost V4 - Fold-safe Target Encoding")
    print("=" * 70)

    # -----------------------------------------------------
    # Load V3
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

    if TARGET not in train.columns:
        raise ValueError(
            f"Missing target: {TARGET}"
        )

    if "fold" not in train.columns:
        raise ValueError(
            "Missing fold column."
        )

    if ID_COL not in test.columns:
        raise ValueError(
            f"Missing test ID: {ID_COL}"
        )

    missing_train = [
        c
        for c in V3_FEATURES
        if c not in train.columns
    ]

    missing_test = [
        c
        for c in V3_FEATURES
        if c not in test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Train missing features: "
            f"{missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Test missing features: "
            f"{missing_test}"
        )

    # -----------------------------------------------------
    # Prepare categorical dtype
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

    X_test = (
        test[V3_FEATURES]
        .copy()
    )

    folds = (
        train["fold"]
        .to_numpy()
    )

    # -----------------------------------------------------
    # Predictions
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

        X_train = (
            X.iloc[train_idx]
            .copy()
        )

        y_train = (
            y.iloc[train_idx]
            .copy()
        )

        X_valid = (
            X.iloc[valid_idx]
            .copy()
        )

        y_valid = (
            y.iloc[valid_idx]
            .copy()
        )

        fold_test = (
            X_test.copy()
        )

        # -------------------------------------------------
        # Fold-safe TE
        # -------------------------------------------------

        (
            X_train_te,
            X_valid_te,
            X_test_te,
            te_features,
        ) = add_fold_safe_te(
            X_train,
            y_train,
            X_valid,
            fold_test,
        )

        print(
            "Base features:",
            len(V3_FEATURES)
        )

        print(
            "TE features:",
            len(te_features)
        )

        print(
            "Total features:",
            X_train_te.shape[1]
        )

        if fold == 0:

            print(
                "\nTE features:"
            )

            for feature in te_features:
                print(
                    " -",
                    feature
                )

        # -------------------------------------------------
        # XGBoost
        #
        # Keep V3 parameters unchanged.
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
            X_train_te,
            y_train,

            eval_set=[
                (
                    X_valid_te,
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
                X_valid_te
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
        # Test
        # -------------------------------------------------

        fold_test_pred = (
            model.predict_proba(
                X_test_te
            )[:, 1]
        )

        test_pred += (
            fold_test_pred
            / N_FOLDS
        )

    # =====================================================
    # Overall
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
        "XGBoost V4 + Fold-safe TE Results"
    )

    print(
        "=" * 70
    )

    print(
        "\nFold AUC:"
    )

    for fold, score in enumerate(
        fold_scores
    ):

        print(
            f"Fold {fold}: "
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
    # Compare V3
    # =====================================================

    v3_path = (
        ROOT
        / "predictions"
        / "oof"
        / "xgb_v3.csv"
    )

    if v3_path.exists():

        v3 = pd.read_csv(
            v3_path
        )

        if (
            TARGET in v3.columns
            and "prediction" in v3.columns
            and len(v3) == len(train)
        ):

            v3_auc = roc_auc_score(
                v3[TARGET],
                v3["prediction"],
            )

            delta = (
                overall_auc
                - v3_auc
            )

            print(
                "\nComparison with XGB V3:"
            )

            print(
                f"XGB V3    : "
                f"{v3_auc:.6f}"
            )

            print(
                f"XGB V4 TE : "
                f"{overall_auc:.6f}"
            )

            print(
                f"Delta     : "
                f"{delta:+.6f}"
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
        / "xgb_v4_te.csv"
    )

    test_path = (
        test_dir
        / "xgb_v4_te.csv"
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