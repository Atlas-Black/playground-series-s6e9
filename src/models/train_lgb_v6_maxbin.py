from pathlib import Path
import time
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "processed" / "train_engineered_v4.parquet"
)

TEST_PATH = (
    ROOT / "data" / "processed" / "test_engineered_v4.parquet"
)

OUTPUT_DIR = ROOT / "outputs" / "lgb_v6_maxbin"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"
ID_COL = "id"

SEED = 42

MAX_BINS = [
    8191,
]

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
# Target
# =========================================================

def normalize_target(series):

    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int8")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }

    result = cleaned.map(mapping)

    if result.isna().any():

        unknown = (
            series[result.isna()]
            .unique()
        )

        raise ValueError(
            f"Unknown target values: {unknown}"
        )

    return result.astype("int8")


# =========================================================
# Categories
# =========================================================

def align_categories(train, test):

    train = train.copy()
    test = test.copy()

    for col in CATEGORICAL_COLS:

        if col not in train.columns:
            continue

        train_values = (
            train[col]
            .astype("string")
            .fillna("__MISSING__")
        )

        test_values = (
            test[col]
            .astype("string")
            .fillna("__MISSING__")
        )

        categories = sorted(
            set(train_values.unique())
            | set(test_values.unique())
        )

        train[col] = pd.Categorical(
            train_values,
            categories=categories,
        )

        test[col] = pd.Categorical(
            test_values,
            categories=categories,
        )

    return train, test


# =========================================================
# Train one configuration
# =========================================================

def train_config(
    train,
    test,
    features,
    max_bin,
):

    print("\n" + "=" * 100)
    print(f"LGB V6 | MAX_BIN = {max_bin}")
    print("=" * 100)

    y = train[TARGET].to_numpy()
    folds = train[FOLD_COL].to_numpy()

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

    active_categorical = [
        col
        for col in CATEGORICAL_COLS
        if col in features
    ]

    total_start = time.time()

    # =====================================================
    # 5 folds
    # =====================================================

    for fold in range(5):

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Fold {fold}"
        )

        print(
            "-" * 80
        )

        fold_start = time.time()

        train_idx = np.where(
            folds != fold
        )[0]

        valid_idx = np.where(
            folds == fold
        )[0]

        X_train = (
            train.iloc[train_idx][features]
        )

        X_valid = (
            train.iloc[valid_idx][features]
        )

        X_test = test[features]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # =================================================
        # Model
        # =================================================

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

            reg_alpha=0.0,
            reg_lambda=0.5,

            # V6 experiment
            max_bin=max_bin,

            random_state=SEED,

            n_jobs=-1,

            verbosity=-1,
        )

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

            categorical_feature=(
                active_categorical
            ),

            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=150,
                    verbose=False,
                )
            ],
        )

        # =================================================
        # Validation
        # =================================================

        valid_pred = (
            model.predict_proba(
                X_valid,
                num_iteration=(
                    model.best_iteration_
                ),
            )[:, 1]
        )

        oof[valid_idx] = valid_pred

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

        # =================================================
        # Test
        # =================================================

        fold_test_pred = (
            model.predict_proba(
                X_test,
                num_iteration=(
                    model.best_iteration_
                ),
            )[:, 1]
        )

        test_pred += (
            fold_test_pred / 5.0
        )

        elapsed = (
            time.time()
            - fold_start
        )

        print(
            f"AUC       : {fold_auc:.6f}"
        )

        print(
            f"Best iter : {model.best_iteration_}"
        )

        print(
            f"Time      : {elapsed / 60:.2f} min"
        )

        del model
        del X_train
        del X_valid
        del X_test

        gc.collect()

    # =====================================================
    # Overall OOF
    # =====================================================

    oof_auc = roc_auc_score(
        y,
        oof,
    )

    mean_fold_auc = float(
        np.mean(fold_scores)
    )

    std_fold_auc = float(
        np.std(fold_scores)
    )

    avg_best_iter = float(
        np.mean(best_iterations)
    )

    total_time = (
        time.time()
        - total_start
    ) / 60

    # =====================================================
    # Results
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"LGB V6 MAX_BIN={max_bin} RESULTS"
    )

    print(
        "=" * 100
    )

    for fold, score in enumerate(
        fold_scores
    ):

        print(
            f"Fold {fold}: "
            f"{score:.6f}"
        )

    print(
        f"\nMean fold AUC : "
        f"{mean_fold_auc:.6f}"
    )

    print(
        f"Std fold AUC  : "
        f"{std_fold_auc:.6f}"
    )

    print(
        f"OOF AUC       : "
        f"{oof_auc:.6f}"
    )

    print(
        f"Avg best iter : "
        f"{avg_best_iter:.1f}"
    )

    print(
        f"Runtime       : "
        f"{total_time:.2f} min"
    )

    # =====================================================
    # Save predictions
    # =====================================================

    oof_df = pd.DataFrame({
        "row_index": np.arange(len(train)),
        TARGET: y,
        "prediction": oof,
        FOLD_COL: folds,
    })

    test_df = pd.DataFrame({
        "row_index": np.arange(len(test)),
        "prediction": test_pred,
    })

    oof_path = (
        OUTPUT_DIR
        / f"oof_lgb_v6_bin{max_bin}.csv"
    )

    test_path = (
        OUTPUT_DIR
        / f"test_lgb_v6_bin{max_bin}.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(
        "\nSaved OOF :",
        oof_path,
    )

    print(
        "Saved test:",
        test_path,
    )

    return {
        "max_bin": max_bin,
        "fold_0": fold_scores[0],
        "fold_1": fold_scores[1],
        "fold_2": fold_scores[2],
        "fold_3": fold_scores[3],
        "fold_4": fold_scores[4],
        "mean_fold_auc": mean_fold_auc,
        "std_fold_auc": std_fold_auc,
        "oof_auc": oof_auc,
        "avg_best_iteration": avg_best_iter,
        "runtime_minutes": total_time,
    }


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=" * 100
    )

    print(
        "LGB V6 - FULL 5-FOLD MAX_BIN"
    )

    print(
        "=" * 100
    )

    # =====================================================
    # Load
    # =====================================================

    train = pd.read_parquet(
        TRAIN_PATH
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    train[TARGET] = normalize_target(
        train[TARGET]
    )

    train, test = align_categories(
        train,
        test,
    )

    features = [
        col
        for col in train.columns
        if col not in [
            TARGET,
            FOLD_COL,
            ID_COL,
        ]
    ]

    print(
        "\nTrain:",
        train.shape
    )

    print(
        "Test :",
        test.shape
    )

    print(
        "Features:",
        len(features)
    )

    if len(features) != 56:

        print(
            f"WARNING: expected 56 "
            f"features, found "
            f"{len(features)}"
        )

    # =====================================================
    # Train both
    # =====================================================

    results = []

    for max_bin in MAX_BINS:

        result = train_config(
            train=train,
            test=test,
            features=features,
            max_bin=max_bin,
        )

        results.append(
            result
        )

    # =====================================================
    # Final comparison
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "oof_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # Existing LGB V5
    V5_OOF = 0.943743

    results_df[
        "delta_vs_v5"
    ] = (
        results_df["oof_auc"]
        - V5_OOF
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL V6 COMPARISON"
    )

    print(
        "=" * 120
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    best = results_df.iloc[0]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BEST V6"
    )

    print(
        "=" * 100
    )

    print(
        f"max_bin      : "
        f"{int(best['max_bin'])}"
    )

    print(
        f"OOF AUC      : "
        f"{best['oof_auc']:.6f}"
    )

    print(
        f"vs LGB V5    : "
        f"{best['delta_vs_v5']:+.6f}"
    )

    # Save summary
    summary_path = (
        OUTPUT_DIR
        / "v6_maxbin_comparison.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nSaved comparison:",
        summary_path,
    )


if __name__ == "__main__":
    main()