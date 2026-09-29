from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "train_engineered_v4.parquet"
)

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"
ID_COL = "id"

SEED = 42

# Same screening folds used before
SCREEN_FOLDS = [0, 2]

MAX_BINS = [
    255,
    511,
    1023,
    2047,
    4095,
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
# Target normalization
# =========================================================

def normalize_target(series):

    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int8")

    cleaned = (
        series
        .astype(str)
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

    mapped = cleaned.map(mapping)

    if mapped.isna().any():

        unknown = (
            series[
                mapped.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"Unknown target values: {unknown}"
        )

    return mapped.astype("int8")


# =========================================================
# Prepare categories
# =========================================================

def prepare_categories(df):

    df = df.copy()

    for col in CATEGORICAL_COLS:

        if col not in df.columns:
            continue

        df[col] = (
            df[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype("category")
        )

    return df


# =========================================================
# Evaluate one max_bin
# =========================================================

def evaluate_max_bin(
    train,
    features,
    max_bin,
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"MAX_BIN = {max_bin}"
    )

    print(
        "=" * 90
    )

    fold_array = (
        train[FOLD_COL]
        .to_numpy()
    )

    y = (
        train[TARGET]
        .to_numpy()
    )

    fold_scores = []
    fold_iterations = []
    fold_times = []

    for fold in SCREEN_FOLDS:

        start = time.time()

        train_idx = np.where(
            fold_array != fold
        )[0]

        valid_idx = np.where(
            fold_array == fold
        )[0]

        X_train = (
            train
            .iloc[train_idx][features]
        )

        y_train = y[
            train_idx
        ]

        X_valid = (
            train
            .iloc[valid_idx][features]
        )

        y_valid = y[
            valid_idx
        ]

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

            # ONLY major parameter changed
            max_bin=max_bin,

            random_state=SEED,
            n_jobs=-1,

            verbosity=-1,
        )

        active_categorical = [
            col
            for col in CATEGORICAL_COLS
            if col in features
        ]

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

        pred = (
            model.predict_proba(
                X_valid,
                num_iteration=(
                    model.best_iteration_
                ),
            )[:, 1]
        )

        score = roc_auc_score(
            y_valid,
            pred,
        )

        elapsed = (
            time.time()
            - start
        )

        fold_scores.append(
            score
        )

        fold_iterations.append(
            model.best_iteration_
        )

        fold_times.append(
            elapsed
        )

        print(
            f"Fold {fold}: "
            f"AUC={score:.6f} | "
            f"best_iter={model.best_iteration_} | "
            f"time={elapsed / 60:.1f} min"
        )

    mean_auc = float(
        np.mean(
            fold_scores
        )
    )

    avg_iter = float(
        np.mean(
            fold_iterations
        )
    )

    total_time = float(
        np.sum(
            fold_times
        )
    )

    print(
        f"\nMean AUC : {mean_auc:.6f}"
    )

    print(
        f"Avg iter : {avg_iter:.1f}"
    )

    return {
        "max_bin": max_bin,

        "fold_0_auc":
            fold_scores[0],

        "fold_2_auc":
            fold_scores[1],

        "mean_auc":
            mean_auc,

        "avg_best_iteration":
            avg_iter,

        "runtime_minutes":
            total_time / 60,
    }


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=" * 100
    )

    print(
        "LGB V5 - MAX_BIN SCREENING"
    )

    print(
        "=" * 100
    )

    print(
        "Screening folds:",
        SCREEN_FOLDS
    )

    print(
        "max_bin candidates:",
        MAX_BINS
    )

    # =====================================================
    # Load
    # =====================================================

    train = pd.read_parquet(
        TRAIN_PATH
    )

    train[TARGET] = (
        normalize_target(
            train[TARGET]
        )
    )

    train = prepare_categories(
        train
    )

    print(
        "\nTrain shape:",
        train.shape
    )

    # Preserve exactly the existing engineered features
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
        "Feature count:",
        len(features)
    )

    if len(features) != 56:

        print(
            "\nWARNING:"
            f" expected 56 V4 features, "
            f"found {len(features)}."
        )

    # =====================================================
    # Run screening
    # =====================================================

    results = []

    total_start = time.time()

    for max_bin in MAX_BINS:

        result = evaluate_max_bin(
            train=train,
            features=features,
            max_bin=max_bin,
        )

        results.append(
            result
        )

    # =====================================================
    # Compare
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    baseline_row = (
        results_df[
            results_df["max_bin"] == 255
        ]
        .iloc[0]
    )

    baseline_auc = float(
        baseline_row["mean_auc"]
    )

    baseline_fold0 = float(
        baseline_row["fold_0_auc"]
    )

    baseline_fold2 = float(
        baseline_row["fold_2_auc"]
    )

    results_df[
        "delta_vs_255"
    ] = (
        results_df["mean_auc"]
        - baseline_auc
    )

    results_df[
        "fold0_delta"
    ] = (
        results_df["fold_0_auc"]
        - baseline_fold0
    )

    results_df[
        "fold2_delta"
    ] = (
        results_df["fold_2_auc"]
        - baseline_fold2
    )

    results_df[
        "positive_folds"
    ] = (
        (
            results_df[
                [
                    "fold0_delta",
                    "fold2_delta",
                ]
            ]
            > 0
        )
        .sum(axis=1)
    )

    results_df = (
        results_df
        .sort_values(
            "mean_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # Ranking
    # =====================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "MAX_BIN SCREENING RESULTS"
    )

    print(
        "=" * 120
    )

    print(
        results_df[
            [
                "max_bin",
                "fold_0_auc",
                "fold0_delta",
                "fold_2_auc",
                "fold2_delta",
                "mean_auc",
                "delta_vs_255",
                "positive_folds",
                "avg_best_iteration",
                "runtime_minutes",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # =====================================================
    # Best
    # =====================================================

    best = (
        results_df.iloc[0]
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BEST MAX_BIN"
    )

    print(
        "=" * 100
    )

    print(
        f"max_bin       : "
        f"{int(best['max_bin'])}"
    )

    print(
        f"Mean AUC      : "
        f"{best['mean_auc']:.6f}"
    )

    print(
        f"Delta vs 255  : "
        f"{best['delta_vs_255']:+.6f}"
    )

    print(
        f"Fold 0 delta  : "
        f"{best['fold0_delta']:+.6f}"
    )

    print(
        f"Fold 2 delta  : "
        f"{best['fold2_delta']:+.6f}"
    )

    print(
        f"Positive folds: "
        f"{int(best['positive_folds'])}/2"
    )

    # =====================================================
    # Save
    # =====================================================

    output = (
        ROOT
        / "logs"
        / "lgb_maxbin_screening.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output,
        index=False,
    )

    total_elapsed = (
        time.time()
        - total_start
    )

    print(
        "\nSaved:",
        output
    )

    print(
        f"Total runtime: "
        f"{total_elapsed / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
