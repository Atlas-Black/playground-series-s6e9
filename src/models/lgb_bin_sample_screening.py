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

# Current best max_bin
MAX_BIN = 8191

# Screening folds
SCREEN_FOLDS = [0, 2]

BIN_SAMPLE_COUNTS = [
    200_000,
    400_000,
    600_000,
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

    result = cleaned.map(mapping)

    if result.isna().any():

        unknown = (
            series[
                result.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"Unknown target values: {unknown}"
        )

    return result.astype("int8")


# =========================================================
# Categories
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
# Evaluate
# =========================================================

def evaluate_config(
    train,
    features,
    bin_sample_count,
):

    print("\n" + "=" * 100)

    print(
        f"BIN_CONSTRUCT_SAMPLE_CNT = "
        f"{bin_sample_count:,}"
    )

    print("=" * 100)

    folds = (
        train[FOLD_COL]
        .to_numpy()
    )

    y = (
        train[TARGET]
        .to_numpy()
    )

    fold_scores = []
    best_iterations = []
    runtimes = []

    active_categorical = [
        col
        for col in CATEGORICAL_COLS
        if col in features
    ]

    for fold in SCREEN_FOLDS:

        start = time.time()

        train_idx = np.where(
            folds != fold
        )[0]

        valid_idx = np.where(
            folds == fold
        )[0]

        X_train = (
            train
            .iloc[train_idx][features]
        )

        X_valid = (
            train
            .iloc[valid_idx][features]
        )

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

            # Current best
            max_bin=MAX_BIN,

            # New experiment
            bin_construct_sample_cnt=(
                bin_sample_count
            ),

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

        best_iterations.append(
            model.best_iteration_
        )

        runtimes.append(
            elapsed
        )

        print(
            f"Fold {fold}: "
            f"AUC={score:.6f} | "
            f"best_iter="
            f"{model.best_iteration_} | "
            f"time={elapsed / 60:.2f} min"
        )

    # =====================================================
    # Summary
    # =====================================================

    mean_auc = float(
        np.mean(fold_scores)
    )

    avg_iter = float(
        np.mean(best_iterations)
    )

    runtime = float(
        np.sum(runtimes) / 60
    )

    print(
        f"\nMean AUC : "
        f"{mean_auc:.6f}"
    )

    print(
        f"Avg iter : "
        f"{avg_iter:.1f}"
    )

    return {
        "bin_sample_count":
            bin_sample_count,

        "fold_0_auc":
            fold_scores[0],

        "fold_2_auc":
            fold_scores[1],

        "mean_auc":
            mean_auc,

        "avg_best_iteration":
            avg_iter,

        "runtime_minutes":
            runtime,
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 100)
    print("LGB V6 - BIN CONSTRUCT SAMPLE SCREENING")
    print("=" * 100)

    print(
        f"Fixed max_bin: {MAX_BIN}"
    )

    print(
        "Screening folds:",
        SCREEN_FOLDS
    )

    print(
        "Candidates:",
        BIN_SAMPLE_COUNTS
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
        "\nTrain shape:",
        train.shape
    )

    print(
        "Feature count:",
        len(features)
    )

    if len(features) != 56:

        print(
            f"WARNING: expected 56 "
            f"features, found "
            f"{len(features)}"
        )

    # =====================================================
    # Run
    # =====================================================

    results = []

    total_start = time.time()

    for count in BIN_SAMPLE_COUNTS:

        result = evaluate_config(
            train=train,
            features=features,
            bin_sample_count=count,
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

    baseline = float(
        results_df.loc[
            results_df[
                "bin_sample_count"
            ] == 200_000,
            "mean_auc",
        ].iloc[0]
    )

    baseline_fold0 = float(
        results_df.loc[
            results_df[
                "bin_sample_count"
            ] == 200_000,
            "fold_0_auc",
        ].iloc[0]
    )

    baseline_fold2 = float(
        results_df.loc[
            results_df[
                "bin_sample_count"
            ] == 200_000,
            "fold_2_auc",
        ].iloc[0]
    )

    results_df[
        "delta_vs_200k"
    ] = (
        results_df["mean_auc"]
        - baseline
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
            ] > 0
        )
        .sum(axis=1)
    )

    results_df = (
        results_df
        .sort_values(
            "mean_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # Results
    # =====================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "BIN SAMPLE SCREENING RESULTS"
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
        "BEST CONFIGURATION"
    )

    print(
        "=" * 100
    )

    print(
        f"bin_construct_sample_cnt : "
        f"{int(best['bin_sample_count']):,}"
    )

    print(
        f"Mean AUC                 : "
        f"{best['mean_auc']:.6f}"
    )

    print(
        f"Delta vs 200k            : "
        f"{best['delta_vs_200k']:+.6f}"
    )

    print(
        f"Fold 0 delta             : "
        f"{best['fold0_delta']:+.6f}"
    )

    print(
        f"Fold 2 delta             : "
        f"{best['fold2_delta']:+.6f}"
    )

    print(
        f"Positive folds           : "
        f"{int(best['positive_folds'])}/2"
    )

    # =====================================================
    # Save
    # =====================================================

    output = (
        ROOT
        / "logs"
        / "lgb_bin_sample_screening.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output,
        index=False,
    )

    elapsed = (
        time.time()
        - total_start
    ) / 60

    print(
        "\nSaved:",
        output
    )

    print(
        f"Total runtime: "
        f"{elapsed:.2f} min"
    )


if __name__ == "__main__":
    main()