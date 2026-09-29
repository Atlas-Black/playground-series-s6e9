from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.build_features_v3 import V3_FEATURES
from src.features.build_features_v4 import V4_NEW_FEATURES


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
SEED = 42

TUNE_FOLDS = [0, 2]


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
# Prepare categorical
# =========================================================

def prepare_categorical(df):

    df = df.copy()

    for col in CATEGORICAL_COLS:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .astype("category")
            )

    return df


# =========================================================
# Evaluate
# =========================================================

def evaluate(
    train,
    features,
    experiment_name,
):

    y = train[TARGET]

    folds = (
        train["fold"]
        .to_numpy()
    )

    scores = []
    iterations = []

    print("\n" + "=" * 75)
    print(experiment_name)
    print("=" * 75)

    print(
        f"Features: {len(features)}"
    )

    for fold in TUNE_FOLDS:

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

        y_train = (
            y.iloc[train_idx]
        )

        X_valid = (
            train
            .iloc[valid_idx][features]
        )

        y_valid = (
            y.iloc[valid_idx]
        )

        active_cat_cols = [
            col
            for col in CATEGORICAL_COLS
            if col in features
        ]

        # ================================================
        # Exactly the LGB V4 configuration
        # ================================================

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

            categorical_feature=active_cat_cols,

            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=150,
                    verbose=False,
                )
            ],
        )

        pred = model.predict_proba(
            X_valid,
            num_iteration=model.best_iteration_,
        )[:, 1]

        auc = roc_auc_score(
            y_valid,
            pred,
        )

        scores.append(
            auc
        )

        iterations.append(
            model.best_iteration_
        )

        print(
            f"Fold {fold}: "
            f"AUC={auc:.6f} | "
            f"best_iter={model.best_iteration_}"
        )

    mean_auc = float(
        np.mean(scores)
    )

    print(
        f"Mean AUC: {mean_auc:.6f}"
    )

    return {
        "experiment": experiment_name,
        "n_features": len(features),

        "fold_0_auc": scores[0],
        "fold_2_auc": scores[1],

        "mean_auc": mean_auc,

        "avg_best_iteration": float(
            np.mean(iterations)
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 75)
    print("Income Structure Feature Screening")
    print("=" * 75)

    print(
        "Tune folds:",
        TUNE_FOLDS
    )

    print(
        "New candidates:",
        len(V4_NEW_FEATURES)
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    train = pd.read_parquet(
        TRAIN_PATH
    )

    print(
        "\nTrain shape:",
        train.shape
    )

    # -----------------------------------------------------
    # Checks
    # -----------------------------------------------------

    required = (
        V3_FEATURES
        + V4_NEW_FEATURES
        + [
            TARGET,
            "fold",
        ]
    )

    missing = [
        col
        for col in required
        if col not in train.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    train = prepare_categorical(
        train
    )

    results = []

    start_time = time.time()

    # =====================================================
    # 1. Baseline
    # =====================================================

    baseline = evaluate(
        train,
        V3_FEATURES,
        "BASELINE_V3",
    )

    results.append(
        baseline
    )

    baseline_auc = (
        baseline["mean_auc"]
    )

    # =====================================================
    # 2. Add ONE feature at a time
    # =====================================================

    for new_feature in V4_NEW_FEATURES:

        features = (
            V3_FEATURES
            + [new_feature]
        )

        result = evaluate(
            train,
            features,
            f"ADD_{new_feature}",
        )

        result["new_feature"] = (
            new_feature
        )

        result["delta_vs_baseline"] = (
            result["mean_auc"]
            - baseline_auc
        )

        results.append(
            result
        )

    # =====================================================
    # 3. Add ALL 8
    # =====================================================

    all_v4_features = (
        V3_FEATURES
        + V4_NEW_FEATURES
    )

    result = evaluate(
        train,
        all_v4_features,
        "ADD_ALL_8",
    )

    result["new_feature"] = (
        "ALL_8"
    )

    result["delta_vs_baseline"] = (
        result["mean_auc"]
        - baseline_auc
    )

    results.append(
        result
    )

    # =====================================================
    # Results
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.loc[
        results_df["experiment"]
        == "BASELINE_V3",
        "delta_vs_baseline",
    ] = 0.0

    results_df = (
        results_df
        .sort_values(
            "delta_vs_baseline",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "INCOME STRUCTURE SCREENING RANKING"
    )

    print(
        "=" * 90
    )

    print(
        results_df[
            [
                "experiment",
                "n_features",
                "fold_0_auc",
                "fold_2_auc",
                "mean_auc",
                "delta_vs_baseline",
                "avg_best_iteration",
            ]
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # Candidate interpretation
    # =====================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CANDIDATE SUMMARY"
    )

    print(
        "=" * 90
    )

    candidates = results_df[
        ~results_df["experiment"].isin(
            [
                "BASELINE_V3",
                "ADD_ALL_8",
            ]
        )
    ].copy()

    for _, row in candidates.iterrows():

        delta = row[
            "delta_vs_baseline"
        ]

        feature = row[
            "new_feature"
        ]

        if delta >= 0.00010:

            verdict = "PROMISING"

        elif delta > 0:

            verdict = "small positive"

        elif delta > -0.00005:

            verdict = "neutral"

        else:

            verdict = "negative"

        print(
            f"{feature:<28} "
            f"{delta:+.6f} | "
            f"{verdict}"
        )

    # =====================================================
    # Positive candidates
    # =====================================================

    positive = candidates[
        candidates[
            "delta_vs_baseline"
        ] > 0
    ].sort_values(
        "delta_vs_baseline",
        ascending=False,
    )

    print(
        "\nPositive candidates:"
    )

    if len(positive) == 0:

        print(
            "None"
        )

    else:

        for _, row in positive.iterrows():

            print(
                f" - {row['new_feature']}: "
                f"{row['delta_vs_baseline']:+.6f}"
            )

    # =====================================================
    # Save
    # =====================================================

    output_path = (
        ROOT
        / "logs"
        / "income_structure_screening.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        f"\nRuntime: "
        f"{elapsed / 60:.1f} minutes"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()