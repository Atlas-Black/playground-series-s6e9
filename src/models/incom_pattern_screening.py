from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.build_features_v4 import V4_FEATURES


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

# screening folds
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
# Candidate features
# =========================================================

PATTERN_FEATURES = [

    # ---------- modulo ----------
    "Income_Mod5",
    "Income_Mod20",
    "Income_Mod25",
    "Income_Mod50",
    "Income_Mod200",
    "Income_Mod500",
    "Income_Mod1000",

    # ---------- distance to round number ----------
    "Income_Dist100",
    "Income_Dist500",
    "Income_Dist1000",

    # ---------- digit combinations ----------
    "Income_OnesTens",
    "Income_TensHundreds",
    "Income_HundredsThousands",

    # ---------- digit statistics ----------
    "Income_DigitSum",
    "Income_Last3_DigitSum",
]


# =========================================================
# Add pattern features
# =========================================================

def add_pattern_features(df):

    df = df.copy()

    income = (
        df["Annual_Income_USD"]
        .fillna(0)
        .round()
        .astype("int64")
        .abs()
    )

    # =====================================================
    # Modulo patterns
    # =====================================================

    df["Income_Mod5"] = (
        income % 5
    ).astype("int16")

    df["Income_Mod20"] = (
        income % 20
    ).astype("int16")

    df["Income_Mod25"] = (
        income % 25
    ).astype("int16")

    df["Income_Mod50"] = (
        income % 50
    ).astype("int16")

    df["Income_Mod200"] = (
        income % 200
    ).astype("int16")

    df["Income_Mod500"] = (
        income % 500
    ).astype("int16")

    df["Income_Mod1000"] = (
        income % 1000
    ).astype("int16")

    # =====================================================
    # Distance to nearest round number
    # =====================================================

    mod100 = income % 100

    df["Income_Dist100"] = np.minimum(
        mod100,
        100 - mod100,
    ).astype("int16")


    mod500 = income % 500

    df["Income_Dist500"] = np.minimum(
        mod500,
        500 - mod500,
    ).astype("int16")


    mod1000 = income % 1000

    df["Income_Dist1000"] = np.minimum(
        mod1000,
        1000 - mod1000,
    ).astype("int16")

    # =====================================================
    # Individual digits
    # =====================================================

    ones = (
        income % 10
    )

    tens = (
        (income // 10) % 10
    )

    hundreds = (
        (income // 100) % 10
    )

    thousands = (
        (income // 1000) % 10
    )

    ten_thousands = (
        (income // 10000) % 10
    )

    # =====================================================
    # Adjacent digit combinations
    # =====================================================

    # e.g. 78436 -> 36
    df["Income_OnesTens"] = (
        tens * 10
        + ones
    ).astype("int16")

    # e.g. 78436 -> 43
    df["Income_TensHundreds"] = (
        hundreds * 10
        + tens
    ).astype("int16")

    # e.g. 78436 -> 84
    df["Income_HundredsThousands"] = (
        thousands * 10
        + hundreds
    ).astype("int16")

    # =====================================================
    # Digit statistics
    # =====================================================

    df["Income_DigitSum"] = (
        ones
        + tens
        + hundreds
        + thousands
        + ten_thousands
    ).astype("int16")

    df["Income_Last3_DigitSum"] = (
        ones
        + tens
        + hundreds
    ).astype("int16")

    return df


# =========================================================
# Categorical preparation
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

    print(
        "\n"
        + "=" * 78
    )

    print(
        experiment_name
    )

    print(
        "=" * 78
    )

    print(
        "Features:",
        len(features)
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

        # Same parameters as LGB V5
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

        fold_auc = roc_auc_score(
            y_valid,
            pred,
        )

        scores.append(
            fold_auc
        )

        iterations.append(
            model.best_iteration_
        )

        print(
            f"Fold {fold}: "
            f"AUC={fold_auc:.6f} | "
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

    print(
        "=" * 90
    )

    print(
        "INCOME PATTERN SCREENING"
    )

    print(
        "=" * 90
    )

    print(
        "Baseline: V4 features / LGB V5"
    )

    print(
        "Tune folds:",
        TUNE_FOLDS
    )

    print(
        "Candidates:",
        len(PATTERN_FEATURES)
    )

    # =====================================================
    # Load
    # =====================================================

    train = pd.read_parquet(
        TRAIN_PATH
    )

    print(
        "\nOriginal shape:",
        train.shape
    )

    # =====================================================
    # Generate candidates
    # =====================================================

    train = add_pattern_features(
        train
    )

    print(
        "Shape after candidates:",
        train.shape
    )

    # =====================================================
    # Checks
    # =====================================================

    required = (
        V4_FEATURES
        + PATTERN_FEATURES
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

    # =====================================================
    # Baseline
    # =====================================================

    results = []

    start_time = time.time()

    baseline = evaluate(
        train,
        V4_FEATURES,
        "BASELINE_V4",
    )

    baseline_auc = (
        baseline["mean_auc"]
    )

    baseline[
        "delta_vs_baseline"
    ] = 0.0

    results.append(
        baseline
    )

    # =====================================================
    # Individual screening
    # =====================================================

    for feature in PATTERN_FEATURES:

        result = evaluate(

            train,

            V4_FEATURES
            + [feature],

            f"ADD_{feature}",
        )

        result[
            "delta_vs_baseline"
        ] = (
            result["mean_auc"]
            - baseline_auc
        )

        results.append(
            result
        )

    # =====================================================
    # All candidate features
    # =====================================================

    result = evaluate(

        train,

        V4_FEATURES
        + PATTERN_FEATURES,

        "ADD_ALL_PATTERNS",
    )

    result[
        "delta_vs_baseline"
    ] = (
        result["mean_auc"]
        - baseline_auc
    )

    results.append(
        result
    )

    # =====================================================
    # Ranking
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

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
        + "=" * 100
    )

    print(
        "INCOME PATTERN RANKING"
    )

    print(
        "=" * 100
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
    # Interpretation
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PATTERN SUMMARY"
    )

    print(
        "=" * 100
    )

    individual = results_df[
        ~results_df[
            "experiment"
        ].isin(
            [
                "BASELINE_V4",
                "ADD_ALL_PATTERNS",
            ]
        )
    ]

    for _, row in individual.iterrows():

        delta = row[
            "delta_vs_baseline"
        ]

        name = (
            row["experiment"]
            .replace(
                "ADD_",
                "",
            )
        )

        if delta >= 0.00010:

            verdict = "PROMISING"

        elif delta >= 0.00003:

            verdict = "small positive"

        elif delta > -0.00003:

            verdict = "neutral"

        else:

            verdict = "negative"

        print(
            f"{name:<32} "
            f"{delta:+.6f} | "
            f"{verdict}"
        )

    # =====================================================
    # Stable positive candidates
    # =====================================================

    positive = individual[
        individual[
            "delta_vs_baseline"
        ] > 0
    ].copy()

    positive[
        "fold0_delta"
    ] = (
        positive["fold_0_auc"]
        - baseline["fold_0_auc"]
    )

    positive[
        "fold2_delta"
    ] = (
        positive["fold_2_auc"]
        - baseline["fold_2_auc"]
    )

    stable = positive[
        (
            positive["fold0_delta"]
            > 0
        )
        &
        (
            positive["fold2_delta"]
            > 0
        )
    ].sort_values(
        "delta_vs_baseline",
        ascending=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "POSITIVE ON BOTH FOLD 0 AND FOLD 2"
    )

    print(
        "=" * 100
    )

    if len(stable) == 0:

        print(
            "None"
        )

    else:

        print(
            stable[
                [
                    "experiment",
                    "fold0_delta",
                    "fold2_delta",
                    "delta_vs_baseline",
                ]
            ].to_string(
                index=False
            )
        )

    # =====================================================
    # Save
    # =====================================================

    output_path = (
        ROOT
        / "logs"
        / "income_pattern_screening.csv"
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

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()