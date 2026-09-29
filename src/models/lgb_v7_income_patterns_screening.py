from pathlib import Path
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "train_engineered_v4.parquet"
)

LOG_PATH = (
    ROOT
    / "logs"
    / "lgb_v7_income_patterns_screening.csv"
)

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"
ID_COL = "id"

SCREEN_FOLDS = [0, 2]

SEED = 42

# Current best V6
MAX_BIN = 8191


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


# ============================================================
# TARGET
# ============================================================

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
            series[result.isna()]
            .unique()
        )

        raise ValueError(
            f"Unknown target values: {unknown}"
        )

    return result.astype("int8")


# ============================================================
# CATEGORY PREPARATION
# ============================================================

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


# ============================================================
# INCOME PATTERNS
# ============================================================

def add_income_patterns(df):

    df = df.copy()

    income = pd.to_numeric(
        df["Annual_Income_USD"],
        errors="coerce",
    )

    # Income should already be complete in official train,
    # but keep this safe.
    income = income.fillna(
        income.median()
    )

    # Work with rounded integer income
    inc = (
        np.rint(income)
        .astype(np.int64)
    )

    # --------------------------------------------------------
    # 1. Modulo patterns
    # --------------------------------------------------------

    df["Income_Mod500"] = (
        inc % 500
    ).astype("int16")

    df["Income_Mod5"] = (
        inc % 5
    ).astype("int8")

    df["Income_Mod20"] = (
        inc % 20
    ).astype("int8")

    df["Income_Mod50"] = (
        inc % 50
    ).astype("int8")

    df["Income_Mod1000"] = (
        inc % 1000
    ).astype("int16")

    df["Income_Mod25"] = (
        inc % 25
    ).astype("int8")

    df["Income_Mod200"] = (
        inc % 200
    ).astype("int16")

    # --------------------------------------------------------
    # 2. Distance to round values
    # --------------------------------------------------------

    mod500 = inc % 500

    df["Income_Dist500"] = np.minimum(
        mod500,
        500 - mod500,
    ).astype("int16")

    mod1000 = inc % 1000

    df["Income_Dist1000"] = np.minimum(
        mod1000,
        1000 - mod1000,
    ).astype("int16")

    mod100 = inc % 100

    df["Income_Dist100"] = np.minimum(
        mod100,
        100 - mod100,
    ).astype("int8")

    # --------------------------------------------------------
    # 3. Digit interactions
    # --------------------------------------------------------

    ones = inc % 10

    tens = (
        inc // 10
    ) % 10

    hundreds = (
        inc // 100
    ) % 10

    thousands = (
        inc // 1000
    ) % 10

    df["Income_TensHundreds"] = (
        tens * 10
        + hundreds
    ).astype("int8")

    df["Income_HundredsThousands"] = (
        hundreds * 10
        + thousands
    ).astype("int8")

    df["Income_OnesTens"] = (
        ones * 10
        + tens
    ).astype("int8")

    # --------------------------------------------------------
    # 4. Digit sums
    # --------------------------------------------------------

    temp = inc.copy()

    digit_sum = np.zeros(
        len(df),
        dtype=np.int16,
    )

    while np.any(temp > 0):

        digit_sum += (
            temp % 10
        ).astype(np.int16)

        temp //= 10

    df["Income_DigitSum"] = (
        digit_sum
    )

    last3 = inc % 1000

    df["Income_Last3_DigitSum"] = (
        (last3 % 10)
        + ((last3 // 10) % 10)
        + ((last3 // 100) % 10)
    ).astype("int8")

    return df


# ============================================================
# PATTERN GROUPS
# ============================================================

TOP3 = [
    "Income_Mod500",
    "Income_Mod5",
    "Income_Dist500",
]

TOP6 = TOP3 + [
    "Income_Dist1000",
    "Income_Mod20",
    "Income_TensHundreds",
]

TOP10 = TOP6 + [
    "Income_HundredsThousands",
    "Income_Mod50",
    "Income_Mod1000",
    "Income_DigitSum",
]

ALL15 = TOP10 + [
    "Income_Mod25",
    "Income_Dist100",
    "Income_Last3_DigitSum",
    "Income_OnesTens",
    "Income_Mod200",
]


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    train,
    base_features,
    extra_features,
    config_name,
):

    features = (
        base_features
        + extra_features
    )

    print("\n" + "=" * 100)
    print(config_name)
    print("=" * 100)

    print(
        f"Feature count: {len(features)}"
    )

    if extra_features:

        print(
            "Extra:",
            ", ".join(extra_features)
        )

    else:

        print(
            "Extra: NONE"
        )

    folds = (
        train[FOLD_COL]
        .to_numpy()
    )

    y = (
        train[TARGET]
        .to_numpy()
    )

    categorical = [
        col
        for col in CATEGORICAL_COLS
        if col in features
    ]

    scores = []
    best_iterations = []
    runtimes = []

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

            # V6 best
            max_bin=MAX_BIN,

            # Keep default V6 bin sampling
            bin_construct_sample_cnt=200_000,

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

            categorical_feature=categorical,

            callbacks=[
                lgb.early_stopping(
                    150,
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

        auc = roc_auc_score(
            y_valid,
            pred,
        )

        elapsed = (
            time.time()
            - start
        ) / 60

        scores.append(
            auc
        )

        best_iterations.append(
            model.best_iteration_
        )

        runtimes.append(
            elapsed
        )

        print(
            f"Fold {fold}: "
            f"AUC={auc:.6f} | "
            f"best_iter={model.best_iteration_} | "
            f"time={elapsed:.2f} min"
        )

        del model
        del X_train
        del X_valid

        gc.collect()

    mean_auc = float(
        np.mean(scores)
    )

    print(
        f"\nMean AUC: {mean_auc:.6f}"
    )

    return {
        "config": config_name,
        "feature_count": len(features),
        "fold_0_auc": scores[0],
        "fold_2_auc": scores[1],
        "mean_auc": mean_auc,
        "avg_best_iteration": float(
            np.mean(best_iterations)
        ),
        "runtime_minutes": float(
            np.sum(runtimes)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("LGB V7 - INCOME PATTERN SCREENING")
    print("=" * 100)

    print(
        f"max_bin = {MAX_BIN}"
    )

    print(
        f"folds = {SCREEN_FOLDS}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train = pd.read_parquet(
        TRAIN_PATH
    )

    train[TARGET] = (
        normalize_target(
            train[TARGET]
        )
    )

    # Capture original V4 feature set BEFORE adding patterns
    base_features = [
        col
        for col in train.columns
        if col not in [
            TARGET,
            FOLD_COL,
            ID_COL,
        ]
    ]

    print(
        "\nOriginal shape:",
        train.shape
    )

    print(
        "Base feature count:",
        len(base_features)
    )

    if len(base_features) != 56:

        print(
            f"WARNING: expected 56 base features, "
            f"found {len(base_features)}"
        )

    # --------------------------------------------------------
    # Add candidate features
    # --------------------------------------------------------

    train = add_income_patterns(
        train
    )

    train = prepare_categories(
        train
    )

    print(
        "After patterns:",
        train.shape
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    for feature in ALL15:

        if feature not in train.columns:

            raise ValueError(
                f"Missing generated feature: {feature}"
            )

    # --------------------------------------------------------
    # Configurations
    # --------------------------------------------------------

    configs = [
        (
            "BASE_V6",
            [],
        ),
        (
            "TOP3",
            TOP3,
        ),
        (
            "TOP6",
            TOP6,
        ),
        (
            "TOP10",
            TOP10,
        ),
        (
            "ALL15",
            ALL15,
        ),
    ]

    results = []

    total_start = time.time()

    for name, extra in configs:

        result = evaluate(
            train=train,
            base_features=base_features,
            extra_features=extra,
            config_name=name,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    baseline_row = (
        results_df[
            results_df["config"]
            == "BASE_V6"
        ]
        .iloc[0]
    )

    baseline_mean = (
        baseline_row["mean_auc"]
    )

    baseline_f0 = (
        baseline_row["fold_0_auc"]
    )

    baseline_f2 = (
        baseline_row["fold_2_auc"]
    )

    results_df[
        "delta_vs_base"
    ] = (
        results_df["mean_auc"]
        - baseline_mean
    )

    results_df[
        "fold0_delta"
    ] = (
        results_df["fold_0_auc"]
        - baseline_f0
    )

    results_df[
        "fold2_delta"
    ] = (
        results_df["fold_2_auc"]
        - baseline_f2
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

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 130
    )

    print(
        "V7 INCOME PATTERN SCREENING RESULTS"
    )

    print(
        "=" * 130
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
        f"Config         : "
        f"{best['config']}"
    )

    print(
        f"Features       : "
        f"{int(best['feature_count'])}"
    )

    print(
        f"Mean AUC       : "
        f"{best['mean_auc']:.6f}"
    )

    print(
        f"Delta vs V6    : "
        f"{best['delta_vs_base']:+.6f}"
    )

    print(
        f"Fold 0 delta   : "
        f"{best['fold0_delta']:+.6f}"
    )

    print(
        f"Fold 2 delta   : "
        f"{best['fold2_delta']:+.6f}"
    )

    print(
        f"Positive folds : "
        f"{int(best['positive_folds'])}/2"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        LOG_PATH,
        index=False,
    )

    total_runtime = (
        time.time()
        - total_start
    ) / 60

    print(
        "\nSaved:",
        LOG_PATH
    )

    print(
        f"Total runtime: "
        f"{total_runtime:.2f} min"
    )


if __name__ == "__main__":
    main()