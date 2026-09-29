from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_PATH = (
    ROOT / "data" / "processed" / "train_engineered_v4.parquet"
)

ORIGINAL_PATH = (
    ROOT
    / "data"
    / "external"
    / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
)

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"

SCREEN_FOLDS = [0, 2]

# Original data weights to test
EXTERNAL_WEIGHTS = [
    0.0,
    1.0,
    2.0,
    5.0,
]


# =========================================================
# Base raw features
# =========================================================

RAW_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
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
            series[result.isna()]
            .unique()
        )

        raise ValueError(
            f"Unknown target values: {unknown}"
        )

    return result.astype("int8")


# =========================================================
# Helpers
# =========================================================

def safe_divide(a, b):

    a = pd.to_numeric(
        a,
        errors="coerce",
    )

    b = pd.to_numeric(
        b,
        errors="coerce",
    )

    result = a / b.replace(
        0,
        np.nan,
    )

    return result


def yes_no_flag(series):

    s = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return s.map({
        "yes": 1.0,
        "no": 0.0,
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
    })


def anxiety_numeric(series):

    # Try numeric first
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().mean() > 0.95:
        return numeric

    s = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "low": 1.0,
        "medium": 2.0,
        "moderate": 2.0,
        "high": 3.0,
    }

    return s.map(mapping)


# =========================================================
# V4 feature engineering for ORIGINAL
# =========================================================

def build_v4_features(df):

    df = df.copy()

    # -----------------------------------------------------
    # Basic engineered features
    # -----------------------------------------------------

    df["Total_Charging_Stations"] = (
        df["Charging_Stations_Near_Home"]
        + df["Charging_Stations_Near_Work"]
    )

    df["Charging_Per_Commute"] = safe_divide(
        df["Total_Charging_Stations"],
        df["Daily_Commute_km"] + 1,
    )

    df["Has_Home_Charging"] = yes_no_flag(
        df["Home_Charging_Possible"]
    )

    df["Income_Per_Age"] = safe_divide(
        df["Annual_Income_USD"],
        df["Age"],
    )

    df["Range_Anxiety_Num"] = anxiety_numeric(
        df["Range_Anxiety_Level"]
    )

    df["Anxiety_Commute_Ratio"] = safe_divide(
        df["Range_Anxiety_Num"],
        df["Daily_Commute_km"] + 1,
    )

    df["Has_Multiple_Cars"] = (
        pd.to_numeric(
            df["Number_of_Cars_Owned"],
            errors="coerce",
        ) >= 2
    ).astype("int8")

    df["Subsidy_Flag"] = yes_no_flag(
        df["Subsidy_Available"]
    )

    df["Eco_Subsidy_Score"] = (
        df["Environmental_Concern_Level"]
        * df["Subsidy_Flag"]
    )

    # -----------------------------------------------------
    # V2 interactions
    # -----------------------------------------------------

    df["Income_Per_Car"] = safe_divide(
        df["Annual_Income_USD"],
        pd.to_numeric(
            df["Number_of_Cars_Owned"],
            errors="coerce",
        ) + 1,
    )

    df["Income_Subsidy_Interaction"] = (
        df["Annual_Income_USD"]
        * df["Subsidy_Flag"]
    )

    df["HomeCharging_Commute"] = (
        df["Has_Home_Charging"]
        * df["Daily_Commute_km"]
    )

    df["Charging_Anxiety_Interaction"] = (
        df["Total_Charging_Stations"]
        * df["Range_Anxiety_Num"]
    )

    df["Commute_Charging_Pressure"] = safe_divide(
        df["Daily_Commute_km"],
        df["Total_Charging_Stations"] + 1,
    )

    df["Eco_Income_Interaction"] = (
        df["Environmental_Concern_Level"]
        * df["Annual_Income_USD"]
    )

    df["Eco_HomeCharging_Score"] = (
        df["Environmental_Concern_Level"]
        * df["Has_Home_Charging"]
    )

    # -----------------------------------------------------
    # Income digits
    # -----------------------------------------------------

    income = pd.to_numeric(
        df["Annual_Income_USD"],
        errors="coerce",
    )

    # Nullable integer preserves missing values
    income_int = (
        income
        .round()
        .abs()
        .astype("Int64")
    )

    df["Income_Ones"] = (
        income_int % 10
    ).astype("Float64")

    df["Income_Tens"] = (
        (income_int // 10) % 10
    ).astype("Float64")

    df["Income_Hundreds"] = (
        (income_int // 100) % 10
    ).astype("Float64")

    df["Income_Thousands"] = (
        (income_int // 1000) % 10
    ).astype("Float64")

    df["Income_TenThousands"] = (
        (income_int // 10000) % 10
    ).astype("Float64")

    df["Income_Log"] = np.log1p(
        income
    )

    # -----------------------------------------------------
    # Income bins
    # -----------------------------------------------------

    df["Income_1k_Bin"] = (
        np.floor(
            income / 1000
        )
    )

    df["Income_5k_Bin"] = (
        np.floor(
            income / 5000
        )
    )

    df["Income_10k_Bin"] = (
        np.floor(
            income / 10000
        )
    )

    # -----------------------------------------------------
    # Frequency features
    #
    # IMPORTANT:
    # These are temporary here.
    # We will overwrite them fold-safely below using
    # official training data.
    # -----------------------------------------------------

    freq_source_cols = [
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

    for col in freq_source_cols:

        freq = (
            df[col]
            .value_counts(
                normalize=True,
                dropna=False,
            )
        )

        df[f"{col}_Freq"] = (
            df[col]
            .map(freq)
            .astype(float)
        )

    # -----------------------------------------------------
    # V4 income structure
    # -----------------------------------------------------

    df["Income_Last2"] = (
        income_int % 100
    ).astype("Float64")

    df["Income_Last3"] = (
        income_int % 1000
    ).astype("Float64")

    df["Income_Last4"] = (
        income_int % 10000
    ).astype("Float64")

    # First two digits
    def first_two(x):

        if pd.isna(x):
            return np.nan

        x = int(abs(x))

        if x < 10:
            return float(x)

        while x >= 100:
            x //= 10

        return float(x)

    df["Income_First2"] = (
        income_int
        .apply(first_two)
        .astype(float)
    )

    df["Income_100_Bin"] = np.floor(
        income / 100
    )

    df["Income_500_Bin"] = np.floor(
        income / 500
    )

    df["Income_2k_Bin"] = np.floor(
        income / 2000
    )

    df["Income_Hundreds_Tens"] = (
        df["Income_Hundreds"] * 10
        + df["Income_Tens"]
    )

    return df


# =========================================================
# Frequency encoding
# =========================================================

FREQ_SOURCE_COLS = [
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


def apply_fold_frequency(
    official_train,
    official_valid,
    external,
):

    official_train = (
        official_train.copy()
    )

    official_valid = (
        official_valid.copy()
    )

    external = (
        external.copy()
    )

    for col in FREQ_SOURCE_COLS:

        freq = (
            official_train[col]
            .value_counts(
                normalize=True,
                dropna=False,
            )
        )

        freq_col = f"{col}_Freq"

        official_train[freq_col] = (
            official_train[col]
            .map(freq)
            .fillna(0)
            .astype(float)
        )

        official_valid[freq_col] = (
            official_valid[col]
            .map(freq)
            .fillna(0)
            .astype(float)
        )

        external[freq_col] = (
            external[col]
            .map(freq)
            .fillna(0)
            .astype(float)
        )

    return (
        official_train,
        official_valid,
        external,
    )


# =========================================================
# Category alignment
# =========================================================

def align_categories(
    train_df,
    valid_df,
):

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    for col in CATEGORICAL_COLS:

        if col not in train_df.columns:
            continue

        train_values = (
            train_df[col]
            .astype("string")
            .fillna("__MISSING__")
        )

        valid_values = (
            valid_df[col]
            .astype("string")
            .fillna("__MISSING__")
        )

        categories = sorted(
            set(train_values.unique())
            | set(valid_values.unique())
        )

        train_df[col] = pd.Categorical(
            train_values,
            categories=categories,
        )

        valid_df[col] = pd.Categorical(
            valid_values,
            categories=categories,
        )

    return train_df, valid_df


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 100)
    print("LGB EXTERNAL DATA SCREENING")
    print("=" * 100)

    # =====================================================
    # Load official V4
    # =====================================================

    official = pd.read_parquet(
        OFFICIAL_PATH
    )

    official[TARGET] = normalize_target(
        official[TARGET]
    )

    print(
        "Official V4:",
        official.shape,
    )

    # Feature columns:
    # remove target / fold / id
    feature_cols = [
        col
        for col in official.columns
        if col not in [
            TARGET,
            FOLD_COL,
            "id",
        ]
    ]

    print(
        "Official features:",
        len(feature_cols),
    )

    # =====================================================
    # Load original
    # =====================================================

    original_raw = pd.read_csv(
        ORIGINAL_PATH
    )

    original_raw[TARGET] = (
        normalize_target(
            original_raw[TARGET]
        )
    )

    original = build_v4_features(
        original_raw
    )

    # Make sure all V4 features exist
    missing = [
        col
        for col in feature_cols
        if col not in original.columns
    ]

    if missing:

        raise ValueError(
            "Original is missing V4 features:\n"
            + "\n".join(missing)
        )

    print(
        "Original engineered:",
        original.shape,
    )

    print(
        "External rows:",
        len(original),
    )

    print(
        "External positive rate:",
        f"{original[TARGET].mean():.6f}",
    )

    # =====================================================
    # Experiments
    # =====================================================

    folds = (
        official[FOLD_COL]
        .to_numpy()
    )

    all_results = []

    for external_weight in (
        EXTERNAL_WEIGHTS
    ):

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"EXTERNAL WEIGHT = "
            f"{external_weight:.1f}"
        )

        print(
            "=" * 100
        )

        fold_scores = []
        fold_iterations = []

        for fold in SCREEN_FOLDS:

            # ---------------------------------------------
            # Split official
            # ---------------------------------------------

            official_train = (
                official[
                    folds != fold
                ]
                .copy()
            )

            official_valid = (
                official[
                    folds == fold
                ]
                .copy()
            )

            external_fold = (
                original.copy()
            )

            # ---------------------------------------------
            # Fold-safe frequency encoding
            # ---------------------------------------------

            (
                official_train,
                official_valid,
                external_fold,
            ) = apply_fold_frequency(
                official_train,
                official_valid,
                external_fold,
            )

            # ---------------------------------------------
            # Training data
            # ---------------------------------------------

            if external_weight > 0:

                train_df = pd.concat(
                    [
                        official_train[
                            feature_cols
                            + [TARGET]
                        ],

                        external_fold[
                            feature_cols
                            + [TARGET]
                        ],
                    ],
                    axis=0,
                    ignore_index=True,
                )

                sample_weight = np.concatenate(
                    [
                        np.ones(
                            len(
                                official_train
                            ),
                            dtype=np.float32,
                        ),

                        np.full(
                            len(
                                external_fold
                            ),
                            external_weight,
                            dtype=np.float32,
                        ),
                    ]
                )

            else:

                train_df = (
                    official_train[
                        feature_cols
                        + [TARGET]
                    ]
                    .copy()
                )

                sample_weight = None

            valid_df = (
                official_valid[
                    feature_cols
                    + [TARGET]
                ]
                .copy()
            )

            # ---------------------------------------------
            # Category alignment
            # ---------------------------------------------

            X_train = (
                train_df[
                    feature_cols
                ]
            )

            X_valid = (
                valid_df[
                    feature_cols
                ]
            )

            X_train, X_valid = (
                align_categories(
                    X_train,
                    X_valid,
                )
            )

            y_train = (
                train_df[TARGET]
                .to_numpy()
            )

            y_valid = (
                valid_df[TARGET]
                .to_numpy()
            )

            # ---------------------------------------------
            # LGB V5 parameters
            # ---------------------------------------------

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

                random_state=42,
                n_jobs=-1,

                verbosity=-1,
            )

            model.fit(

                X_train,
                y_train,

                sample_weight=sample_weight,

                eval_set=[
                    (
                        X_valid,
                        y_valid,
                    )
                ],

                eval_metric="auc",

                categorical_feature=[
                    col
                    for col
                    in CATEGORICAL_COLS
                    if col in feature_cols
                ],

                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=150,
                        verbose=False,
                    )
                ],
            )

            pred = model.predict_proba(
                X_valid,
                num_iteration=(
                    model.best_iteration_
                ),
            )[:, 1]

            score = roc_auc_score(
                y_valid,
                pred,
            )

            fold_scores.append(
                score
            )

            fold_iterations.append(
                model.best_iteration_
            )

            print(
                f"Fold {fold}: "
                f"AUC={score:.6f} | "
                f"best_iter="
                f"{model.best_iteration_}"
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

        all_results.append({
            "external_weight":
                external_weight,

            "fold_0_auc":
                fold_scores[0],

            "fold_2_auc":
                fold_scores[1],

            "mean_auc":
                mean_auc,

            "avg_best_iteration":
                avg_iter,
        })

        print(
            f"\nMean AUC: "
            f"{mean_auc:.6f}"
        )

    # =====================================================
    # Results
    # =====================================================

    results = pd.DataFrame(
        all_results
    )

    baseline = float(
        results.loc[
            results[
                "external_weight"
            ] == 0,
            "mean_auc",
        ].iloc[0]
    )

    results[
        "delta_vs_official"
    ] = (
        results["mean_auc"]
        - baseline
    )

    results = (
        results
        .sort_values(
            "mean_auc",
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
        "EXTERNAL DATA SCREENING RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        results.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # =====================================================
    # Best
    # =====================================================

    best = results.iloc[0]

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
        f"External weight : "
        f"{best['external_weight']:.1f}"
    )

    print(
        f"Mean AUC        : "
        f"{best['mean_auc']:.6f}"
    )

    print(
        f"Delta           : "
        f"{best['delta_vs_official']:+.6f}"
    )

    # =====================================================
    # Save
    # =====================================================

    output = (
        ROOT
        / "logs"
        / "lgb_external_screening.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output,
        index=False,
    )

    print(
        "\nSaved:",
        output,
    )


if __name__ == "__main__":
    main()