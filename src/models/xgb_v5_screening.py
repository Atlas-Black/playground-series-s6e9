from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import xgboost as xgb
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

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"
ID_COL = "id"

SCREEN_FOLDS = [0, 2]

SEED = 42

MAX_BINS = [
    256,
    512,
    1024,
    2048,
    4096,
]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "xgb_v5_maxbin_screening"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TOP6 INCOME FEATURES
# Same features used by LGB V7
# ============================================================

TOP6 = [
    "Income_Mod500",
    "Income_Mod5",
    "Income_Dist500",
    "Income_Dist1000",
    "Income_Mod20",
    "Income_TensHundreds",
]


def add_top6_income_features(df):

    df = df.copy()

    income = pd.to_numeric(
        df["Annual_Income_USD"],
        errors="coerce",
    )

    # Safety only.
    # Official train should not contain missing income.
    if income.isna().any():

        income = income.fillna(
            income.median()
        )

    inc = np.rint(
        income
    ).astype(np.int64)

    # --------------------------------------------------------
    # 1. Income mod 500
    # --------------------------------------------------------

    df["Income_Mod500"] = (
        inc % 500
    ).astype("int16")

    # --------------------------------------------------------
    # 2. Income mod 5
    # --------------------------------------------------------

    df["Income_Mod5"] = (
        inc % 5
    ).astype("int8")

    # --------------------------------------------------------
    # 3. Distance to nearest multiple of 500
    # --------------------------------------------------------

    mod500 = (
        inc % 500
    )

    df["Income_Dist500"] = np.minimum(
        mod500,
        500 - mod500,
    ).astype("int16")

    # --------------------------------------------------------
    # 4. Distance to nearest multiple of 1000
    # --------------------------------------------------------

    mod1000 = (
        inc % 1000
    )

    df["Income_Dist1000"] = np.minimum(
        mod1000,
        1000 - mod1000,
    ).astype("int16")

    # --------------------------------------------------------
    # 5. Income mod 20
    # --------------------------------------------------------

    df["Income_Mod20"] = (
        inc % 20
    ).astype("int8")

    # --------------------------------------------------------
    # 6. Tens + hundreds pattern
    # --------------------------------------------------------

    tens = (
        inc // 10
    ) % 10

    hundreds = (
        inc // 100
    ) % 10

    df["Income_TensHundreds"] = (
        tens * 10
        + hundreds
    ).astype("int8")

    return df


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(series):

    if pd.api.types.is_numeric_dtype(
        series
    ):

        return series.astype(
            "int8"
        )

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

    result = cleaned.map(
        mapping
    )

    if result.isna().any():

        unknown = (
            series[
                result.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"Unknown target values: "
            f"{unknown}"
        )

    return result.astype(
        "int8"
    )


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def encode_categories(
    df,
    features,
):

    df = df.copy()

    print(
        "\nEncoding categorical features..."
    )

    encoded_cols = []

    for col in features:

        if (
            df[col].dtype.name
            == "category"
            or df[col].dtype
            == "object"
            or pd.api.types.is_string_dtype(
                df[col]
            )
        ):

            df[col] = (
                df[col]
                .astype("category")
                .cat.codes
                .astype("int16")
            )

            encoded_cols.append(
                col
            )

    print(
        "Encoded categorical columns:",
        len(encoded_cols),
    )

    for col in encoded_cols:

        print(
            " -",
            col,
        )

    return df


# ============================================================
# EVALUATE ONE MAX_BIN
# ============================================================

def evaluate_max_bin(
    train,
    features,
    max_bin,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"MAX_BIN = {max_bin}"
    )

    print(
        "=" * 100
    )

    scores = []
    iterations = []
    runtimes = []

    for fold in SCREEN_FOLDS:

        fold_start = time.time()

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        train_mask = (
            train[FOLD_COL]
            != fold
        )

        valid_mask = (
            train[FOLD_COL]
            == fold
        )

        X_train = train.loc[
            train_mask,
            features,
        ]

        y_train = train.loc[
            train_mask,
            TARGET,
        ]

        X_valid = train.loc[
            valid_mask,
            features,
        ]

        y_valid = train.loc[
            valid_mask,
            TARGET,
        ]

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = xgb.XGBClassifier(

            objective="binary:logistic",

            eval_metric="auc",

            # Enough trees because early stopping
            # will determine the useful iteration.
            n_estimators=5000,

            learning_rate=0.03,

            # Histogram algorithm
            tree_method="hist",

            # ================================================
            # VARIABLE BEING SCREENED
            # ================================================
            max_bin=max_bin,

            # ================================================
            # XGB V5 BASE PARAMETERS
            # ================================================
            max_depth=6,

            min_child_weight=5,

            subsample=0.85,

            colsample_bytree=0.85,

            reg_alpha=0.0,

            reg_lambda=1.0,

            gamma=0.0,

            # ================================================
            # OTHER
            # ================================================
            random_state=SEED,

            n_jobs=-1,

            early_stopping_rounds=150,
        )

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        model.fit(

            X_train,
            y_train,

            eval_set=[
                (
                    X_valid,
                    y_valid,
                )
            ],

            verbose=False,
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        pred = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        # ----------------------------------------------------
        # AUC
        # ----------------------------------------------------

        score = roc_auc_score(
            y_valid,
            pred,
        )

        best_iter = (
            model.best_iteration
            if model.best_iteration
            is not None
            else model.n_estimators
        )

        elapsed = (
            time.time()
            - fold_start
        ) / 60

        scores.append(
            score
        )

        iterations.append(
            best_iter
        )

        runtimes.append(
            elapsed
        )

        print(
            f"Fold {fold}: "
            f"AUC={score:.6f} | "
            f"best_iter={best_iter} | "
            f"time={elapsed:.2f} min"
        )

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        del model
        del X_train
        del X_valid
        del y_train
        del y_valid
        del pred

        gc.collect()

    # ========================================================
    # SUMMARY
    # ========================================================

    mean_auc = float(
        np.mean(scores)
    )

    avg_best_iteration = float(
        np.mean(iterations)
    )

    runtime_minutes = float(
        np.sum(runtimes)
    )

    print(
        f"\nMean AUC : "
        f"{mean_auc:.6f}"
    )

    print(
        f"Avg iter : "
        f"{avg_best_iteration:.1f}"
    )

    print(
        f"Runtime  : "
        f"{runtime_minutes:.2f} min"
    )

    return {

        "max_bin":
            max_bin,

        "fold_0_auc":
            scores[0],

        "fold_2_auc":
            scores[1],

        "mean_auc":
            mean_auc,

        "avg_best_iteration":
            avg_best_iteration,

        "runtime_minutes":
            runtime_minutes,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "XGB V5 MAX_BIN SCREENING"
    )

    print(
        "=" * 100
    )

    print(
        "Screen folds:",
        SCREEN_FOLDS,
    )

    print(
        "MAX_BIN candidates:",
        MAX_BINS,
    )

    # ========================================================
    # LOAD TRAIN
    # ========================================================

    print(
        "\nLoading:"
    )

    print(
        TRAIN_PATH
    )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    print(
        "Train shape:",
        train.shape,
    )

    # ========================================================
    # TARGET
    # ========================================================

    train[TARGET] = (
        normalize_target(
            train[TARGET]
        )
    )

    print(
        "\nTarget distribution:"
    )

    print(
        train[TARGET]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # BASE 56 FEATURES
    # ========================================================

    base_features = [

        col

        for col
        in train.columns

        if col not in [
            TARGET,
            FOLD_COL,
            ID_COL,
        ]
    ]

    print(
        "\nBase features:",
        len(base_features),
    )

    if len(base_features) != 56:

        print(
            "WARNING:"
            f" expected 56 base features, "
            f"found {len(base_features)}"
        )

    # ========================================================
    # ADD TOP6
    # ========================================================

    train = (
        add_top6_income_features(
            train
        )
    )

    features = (
        base_features
        + TOP6
    )

    print(
        "TOP6 features:",
        len(TOP6),
    )

    print(
        "Total features:",
        len(features),
    )

    if len(features) != 62:

        raise ValueError(
            f"Expected 62 features, "
            f"found {len(features)}"
        )

    print(
        "\nTOP6 income features:"
    )

    for feature in TOP6:

        print(
            " -",
            feature,
        )

    # ========================================================
    # CHECK DUPLICATES
    # ========================================================

    duplicates = [

        feature

        for feature
        in TOP6

        if feature
        in base_features
    ]

    if duplicates:

        raise ValueError(
            "TOP6 already exist in "
            f"base features: {duplicates}"
        )

    # ========================================================
    # ENCODE CATEGORICAL FEATURES
    # ========================================================

    train = (
        encode_categories(
            train,
            features,
        )
    )

    # ========================================================
    # CHECK NaN / INF
    # ========================================================

    print(
        "\nChecking feature matrix..."
    )

    numeric_check = (
        train[features]
        .select_dtypes(
            include=[np.number]
        )
    )

    inf_count = int(
        np.isinf(
            numeric_check
            .to_numpy()
        ).sum()
    )

    print(
        "Infinite values:",
        inf_count,
    )

    if inf_count > 0:

        raise ValueError(
            "Feature matrix contains "
            "infinite values."
        )

    # ========================================================
    # SCREENING
    # ========================================================

    results = []

    total_start = time.time()

    for max_bin in MAX_BINS:

        result = (
            evaluate_max_bin(
                train=train,
                features=features,
                max_bin=max_bin,
            )
        )

        results.append(
            result
        )

    total_runtime = (
        time.time()
        - total_start
    ) / 60

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Baseline = max_bin 256
    # --------------------------------------------------------

    baseline_rows = (
        result_df[
            result_df["max_bin"]
            == 256
        ]
    )

    if baseline_rows.empty:

        raise ValueError(
            "max_bin=256 baseline "
            "was not found."
        )

    baseline = (
        baseline_rows.iloc[0]
    )

    baseline_mean = float(
        baseline["mean_auc"]
    )

    baseline_fold0 = float(
        baseline["fold_0_auc"]
    )

    baseline_fold2 = float(
        baseline["fold_2_auc"]
    )

    # --------------------------------------------------------
    # Deltas
    # --------------------------------------------------------

    result_df[
        "delta_vs_256"
    ] = (
        result_df["mean_auc"]
        - baseline_mean
    )

    result_df[
        "fold0_delta"
    ] = (
        result_df["fold_0_auc"]
        - baseline_fold0
    )

    result_df[
        "fold2_delta"
    ] = (
        result_df["fold_2_auc"]
        - baseline_fold2
    )

    result_df[
        "positive_folds"
    ] = (
        (
            result_df[
                "fold0_delta"
            ] > 0
        ).astype(int)
        +
        (
            result_df[
                "fold2_delta"
            ] > 0
        ).astype(int)
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result_df = (
        result_df
        .sort_values(
            "mean_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 130
    )

    print(
        "XGB V5 MAX_BIN SCREENING RESULTS"
    )

    print(
        "=" * 130
    )

    print(
        result_df.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # BEST
    # ========================================================

    best = (
        result_df.iloc[0]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST MAX_BIN"
    )

    print(
        "=" * 80
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
        f"Delta vs 256  : "
        f"{best['delta_vs_256']:+.6f}"
    )

    print(
        f"Fold 0 AUC    : "
        f"{best['fold_0_auc']:.6f}"
    )

    print(
        f"Fold 0 delta  : "
        f"{best['fold0_delta']:+.6f}"
    )

    print(
        f"Fold 2 AUC    : "
        f"{best['fold_2_auc']:.6f}"
    )

    print(
        f"Fold 2 delta  : "
        f"{best['fold2_delta']:+.6f}"
    )

    print(
        f"Positive folds: "
        f"{int(best['positive_folds'])}/2"
    )

    print(
        f"Avg best iter : "
        f"{best['avg_best_iteration']:.1f}"
    )

    print(
        f"Runtime       : "
        f"{best['runtime_minutes']:.2f} min"
    )

    print(
        f"\nTotal screening runtime: "
        f"{total_runtime:.2f} min"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / "xgb_v5_maxbin_screening.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        "\nDONE."
    )


if __name__ == "__main__":
    main()