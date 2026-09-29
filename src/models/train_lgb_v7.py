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
    ROOT / "data" / "processed" / "train_engineered_v4.parquet"
)

TEST_PATH = (
    ROOT / "data" / "processed" / "test_engineered_v4.parquet"
)

RAW_TEST_PATH = (
    ROOT / "data" / "raw" / "test.csv"
)

SAMPLE_SUB_PATH = (
    ROOT / "data" / "raw" / "sample_submission.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "lgb_v7"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"
ID_COL = "id"

SEED = 42

MAX_BIN = 8191
BIN_SAMPLE_COUNT = 200_000


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
# TOP6 INCOME FEATURES
# ============================================================

def add_top6_income_features(df):

    df = df.copy()

    income = pd.to_numeric(
        df["Annual_Income_USD"],
        errors="coerce",
    )

    # Official train/test should be complete,
    # but keep this safe.
    if income.isna().any():
        income = income.fillna(
            income.median()
        )

    inc = (
        np.rint(income)
        .astype(np.int64)
    )

    # 1
    df["Income_Mod500"] = (
        inc % 500
    ).astype("int16")

    # 2
    df["Income_Mod5"] = (
        inc % 5
    ).astype("int8")

    # 3
    mod500 = inc % 500

    df["Income_Dist500"] = np.minimum(
        mod500,
        500 - mod500,
    ).astype("int16")

    # 4
    mod1000 = inc % 1000

    df["Income_Dist1000"] = np.minimum(
        mod1000,
        1000 - mod1000,
    ).astype("int16")

    # 5
    df["Income_Mod20"] = (
        inc % 20
    ).astype("int8")

    # 6
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


TOP6 = [
    "Income_Mod500",
    "Income_Mod5",
    "Income_Dist500",
    "Income_Dist1000",
    "Income_Mod20",
    "Income_TensHundreds",
]


# ============================================================
# CATEGORIES
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("LGB V7 - FULL 5-FOLD")
    print("=" * 100)

    print(
        f"max_bin                  = {MAX_BIN}"
    )

    print(
        f"bin_construct_sample_cnt = {BIN_SAMPLE_COUNT:,}"
    )

    print(
        "Income patterns          = TOP6"
    )

    # ========================================================
    # LOAD
    # ========================================================

    train = pd.read_parquet(
        TRAIN_PATH
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    print(
        "\nOriginal train:",
        train.shape
    )

    print(
        "Original test :",
        test.shape
    )

    train[TARGET] = normalize_target(
        train[TARGET]
    )

    # ========================================================
    # BASE FEATURES
    # ========================================================

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
        "Base features :",
        len(base_features)
    )

    if len(base_features) != 56:

        print(
            f"WARNING: expected 56 base features, "
            f"found {len(base_features)}"
        )

    # ========================================================
    # TOP6
    # ========================================================

    train = add_top6_income_features(
        train
    )

    test = add_top6_income_features(
        test
    )

    features = (
        base_features
        + TOP6
    )

    print(
        "V7 features   :",
        len(features)
    )

    print(
        "\nTOP6:"
    )

    for feature in TOP6:
        print(
            " -",
            feature,
        )

    if len(features) != 62:
        raise ValueError(
            f"Expected 62 features, "
            f"found {len(features)}"
        )

    # ========================================================
    # CATEGORY ALIGNMENT
    # ========================================================

    train, test = align_categories(
        train,
        test,
    )

    active_categorical = [
        col
        for col in CATEGORICAL_COLS
        if col in features
    ]

    # ========================================================
    # DATA
    # ========================================================

    y = (
        train[TARGET]
        .to_numpy()
    )

    folds = (
        train[FOLD_COL]
        .to_numpy()
    )

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
    fold_times = []

    total_start = time.time()

    # ========================================================
    # 5 FOLDS
    # ========================================================

    for fold in range(5):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"FOLD {fold}"
        )

        print(
            "=" * 80
        )

        fold_start = time.time()

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

        X_test = (
            test[features]
        )

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # ====================================================
        # MODEL
        # ====================================================

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

            max_bin=MAX_BIN,

            bin_construct_sample_cnt=(
                BIN_SAMPLE_COUNT
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

        # ====================================================
        # VALID
        # ====================================================

        valid_pred = (
            model.predict_proba(
                X_valid,
                num_iteration=(
                    model.best_iteration_
                ),
            )[:, 1]
        )

        oof[valid_idx] = (
            valid_pred
        )

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred,
        )

        # ====================================================
        # TEST
        # ====================================================

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

        # ====================================================
        # LOG
        # ====================================================

        elapsed = (
            time.time()
            - fold_start
        ) / 60

        fold_scores.append(
            fold_auc
        )

        best_iterations.append(
            model.best_iteration_
        )

        fold_times.append(
            elapsed
        )

        print(
            f"AUC       : {fold_auc:.6f}"
        )

        print(
            f"Best iter : "
            f"{model.best_iteration_}"
        )

        print(
            f"Time      : "
            f"{elapsed:.2f} min"
        )

        del model
        del X_train
        del X_valid
        del X_test

        gc.collect()

    # ========================================================
    # FINAL METRICS
    # ========================================================

    oof_auc = roc_auc_score(
        y,
        oof,
    )

    mean_auc = float(
        np.mean(fold_scores)
    )

    std_auc = float(
        np.std(fold_scores)
    )

    avg_iter = float(
        np.mean(best_iterations)
    )

    total_time = (
        time.time()
        - total_start
    ) / 60

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LGB V7 FINAL RESULTS"
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
        f"{mean_auc:.6f}"
    )

    print(
        f"Std fold AUC  : "
        f"{std_auc:.6f}"
    )

    print(
        f"OOF AUC       : "
        f"{oof_auc:.6f}"
    )

    print(
        f"Avg best iter : "
        f"{avg_iter:.1f}"
    )

    print(
        f"Runtime       : "
        f"{total_time:.2f} min"
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    V5_OOF = 0.943743
    V6_OOF = 0.943909

    print(
        "\n"
        + "-" * 100
    )

    print(
        "COMPARISON"
    )

    print(
        "-" * 100
    )

    print(
        f"LGB V5 : {V5_OOF:.6f}"
    )

    print(
        f"LGB V6 : {V6_OOF:.6f}"
    )

    print(
        f"LGB V7 : {oof_auc:.6f}"
    )

    print(
        f"\nV7 vs V6: "
        f"{oof_auc - V6_OOF:+.6f}"
    )

    print(
        f"V7 vs V5: "
        f"{oof_auc - V5_OOF:+.6f}"
    )

    # ========================================================
    # SAVE OOF
    # ========================================================

    oof_df = pd.DataFrame({
        "row_index":
            np.arange(len(train)),

        TARGET:
            y,

        "prediction":
            oof,

        FOLD_COL:
            folds,
    })

    oof_path = (
        OUTPUT_DIR
        / "oof_lgb_v7.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    # ========================================================
    # SAVE TEST PREDICTION
    # ========================================================

    test_pred_df = pd.DataFrame({
        "row_index":
            np.arange(len(test)),

        "prediction":
            test_pred,
    })

    test_pred_path = (
        OUTPUT_DIR
        / "test_lgb_v7.csv"
    )

    test_pred_df.to_csv(
        test_pred_path,
        index=False,
    )

    # ========================================================
    # SUBMISSION
    # ========================================================

    sample_sub = pd.read_csv(
        SAMPLE_SUB_PATH
    )

    print(
        "\nSample submission columns:",
        sample_sub.columns.tolist()
    )

    if len(sample_sub) != len(test_pred):

        raise ValueError(
            "Sample submission row count "
            "does not match test predictions."
        )

    submission = (
        sample_sub.copy()
    )

    # Keep official ID exactly as provided
    submission[TARGET] = (
        test_pred
    )

    submission_path = (
        OUTPUT_DIR
        / "submission_lgb_v7.csv"
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame({
        "fold": [
            0,
            1,
            2,
            3,
            4,
        ],
        "auc":
            fold_scores,
        "best_iteration":
            best_iterations,
        "runtime_minutes":
            fold_times,
    })

    summary_path = (
        OUTPUT_DIR
        / "fold_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 100
    )

    print(
        "OOF       :",
        oof_path,
    )

    print(
        "Test pred :",
        test_pred_path,
    )

    print(
        "Submission:",
        submission_path,
    )

    print(
        "Summary   :",
        summary_path,
    )

    print(
        "\nSubmission shape:",
        submission.shape,
    )

    print(
        "Prediction min:",
        test_pred.min(),
    )

    print(
        "Prediction max:",
        test_pred.max(),
    )

    print(
        "Prediction mean:",
        test_pred.mean(),
    )

    print(
        "\nDONE."
    )


if __name__ == "__main__":
    main()