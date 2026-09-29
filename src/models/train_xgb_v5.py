from pathlib import Path
import gc
import shutil
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_TRAIN_PATH = (
    ROOT / "data" / "raw" / "train.csv"
)

RAW_TEST_PATH = (
    ROOT / "data" / "raw" / "test.csv"
)

TRAIN_PATH = (
    ROOT / "data" / "processed" / "train_engineered_v4.parquet"
)

TEST_PATH = (
    ROOT / "data" / "processed" / "test_engineered_v4.parquet"
)

SAMPLE_PATH = (
    ROOT / "data" / "raw" / "sample_submission.csv"
)

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"
ID_COL = "id"

SEED = 42
N_FOLDS = 5

OUTPUT_DIR = ROOT / "outputs" / "xgb_v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_DIR = ROOT / "predictions" / "oof"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TOP6 INCOME FEATURES
# Same TOP6 used by LGB V7
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

    if income.isna().any():
        income = income.fillna(
            income.median()
        )

    inc = np.rint(
        income
    ).astype(np.int64)

    # --------------------------------------------------------
    # Mod 500
    # --------------------------------------------------------

    df["Income_Mod500"] = (
        inc % 500
    ).astype("int16")

    # --------------------------------------------------------
    # Mod 5
    # --------------------------------------------------------

    df["Income_Mod5"] = (
        inc % 5
    ).astype("int8")

    # --------------------------------------------------------
    # Distance to nearest multiple of 500
    # --------------------------------------------------------

    mod500 = inc % 500

    df["Income_Dist500"] = np.minimum(
        mod500,
        500 - mod500,
    ).astype("int16")

    # --------------------------------------------------------
    # Distance to nearest multiple of 1000
    # --------------------------------------------------------

    mod1000 = inc % 1000

    df["Income_Dist1000"] = np.minimum(
        mod1000,
        1000 - mod1000,
    ).astype("int16")

    # --------------------------------------------------------
    # Mod 20
    # --------------------------------------------------------

    df["Income_Mod20"] = (
        inc % 20
    ).astype("int8")

    # --------------------------------------------------------
    # Tens + Hundreds
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
# CATEGORICAL ENCODING
#
# Important:
# train + test are encoded together so category codes match.
# This uses no target information.
# ============================================================

def encode_categories(
    train,
    test,
    features,
):

    train = train.copy()
    test = test.copy()

    categorical_cols = []

    for col in features:

        train_is_cat = (
            train[col].dtype.name == "category"
            or train[col].dtype == "object"
            or pd.api.types.is_string_dtype(
                train[col]
            )
        )

        test_is_cat = (
            test[col].dtype.name == "category"
            or test[col].dtype == "object"
            or pd.api.types.is_string_dtype(
                test[col]
            )
        )

        if train_is_cat or test_is_cat:

            categorical_cols.append(col)

            combined = pd.concat(
                [
                    train[col].astype(str),
                    test[col].astype(str),
                ],
                axis=0,
                ignore_index=True,
            )

            categories = pd.Index(
                combined.unique()
            )

            train[col] = pd.Categorical(
                train[col].astype(str),
                categories=categories,
            ).codes.astype("int16")

            test[col] = pd.Categorical(
                test[col].astype(str),
                categories=categories,
            ).codes.astype("int16")

    print(
        f"Encoded categorical columns: "
        f"{len(categorical_cols)}"
    )

    for col in categorical_cols:
        print(" -", col)

    return (
        train,
        test,
        categorical_cols,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print("=" * 100)
    print("XGB V5 FINAL TRAINING")
    print("=" * 100)

    print("\nConfiguration:")
    print("Features       : V4 56 + TOP6 = 62")
    print("Folds          : 5")
    print("max_bin        : 2048")
    print("max_depth      : 6")
    print("learning_rate  : 0.03")
    print("seed           : 42")

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\n" + "=" * 100)
    print("LOADING DATA")
    print("=" * 100)

    train = pd.read_parquet(
        TRAIN_PATH
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    print(
        "Train shape:",
        train.shape,
    )

    print(
        "Test shape :",
        test.shape,
    )

    print(
        "Sample shape:",
        sample.shape,
    )

    # ========================================================
    # TARGET
    # ========================================================

    train[TARGET] = normalize_target(
        train[TARGET]
    )

    print("\nTarget distribution:")

    print(
        train[TARGET]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # LOAD IDS FROM RAW DATA
    # ========================================================

    print("\nLoading IDs from raw data...")

    raw_train = pd.read_csv(
        RAW_TRAIN_PATH,
        usecols=[ID_COL],
    )

    raw_test = pd.read_csv(
        RAW_TEST_PATH,
        usecols=[ID_COL],
    )

    if len(raw_train) != len(train):
        raise ValueError(
            f"Raw/processed train length mismatch: "
            f"{len(raw_train)} vs {len(train)}"
        )

    if len(raw_test) != len(test):
        raise ValueError(
            f"Raw/processed test length mismatch: "
            f"{len(raw_test)} vs {len(test)}"
        )

    train_ids = (
        raw_train[ID_COL]
        .to_numpy()
        .copy()
    )

    test_ids = (
        raw_test[ID_COL]
        .to_numpy()
        .copy()
    )

    print(
        "Train IDs:",
        len(train_ids),
    )

    print(
        "Test IDs :",
        len(test_ids),
    )

    del raw_train
    del raw_test

    gc.collect()

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
        "\nBase feature count:",
        len(base_features),
    )

    if len(base_features) != 56:

        raise ValueError(
            f"Expected 56 V4 features, "
            f"found {len(base_features)}"
        )

    # ========================================================
    # ADD TOP6
    # ========================================================

    train = add_top6_income_features(
        train
    )

    test = add_top6_income_features(
        test
    )

    duplicate_top6 = [
        col
        for col in TOP6
        if col in base_features
    ]

    if duplicate_top6:

        raise ValueError(
            "TOP6 already present in base features: "
            f"{duplicate_top6}"
        )

    features = (
        base_features
        + TOP6
    )

    print(
        "TOP6 count:",
        len(TOP6),
    )

    print(
        "Total feature count:",
        len(features),
    )

    if len(features) != 62:

        raise ValueError(
            f"Expected 62 features, "
            f"found {len(features)}"
        )

    print("\nTOP6:")

    for col in TOP6:
        print(" -", col)

    # ========================================================
    # CHECK FEATURE CONSISTENCY
    # ========================================================

    missing_test = [
        col
        for col in features
        if col not in test.columns
    ]

    if missing_test:

        raise ValueError(
            f"Features missing from test: "
            f"{missing_test}"
        )

    # ========================================================
    # ENCODE CATEGORIES
    # ========================================================

    print("\n" + "=" * 100)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 100)

    (
        train,
        test,
        categorical_cols,
    ) = encode_categories(
        train,
        test,
        features,
    )

    # ========================================================
    # CHECK MATRIX
    # ========================================================

    print("\n" + "=" * 100)
    print("DATA CHECK")
    print("=" * 100)

    train_numeric = (
        train[features]
        .select_dtypes(
            include=[np.number]
        )
    )

    test_numeric = (
        test[features]
        .select_dtypes(
            include=[np.number]
        )
    )

    train_inf = int(
        np.isinf(
            train_numeric.to_numpy()
        ).sum()
    )

    test_inf = int(
        np.isinf(
            test_numeric.to_numpy()
        ).sum()
    )

    print(
        "Train infinite values:",
        train_inf,
    )

    print(
        "Test infinite values :",
        test_inf,
    )

    if train_inf > 0 or test_inf > 0:

        raise ValueError(
            "Infinite values found."
        )

    # ========================================================
    # ARRAYS
    # ========================================================

    oof_pred = np.zeros(
        len(train),
        dtype=np.float64,
    )

    test_pred = np.zeros(
        len(test),
        dtype=np.float64,
    )

    fold_results = []

    # ========================================================
    # TRAIN 5 FOLDS
    # ========================================================

    print("\n" + "=" * 100)
    print("5-FOLD TRAINING")
    print("=" * 100)

    for fold in range(N_FOLDS):

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"FOLD {fold}"
        )

        print(
            "-" * 100
        )

        fold_start = time.time()

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

        X_test = test[
            features
        ]

        print(
            "Train rows:",
            len(X_train),
        )

        print(
            "Valid rows:",
            len(X_valid),
        )

        # ====================================================
        # MODEL
        # ====================================================

        model = xgb.XGBClassifier(

            objective="binary:logistic",

            eval_metric="auc",

            n_estimators=5000,

            learning_rate=0.03,

            tree_method="hist",

            # ================================================
            # SCREENING WINNER
            # ================================================
            max_bin=2048,

            # ================================================
            # XGB V5 BASE
            # ================================================
            max_depth=6,

            min_child_weight=5,

            subsample=0.85,

            colsample_bytree=0.85,

            reg_alpha=0.0,

            reg_lambda=1.0,

            gamma=0.0,

            random_state=SEED,

            n_jobs=-1,

            early_stopping_rounds=150,
        )

        # ====================================================
        # TRAIN
        # ====================================================

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

        # ====================================================
        # VALID PREDICTION
        # ====================================================

        valid_pred = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        oof_pred[
            valid_mask.to_numpy()
        ] = valid_pred

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred,
        )

        # ====================================================
        # TEST PREDICTION
        # ====================================================

        fold_test_pred = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        test_pred += (
            fold_test_pred
            / N_FOLDS
        )

        # ====================================================
        # INFO
        # ====================================================

        best_iter = (
            model.best_iteration
            if model.best_iteration
            is not None
            else 5000
        )

        elapsed = (
            time.time()
            - fold_start
        ) / 60

        print(
            f"Fold {fold}: "
            f"AUC={fold_auc:.6f} | "
            f"best_iter={best_iter} | "
            f"time={elapsed:.2f} min"
        )

        fold_results.append(
            {
                "fold":
                    fold,

                "auc":
                    fold_auc,

                "best_iteration":
                    best_iter,

                "runtime_minutes":
                    elapsed,
            }
        )

        # ====================================================
        # CLEAN
        # ====================================================

        del model
        del X_train
        del X_valid
        del X_test
        del y_train
        del y_valid
        del valid_pred
        del fold_test_pred

        gc.collect()

    # ========================================================
    # FINAL SCORES
    # ========================================================

    y_all = (
        train[TARGET]
        .to_numpy()
    )

    oof_auc = roc_auc_score(
        y_all,
        oof_pred,
    )

    fold_df = pd.DataFrame(
        fold_results
    )

    mean_fold_auc = (
        fold_df["auc"].mean()
    )

    std_fold_auc = (
        fold_df["auc"].std(
            ddof=0
        )
    )

    avg_best_iter = (
        fold_df[
            "best_iteration"
        ].mean()
    )

    total_runtime = (
        time.time()
        - total_start
    ) / 60

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "XGB V5 FINAL RESULTS"
    )

    print(
        "=" * 100
    )

    for _, row in fold_df.iterrows():

        print(
            f"Fold {int(row['fold'])}: "
            f"{row['auc']:.6f}"
        )

    print()

    print(
        f"Mean fold AUC : "
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
        f"{total_runtime:.2f} min"
    )

    # ========================================================
    # SAVE OOF
    # ========================================================

    oof_path = (
        OUTPUT_DIR
        / "oof_xgb_v5.csv"
    )

    oof_df = pd.DataFrame(
        {
            ID_COL:
                train_ids,

            TARGET:
                oof_pred,
        }
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    # ========================================================
    # SAVE TEST PREDICTION
    # ========================================================

    test_path = (
        OUTPUT_DIR
        / "test_xgb_v5.csv"
    )

    test_df = pd.DataFrame(
        {
            ID_COL:
                test_ids,

            TARGET:
                test_pred,
        }
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    # ========================================================
    # SUBMISSION
    # ========================================================

    print(
        "\nSample submission columns:",
        list(sample.columns),
    )

    if (
        ID_COL not in sample.columns
        or TARGET not in sample.columns
    ):

        raise ValueError(
            "Unexpected sample_submission columns: "
            f"{list(sample.columns)}"
        )

    submission = sample.copy()

    if len(submission) != len(test_pred):

        raise ValueError(
            "Submission/test length mismatch: "
            f"{len(submission)} vs "
            f"{len(test_pred)}"
        )

    # Preserve sample IDs exactly.
    submission[TARGET] = (
        test_pred
    )

    submission_path = (
        OUTPUT_DIR
        / "submission_xgb_v5.csv"
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    # ========================================================
    # SAVE FOLD SUMMARY
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "fold_summary.csv"
    )

    fold_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # COPY OOF FOR ENSEMBLE SCRIPT
    # ========================================================

    prediction_oof_path = (
        PREDICTION_DIR
        / "xgb_v5.csv"
    )

    shutil.copyfile(
        oof_path,
        prediction_oof_path,
    )

    # ========================================================
    # FINAL VALIDATION
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
        test_path,
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
        "OOF copy  :",
        prediction_oof_path,
    )

    print()

    print(
        "OOF shape       :",
        oof_df.shape,
    )

    print(
        "Submission shape:",
        submission.shape,
    )

    print(
        "Prediction min  :",
        float(test_pred.min()),
    )

    print(
        "Prediction max  :",
        float(test_pred.max()),
    )

    print(
        "Prediction mean :",
        float(test_pred.mean()),
    )

    if not np.isfinite(
        test_pred
    ).all():

        raise ValueError(
            "Non-finite test predictions."
        )

    if (
        (test_pred < 0).any()
        or (test_pred > 1).any()
    ):

        raise ValueError(
            "Predictions outside [0, 1]."
        )

    print(
        "\nDONE."
    )


if __name__ == "__main__":
    main()