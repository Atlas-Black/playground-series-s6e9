from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train_engineered_v4.parquet"
TRIPLE_OOF_PATH = ROOT / "predictions" / "oof" / "xgb_triple_te.csv"

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"
ID_COL = "id"

SCREEN_FOLDS = [0, 2]

TE_SMOOTHING = 20.0

OUTPUT_DIR = ROOT / "outputs" / "lgb_v8_te_screening"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TOP6 INCOME PATTERNS
# Same definitions as LGB V7 / XGB V5
# =============================================================================

def add_top6_income_features(df):

    df = df.copy()

    inc = df["Annual_Income_USD"].fillna(0)

    df["Income_Mod500"] = inc % 500
    df["Income_Mod5"] = inc % 5

    mod500 = inc % 500
    df["Income_Dist500"] = np.minimum(
        mod500,
        500 - mod500
    )

    mod1000 = inc % 1000
    df["Income_Dist1000"] = np.minimum(
        mod1000,
        1000 - mod1000
    )

    df["Income_Mod20"] = inc % 20

    tens = (inc // 10) % 10
    hundreds = (inc // 100) % 10

    df["Income_TensHundreds"] = (
        tens * 10 + hundreds
    )

    return df


# =============================================================================
# TARGET NORMALIZATION
# =============================================================================

def normalize_target(series):

    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int8")

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }

    result = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if result.isna().any():
        raise ValueError(
            f"Unknown target values: "
            f"{series[result.isna()].unique()}"
        )

    return result.astype("int8")


# =============================================================================
# CATEGORICAL ENCODING
# =============================================================================

def encode_categories(train_df, valid_df, features):

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    encoded_cols = []

    for col in features:

        if (
            train_df[col].dtype == "object"
            or isinstance(
                train_df[col].dtype,
                pd.CategoricalDtype
            )
        ):

            train_cat = train_df[col].astype("category")

            categories = train_cat.cat.categories

            train_df[col] = (
                train_cat.cat.codes
                .astype("int16")
            )

            valid_df[col] = (
                pd.Categorical(
                    valid_df[col],
                    categories=categories
                )
                .codes
                .astype("int16")
            )

            encoded_cols.append(col)

    return (
        train_df,
        valid_df,
        encoded_cols,
    )


# =============================================================================
# FOLD-SAFE FREQUENCY ENCODING
# =============================================================================

def add_income_frequency(
    train_df,
    valid_df
):

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    freq = (
        train_df["Annual_Income_USD"]
        .value_counts(dropna=False)
    )

    train_df["Income_Freq_V8"] = (
        train_df["Annual_Income_USD"]
        .map(freq)
        .fillna(0)
        .astype("float32")
    )

    valid_df["Income_Freq_V8"] = (
        valid_df["Annual_Income_USD"]
        .map(freq)
        .fillna(0)
        .astype("float32")
    )

    return train_df, valid_df


# =============================================================================
# FOLD-SAFE TARGET ENCODING
# =============================================================================

TE_COLS = [
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


def add_target_encoding(
    train_df,
    valid_df,
    target_col,
    smoothing=20.0,
):

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    global_mean = (
        train_df[target_col].mean()
    )

    te_feature_names = []

    for col in TE_COLS:

        if col not in train_df.columns:
            continue

        temp = pd.DataFrame(
            {
                "feature": train_df[col],
                "target": train_df[target_col],
            }
        )

        stats = (
            temp
            .groupby(
                "feature",
                dropna=False,
                observed=False
            )["target"]
            .agg(["mean", "count"])
        )

        stats["smooth"] = (
            (
                stats["mean"]
                * stats["count"]
            )
            +
            (
                global_mean
                * smoothing
            )
        ) / (
            stats["count"]
            + smoothing
        )

        mapping = stats["smooth"]

        te_name = f"{col}_TE"

        # -----------------------------------------------------
        # IMPORTANT:
        # For the fold training portion, this is currently
        # mapping back onto the same training rows.
        #
        # This is acceptable for the FIRST coarse screen only
        # because these columns are low-cardinality and heavily
        # smoothed.
        #
        # If TE wins, final V8 will use inner-fold OOF TE for
        # training rows to remove self-target contamination.
        # -----------------------------------------------------

        train_df[te_name] = (
            train_df[col]
            .map(mapping)
            .fillna(global_mean)
            .astype("float32")
        )

        valid_df[te_name] = (
            valid_df[col]
            .map(mapping)
            .fillna(global_mean)
            .astype("float32")
        )

        te_feature_names.append(
            te_name
        )

    return (
        train_df,
        valid_df,
        te_feature_names,
    )


# =============================================================================
# LOAD TRIPLE TE OOF
# =============================================================================

def load_triple_oof(n_rows):

    if not TRIPLE_OOF_PATH.exists():

        raise FileNotFoundError(
            f"Triple TE OOF not found:\n"
            f"{TRIPLE_OOF_PATH}"
        )

    df = pd.read_csv(
        TRIPLE_OOF_PATH
    )

    print(
        "\nTriple TE columns:",
        list(df.columns)
    )

    print(
        "Triple TE shape:",
        df.shape
    )

    if len(df) != n_rows:

        raise ValueError(
            f"Triple TE length mismatch: "
            f"{len(df)} vs {n_rows}"
        )

    preferred = [
        "prediction",
        "pred",
        "oof_prediction",
        "oof_pred",
        "probability",
        "prob",
    ]

    pred_col = None

    for col in preferred:
        if col in df.columns:
            pred_col = col
            break

    if pred_col is None:

        candidates = [
            col
            for col in df.columns
            if col not in [
                ID_COL,
                TARGET
            ]
        ]

        if len(candidates) == 1:
            pred_col = candidates[0]
        else:
            raise ValueError(
                "Cannot safely determine "
                "Triple TE prediction column. "
                f"Columns={list(df.columns)}"
            )

    pred = pd.to_numeric(
        df[pred_col],
        errors="raise"
    ).to_numpy(
        dtype=np.float64
    )

    print(
        "Triple TE prediction column:",
        pred_col
    )

    return pred


# =============================================================================
# MODEL
# =============================================================================

def build_model():

    return lgb.LGBMClassifier(

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

        # Critical finding from LGB V6/V7
        max_bin=8191,

        random_state=42,

        n_jobs=-1,

        verbosity=-1,
    )


# =============================================================================
# BLEND SEARCH
# weight = Triple TE weight
# =============================================================================

BLEND_WEIGHTS = np.arange(
    0.50,
    1.0001,
    0.05
)


def search_blend(
    y,
    triple_pred,
    lgb_pred,
):

    results = []

    for triple_weight in BLEND_WEIGHTS:

        lgb_weight = (
            1.0 - triple_weight
        )

        pred = (
            triple_weight
            * triple_pred
            +
            lgb_weight
            * lgb_pred
        )

        auc = roc_auc_score(
            y,
            pred
        )

        results.append(
            (
                triple_weight,
                lgb_weight,
                auc
            )
        )

    return results


# =============================================================================
# ONE EXPERIMENT / ONE FOLD
# =============================================================================

def run_fold(
    full_train,
    triple_oof,
    base_features,
    fold,
    experiment_name,
    use_te=False,
    use_income_freq=False,
):

    print()
    print("-" * 100)
    print(
        f"{experiment_name} | FOLD {fold}"
    )
    print("-" * 100)

    train_mask = (
        full_train[FOLD_COL] != fold
    )

    valid_mask = (
        full_train[FOLD_COL] == fold
    )

    fold_train = (
        full_train.loc[
            train_mask
        ].copy()
    )

    fold_valid = (
        full_train.loc[
            valid_mask
        ].copy()
    )

    y_train = (
        fold_train[TARGET]
        .to_numpy()
    )

    y_valid = (
        fold_valid[TARGET]
        .to_numpy()
    )

    features = list(
        base_features
    )

    # ---------------------------------------------------------
    # Income frequency
    # ---------------------------------------------------------

    if use_income_freq:

        fold_train, fold_valid = (
            add_income_frequency(
                fold_train,
                fold_valid
            )
        )

        features.append(
            "Income_Freq_V8"
        )

    # ---------------------------------------------------------
    # Target encoding
    # ---------------------------------------------------------

    if use_te:

        (
            fold_train,
            fold_valid,
            te_features,
        ) = add_target_encoding(
            fold_train,
            fold_valid,
            TARGET,
            smoothing=TE_SMOOTHING,
        )

        features.extend(
            te_features
        )

    # ---------------------------------------------------------
    # Prepare X
    # ---------------------------------------------------------

    X_train = (
        fold_train[features]
        .copy()
    )

    X_valid = (
        fold_valid[features]
        .copy()
    )

    (
        X_train,
        X_valid,
        encoded_cols,
    ) = encode_categories(
        X_train,
        X_valid,
        features,
    )

    # ---------------------------------------------------------
    # Sanity
    # ---------------------------------------------------------

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_valid = X_valid.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        "Train rows:",
        len(X_train)
    )

    print(
        "Valid rows:",
        len(X_valid)
    )

    print(
        "Features  :",
        len(features)
    )

    if use_te:
        print(
            "TE count  :",
            len(te_features)
        )

    if use_income_freq:
        print(
            "IncomeFreq: ON"
        )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    model = build_model()

    start = time.time()

    model.fit(

        X_train,
        y_train,

        eval_set=[
            (
                X_valid,
                y_valid
            )
        ],

        eval_metric="auc",

        callbacks=[
            lgb.early_stopping(
                150,
                verbose=False
            )
        ],
    )

    elapsed = (
        time.time()
        - start
    ) / 60

    valid_pred = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    lgb_auc = roc_auc_score(
        y_valid,
        valid_pred
    )

    # ---------------------------------------------------------
    # Triple TE same fold
    # ---------------------------------------------------------

    triple_valid = (
        triple_oof[
            valid_mask.to_numpy()
        ]
    )

    triple_auc = roc_auc_score(
        y_valid,
        triple_valid
    )

    # ---------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------

    pearson = pearsonr(
        triple_valid,
        valid_pred
    ).statistic

    spearman = spearmanr(
        triple_valid,
        valid_pred
    ).statistic

    # ---------------------------------------------------------
    # Blend
    # ---------------------------------------------------------

    blend_results = search_blend(
        y_valid,
        triple_valid,
        valid_pred,
    )

    best_blend = max(
        blend_results,
        key=lambda x: x[2]
    )

    best_triple_weight = (
        best_blend[0]
    )

    best_lgb_weight = (
        best_blend[1]
    )

    best_blend_auc = (
        best_blend[2]
    )

    blend_gain = (
        best_blend_auc
        - triple_auc
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print()

    print(
        f"LGB AUC       : "
        f"{lgb_auc:.6f}"
    )

    print(
        f"Triple TE AUC : "
        f"{triple_auc:.6f}"
    )

    print(
        f"Pearson       : "
        f"{pearson:.6f}"
    )

    print(
        f"Spearman      : "
        f"{spearman:.6f}"
    )

    print()

    print(
        f"Best blend    : "
        f"Triple {best_triple_weight:.2f} "
        f"+ LGB {best_lgb_weight:.2f}"
    )

    print(
        f"Blend AUC     : "
        f"{best_blend_auc:.6f}"
    )

    print(
        f"Gain vs Triple: "
        f"{blend_gain:+.6f}"
    )

    print(
        f"Best iteration: "
        f"{model.best_iteration_}"
    )

    print(
        f"Runtime       : "
        f"{elapsed:.2f} min"
    )

    result = {

        "experiment":
            experiment_name,

        "fold":
            fold,

        "feature_count":
            len(features),

        "lgb_auc":
            lgb_auc,

        "triple_auc":
            triple_auc,

        "pearson":
            pearson,

        "spearman":
            spearman,

        "best_triple_weight":
            best_triple_weight,

        "best_lgb_weight":
            best_lgb_weight,

        "best_blend_auc":
            best_blend_auc,

        "blend_gain":
            blend_gain,

        "best_iteration":
            model.best_iteration_,

        "runtime_min":
            elapsed,
    }

    del (
        fold_train,
        fold_valid,
        X_train,
        X_valid,
        model,
        valid_pred,
        triple_valid,
    )

    gc.collect()

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "LGB V8 TE SCREENING"
    )

    print(
        "Goal: complementary model "
        "for XGB Triple TE"
    )

    print(
        "=" * 100
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    print(
        "\nLoading train..."
    )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    print(
        "Train shape:",
        train.shape
    )

    if TARGET not in train.columns:
        raise ValueError(
            f"{TARGET} missing."
        )

    if FOLD_COL not in train.columns:
        raise ValueError(
            f"{FOLD_COL} missing."
        )

    train[TARGET] = (
        normalize_target(
            train[TARGET]
        )
    )

    # -------------------------------------------------------------------------
    # TOP6
    # -------------------------------------------------------------------------

    train = (
        add_top6_income_features(
            train
        )
    )

    # -------------------------------------------------------------------------
    # Base features
    # -------------------------------------------------------------------------

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
        "Base features:",
        len(base_features)
    )

    print(
        "Expected:",
        62
    )

    if len(base_features) != 62:

        print(
            "WARNING: expected 62 features "
            "for LGB V7 baseline, "
            f"found {len(base_features)}."
        )

    # -------------------------------------------------------------------------
    # Triple OOF
    # -------------------------------------------------------------------------

    triple_oof = (
        load_triple_oof(
            len(train)
        )
    )

    full_triple_auc = (
        roc_auc_score(
            train[TARGET],
            triple_oof
        )
    )

    print(
        f"\nFull Triple TE OOF AUC: "
        f"{full_triple_auc:.6f}"
    )

    if full_triple_auc > 0.99:

        raise ValueError(
            "Suspicious Triple TE AUC. "
            "Possible target leakage / "
            "wrong prediction column."
        )

    # -------------------------------------------------------------------------
    # Experiments
    # -------------------------------------------------------------------------

    experiments = [

        {
            "name":
                "BASE_V7",

            "use_te":
                False,

            "use_income_freq":
                False,
        },

        {
            "name":
                "TE_BASIC",

            "use_te":
                True,

            "use_income_freq":
                False,
        },

        {
            "name":
                "INCOME_FREQ",

            "use_te":
                False,

            "use_income_freq":
                True,
        },

        {
            "name":
                "TE_PLUS_INCOME_FREQ",

            "use_te":
                True,

            "use_income_freq":
                True,
        },
    ]

    all_results = []

    total_start = time.time()

    for exp in experiments:

        print()
        print()
        print(
            "=" * 100
        )

        print(
            "EXPERIMENT:",
            exp["name"]
        )

        print(
            "=" * 100
        )

        for fold in SCREEN_FOLDS:

            result = run_fold(

                full_train=train,

                triple_oof=triple_oof,

                base_features=base_features,

                fold=fold,

                experiment_name=
                    exp["name"],

                use_te=
                    exp["use_te"],

                use_income_freq=
                    exp["use_income_freq"],
            )

            all_results.append(
                result
            )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    result_path = (
        OUTPUT_DIR
        / "screening_results.csv"
    )

    results.to_csv(
        result_path,
        index=False
    )

    print()
    print(
        "=" * 100
    )

    print(
        "PER-FOLD RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        results[
            [
                "experiment",
                "fold",
                "lgb_auc",
                "triple_auc",
                "spearman",
                "best_blend_auc",
                "blend_gain",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------

    summary = (
        results
        .groupby(
            "experiment",
            as_index=False
        )
        .agg(
            mean_lgb_auc=(
                "lgb_auc",
                "mean"
            ),

            mean_triple_auc=(
                "triple_auc",
                "mean"
            ),

            mean_spearman=(
                "spearman",
                "mean"
            ),

            mean_blend_auc=(
                "best_blend_auc",
                "mean"
            ),

            mean_blend_gain=(
                "blend_gain",
                "mean"
            ),

            mean_best_iteration=(
                "best_iteration",
                "mean"
            ),
        )
    )

    summary = summary.sort_values(
        [
            "mean_blend_auc",
            "mean_lgb_auc",
        ],
        ascending=False
    )

    summary_path = (
        OUTPUT_DIR
        / "screening_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print(
        "=" * 100
    )

    print(
        "SCREENING SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    total_runtime = (
        time.time()
        - total_start
    ) / 60

    print()

    print(
        "Total runtime:",
        f"{total_runtime:.2f} min"
    )

    print(
        "Saved:",
        result_path
    )

    print(
        "Saved:",
        summary_path
    )

    print(
        "\nDONE."
    )


if __name__ == "__main__":
    main()