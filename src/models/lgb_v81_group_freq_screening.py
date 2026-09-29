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

OUTPUT_DIR = ROOT / "outputs" / "lgb_v81_group_freq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BASIC CONFIG
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


# =============================================================================
# GROUP FREQUENCY CANDIDATES
# =============================================================================

GROUP_FREQ_CANDIDATES = {

    "INC_AGE":
        ["Income_100_Bin", "Age"],

    "INC_CITY":
        ["Income_100_Bin", "City_Type"],

    "INC_CAR":
        ["Income_100_Bin", "Current_Car_Type"],

    "INC_SUBSIDY":
        ["Income_100_Bin", "Subsidy_Available"],

    "INC_HOME":
        ["Income_100_Bin", "Home_Charging_Possible"],

    "CITY_CAR":
        ["City_Type", "Current_Car_Type"],

    "COMMUTE_CHARGING":
        ["Daily_Commute_km", "Total_Charging_Stations"],
}


# =============================================================================
# TARGET
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
# TOP6 INCOME FEATURES
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
# CATEGORY ENCODING
# =============================================================================

def encode_categories(train_df, valid_df, features):

    train_df = train_df.copy()
    valid_df = valid_df.copy()

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
                .astype("int32")
            )

            valid_df[col] = (
                pd.Categorical(
                    valid_df[col],
                    categories=categories
                )
                .codes
                .astype("int32")
            )

    return train_df, valid_df


# =============================================================================
# INCOME FREQUENCY
# =============================================================================

def add_income_frequency(train_df, valid_df):

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
# BASIC TE
# =============================================================================

def add_target_encoding(
    train_df,
    valid_df,
    smoothing=20.0,
):

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    global_mean = train_df[TARGET].mean()

    te_features = []

    for col in TE_COLS:

        if col not in train_df.columns:
            continue

        stats = (
            train_df
            .groupby(
                col,
                dropna=False,
                observed=False
            )[TARGET]
            .agg(["mean", "count"])
        )

        stats["smooth"] = (
            stats["mean"] * stats["count"]
            +
            global_mean * smoothing
        ) / (
            stats["count"] + smoothing
        )

        mapping = stats["smooth"]

        name = f"{col}_TE"

        train_df[name] = (
            train_df[col]
            .map(mapping)
            .fillna(global_mean)
            .astype("float32")
        )

        valid_df[name] = (
            valid_df[col]
            .map(mapping)
            .fillna(global_mean)
            .astype("float32")
        )

        te_features.append(name)

    return train_df, valid_df, te_features


# =============================================================================
# GROUP FREQUENCY
# =============================================================================

def add_group_frequency(
    train_df,
    valid_df,
    cols,
    feature_name,
):

    """
    IMPORTANT:
    Mapping is learned ONLY from fold-training rows.
    No validation rows are used to construct frequency counts.
    """

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    missing = [
        c for c in cols
        if c not in train_df.columns
    ]

    if missing:
        raise ValueError(
            f"{feature_name}: missing columns {missing}"
        )

    counts = (
        train_df
        .groupby(
            cols,
            dropna=False,
            observed=False
        )
        .size()
        .rename(feature_name)
        .reset_index()
    )

    # Train
    train_df = train_df.merge(
        counts,
        on=cols,
        how="left",
        sort=False,
    )

    # Valid
    valid_df = valid_df.merge(
        counts,
        on=cols,
        how="left",
        sort=False,
    )

    train_df[feature_name] = (
        train_df[feature_name]
        .fillna(0)
        .astype("float32")
    )

    valid_df[feature_name] = (
        valid_df[feature_name]
        .fillna(0)
        .astype("float32")
    )

    return train_df, valid_df


# =============================================================================
# LOAD TRIPLE OOF
# =============================================================================

def load_triple_oof(n_rows):

    df = pd.read_csv(TRIPLE_OOF_PATH)

    if len(df) != n_rows:
        raise ValueError(
            f"Triple OOF length mismatch: "
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
            c for c in df.columns
            if c not in [ID_COL, TARGET]
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"Cannot determine prediction column: "
                f"{list(df.columns)}"
            )

        pred_col = candidates[0]

    pred = pd.to_numeric(
        df[pred_col],
        errors="raise"
    ).to_numpy(dtype=np.float64)

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

        max_bin=8191,

        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


# =============================================================================
# BLEND
# =============================================================================

BLEND_WEIGHTS = np.arange(
    0.70,
    0.951,
    0.025
)


def evaluate_blend(
    y,
    triple,
    pred,
):

    rows = []

    for wt in BLEND_WEIGHTS:

        wl = 1.0 - wt

        blend = (
            wt * triple
            +
            wl * pred
        )

        auc = roc_auc_score(
            y,
            blend
        )

        rows.append(
            (wt, wl, auc)
        )

    return max(
        rows,
        key=lambda x: x[2]
    )


# =============================================================================
# RUN ONE FOLD
# =============================================================================

def run_fold(
    full_train,
    triple_oof,
    base_features,
    fold,
    experiment,
    groups,
):

    print()
    print("-" * 100)
    print(
        f"{experiment} | FOLD {fold}"
    )
    print("-" * 100)

    train_mask = (
        full_train[FOLD_COL] != fold
    )

    valid_mask = (
        full_train[FOLD_COL] == fold
    )

    tr = full_train.loc[
        train_mask
    ].copy()

    va = full_train.loc[
        valid_mask
    ].copy()

    # Keep original row identity before merges.
    tr["_row_order"] = np.arange(len(tr))
    va["_row_order"] = np.arange(len(va))

    # -------------------------------------------------------------------------
    # BASE = V7 + IncomeFreq + Basic TE
    # -------------------------------------------------------------------------

    tr, va = add_income_frequency(
        tr,
        va
    )

    tr, va, te_features = (
        add_target_encoding(
            tr,
            va,
            smoothing=TE_SMOOTHING
        )
    )

    features = (
        list(base_features)
        + ["Income_Freq_V8"]
        + te_features
    )

    # -------------------------------------------------------------------------
    # Additional group frequency features
    # -------------------------------------------------------------------------

    group_feature_names = []

    for group_name in groups:

        cols = GROUP_FREQ_CANDIDATES[
            group_name
        ]

        feature_name = (
            f"GF_{group_name}"
        )

        tr, va = add_group_frequency(
            tr,
            va,
            cols,
            feature_name
        )

        # Restore exact original order after merge.
        tr = (
            tr.sort_values("_row_order")
            .reset_index(drop=True)
        )

        va = (
            va.sort_values("_row_order")
            .reset_index(drop=True)
        )

        group_feature_names.append(
            feature_name
        )

    features += group_feature_names

    y_train = (
        tr[TARGET]
        .to_numpy()
    )

    y_valid = (
        va[TARGET]
        .to_numpy()
    )

    X_train = tr[features].copy()
    X_valid = va[features].copy()

    X_train, X_valid = (
        encode_categories(
            X_train,
            X_valid,
            features
        )
    )

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_valid = X_valid.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    model = build_model()

    start = time.time()

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_valid, y_valid)
        ],

        eval_metric="auc",

        callbacks=[
            lgb.early_stopping(
                150,
                verbose=False
            )
        ],
    )

    runtime = (
        time.time() - start
    ) / 60

    pred = model.predict_proba(
        X_valid
    )[:, 1]

    lgb_auc = roc_auc_score(
        y_valid,
        pred
    )

    # -------------------------------------------------------------------------
    # Triple
    # -------------------------------------------------------------------------

    triple_valid = triple_oof[
        valid_mask.to_numpy()
    ]

    triple_auc = roc_auc_score(
        y_valid,
        triple_valid
    )

    pearson = pearsonr(
        triple_valid,
        pred
    ).statistic

    spearman = spearmanr(
        triple_valid,
        pred
    ).statistic

    wt, wl, blend_auc = (
        evaluate_blend(
            y_valid,
            triple_valid,
            pred
        )
    )

    blend_gain = (
        blend_auc - triple_auc
    )

    print(
        f"LGB AUC       : {lgb_auc:.6f}"
    )

    print(
        f"Triple AUC    : {triple_auc:.6f}"
    )

    print(
        f"Pearson       : {pearson:.6f}"
    )

    print(
        f"Spearman      : {spearman:.6f}"
    )

    print(
        f"Best blend    : "
        f"Triple {wt:.3f} + "
        f"LGB {wl:.3f}"
    )

    print(
        f"Blend AUC     : {blend_auc:.6f}"
    )

    print(
        f"Blend gain    : {blend_gain:+.6f}"
    )

    print(
        f"Best iteration: "
        f"{model.best_iteration_}"
    )

    print(
        f"Runtime       : {runtime:.2f} min"
    )

    result = {
        "experiment":
            experiment,

        "fold":
            fold,

        "groups":
            "+".join(groups)
            if groups else "NONE",

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
            wt,

        "best_lgb_weight":
            wl,

        "best_blend_auc":
            blend_auc,

        "blend_gain":
            blend_gain,

        "best_iteration":
            model.best_iteration_,

        "runtime_min":
            runtime,
    }

    del (
        tr,
        va,
        X_train,
        X_valid,
        model,
        pred,
        triple_valid,
    )

    gc.collect()

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("LGB V8.1 GROUP FREQUENCY SCREENING")
    print("BASE = V7 + IncomeFreq + Basic TE")
    print("=" * 100)

    train = pd.read_parquet(
        TRAIN_PATH
    )

    train[TARGET] = normalize_target(
        train[TARGET]
    )

    train = add_top6_income_features(
        train
    )

    base_features = [
        c for c in train.columns
        if c not in [
            TARGET,
            FOLD_COL,
            ID_COL
        ]
    ]

    print(
        "\nV7 base feature count:",
        len(base_features)
    )

    if len(base_features) != 62:
        print(
            "WARNING: expected 62 V7 features."
        )

    triple_oof = load_triple_oof(
        len(train)
    )

    full_auc = roc_auc_score(
        train[TARGET],
        triple_oof
    )

    print(
        f"Triple TE full OOF: "
        f"{full_auc:.6f}"
    )

    if full_auc > 0.99:
        raise ValueError(
            "Possible target leakage."
        )

    # =========================================================================
    # Stage 1
    # =========================================================================

    experiments = [
        ("BASE_V81", [])
    ]

    for group_name in (
        GROUP_FREQ_CANDIDATES
    ):
        experiments.append(
            (
                f"GF_{group_name}",
                [group_name]
            )
        )

    all_results = []

    print()
    print("=" * 100)
    print("STAGE 1: INDIVIDUAL GROUP FREQUENCY")
    print("=" * 100)

    for experiment, groups in experiments:

        for fold in SCREEN_FOLDS:

            result = run_fold(
                train,
                triple_oof,
                base_features,
                fold,
                experiment,
                groups,
            )

            all_results.append(
                result
            )

    stage1 = pd.DataFrame(
        all_results
    )

    # =========================================================================
    # Stage 1 summary
    # =========================================================================

    summary1 = (
        stage1
        .groupby(
            "experiment",
            as_index=False
        )
        .agg(
            mean_lgb_auc=(
                "lgb_auc",
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
        )
    )

    base_row = summary1[
        summary1["experiment"]
        == "BASE_V81"
    ].iloc[0]

    summary1[
        "delta_lgb_vs_base"
    ] = (
        summary1["mean_lgb_auc"]
        - base_row["mean_lgb_auc"]
    )

    summary1[
        "delta_blend_vs_base"
    ] = (
        summary1["mean_blend_auc"]
        - base_row["mean_blend_auc"]
    )

    summary1 = summary1.sort_values(
        [
            "delta_blend_vs_base",
            "delta_lgb_vs_base",
        ],
        ascending=False
    )

    print()
    print("=" * 100)
    print("STAGE 1 SUMMARY")
    print("=" * 100)

    print(
        summary1.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # =========================================================================
    # Select TOP3 automatically
    # =========================================================================

    candidates = summary1[
        summary1["experiment"]
        != "BASE_V81"
    ].copy()

    # Rank primarily by ensemble contribution.
    candidates = candidates.sort_values(
        [
            "delta_blend_vs_base",
            "delta_lgb_vs_base",
        ],
        ascending=False
    )

    top3_experiments = (
        candidates
        .head(3)["experiment"]
        .tolist()
    )

    top3_groups = [
        name.replace("GF_", "", 1)
        for name in top3_experiments
    ]

    print()
    print(
        "Selected TOP3:",
        top3_groups
    )

    # =========================================================================
    # Stage 2 combinations
    # =========================================================================

    combo_experiments = []

    if len(top3_groups) >= 2:

        combo_experiments.append(
            (
                "TOP2_GROUP_FREQ",
                top3_groups[:2]
            )
        )

    if len(top3_groups) >= 3:

        combo_experiments.append(
            (
                "TOP3_GROUP_FREQ",
                top3_groups[:3]
            )
        )

    stage2_results = []

    print()
    print("=" * 100)
    print("STAGE 2: TOP COMBINATIONS")
    print("=" * 100)

    for experiment, groups in (
        combo_experiments
    ):

        for fold in SCREEN_FOLDS:

            result = run_fold(
                train,
                triple_oof,
                base_features,
                fold,
                experiment,
                groups,
            )

            stage2_results.append(
                result
            )

    # =========================================================================
    # Final
    # =========================================================================

    if stage2_results:

        stage2 = pd.DataFrame(
            stage2_results
        )

        final_results = pd.concat(
            [
                stage1,
                stage2
            ],
            ignore_index=True
        )

    else:

        final_results = stage1

    final_summary = (
        final_results
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

            mean_pearson=(
                "pearson",
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

    final_summary[
        "delta_lgb_vs_base"
    ] = (
        final_summary["mean_lgb_auc"]
        - base_row["mean_lgb_auc"]
    )

    final_summary[
        "delta_blend_vs_base"
    ] = (
        final_summary["mean_blend_auc"]
        - base_row["mean_blend_auc"]
    )

    final_summary = (
        final_summary
        .sort_values(
            [
                "delta_blend_vs_base",
                "delta_lgb_vs_base",
            ],
            ascending=False
        )
    )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        final_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    final_results.to_csv(
        OUTPUT_DIR
        / "all_results.csv",
        index=False
    )

    final_summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False
    )

    print()
    print(
        "Saved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()