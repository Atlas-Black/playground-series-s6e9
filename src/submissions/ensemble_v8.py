from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_TRAIN_PATH = (
    ROOT / "data" / "raw" / "train.csv"
)

SAMPLE_PATH = (
    ROOT / "data" / "raw" / "sample_submission.csv"
)

# ------------------------------------------------------------
# OOF
# ------------------------------------------------------------

XGB_OOF_PATH = (
    ROOT / "predictions" / "oof" / "xgb_v5.csv"
)

LGB_OOF_PATH = (
    ROOT / "outputs" / "lgb_v7" / "oof_lgb_v7.csv"
)

# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

XGB_TEST_PATH = (
    ROOT / "outputs" / "xgb_v5" / "test_xgb_v5.csv"
)

LGB_TEST_PATH = (
    ROOT / "outputs" / "lgb_v7" / "test_lgb_v7.csv"
)

TARGET = "Will_Buy_EV"
ID_COL = "id"

OUTPUT_DIR = (
    ROOT / "outputs" / "ensemble_v8"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Search every 1%
WEIGHTS = np.arange(
    0.0,
    1.0001,
    0.01,
)


# ============================================================
# TARGET NORMALIZATION
# ============================================================

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
            "Unknown target values: "
            f"{series[result.isna()].unique()}"
        )

    return result.astype("int8")


# ============================================================
# LOAD PREDICTION
# ============================================================

def load_prediction(
    path,
    expected_length,
    name,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_csv(path)

    if len(df) != expected_length:

        raise ValueError(
            f"{name}: expected "
            f"{expected_length} rows, "
            f"found {len(df)}"
        )

    # --------------------------------------------------------
    # Find prediction column
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Find prediction column
    # --------------------------------------------------------

    preferred_prediction_columns = [
        "prediction",
        "pred",
        "probability",
        "prob",
        "oof_prediction",
        "oof_pred",
    ]

    pred_col = None

    # 优先寻找明确的预测列
    for col in preferred_prediction_columns:
        if col in df.columns:
            pred_col = col
            break

    # 如果没有，再考虑 TARGET
    # 这种情况适用于 xgb_v5.csv：
    # ['id', 'Will_Buy_EV']
    if pred_col is None:

        non_id_columns = [
            col
            for col in df.columns
            if col != ID_COL
        ]

        if len(non_id_columns) == 1:
            pred_col = non_id_columns[0]

        else:
            raise ValueError(
                f"Cannot safely determine prediction column "
                f"for {name}.\n"
                f"Columns = {list(df.columns)}"
            )

    else:

        candidates = [
            col
            for col in df.columns
            if col.lower() in [
                "prediction",
                "pred",
                "probability",
                "prob",
            ]
        ]

        if len(candidates) == 1:

            pred_col = candidates[0]

        else:

            non_id = [
                col
                for col in df.columns
                if col != ID_COL
            ]

            if len(non_id) != 1:

                raise ValueError(
                    f"Cannot determine prediction "
                    f"column for {name}. "
                    f"Columns={list(df.columns)}"
                )

            pred_col = non_id[0]

    pred = (
        pd.to_numeric(
            df[pred_col],
            errors="raise",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(pred).all():

        raise ValueError(
            f"{name} contains "
            "non-finite predictions."
        )

    print(
        f"Loaded {name:<10} "
        f"rows={len(pred)} | "
        f"column={pred_col}"
    )

    return pred


# ============================================================
# RANK NORMALIZATION
# ============================================================

def rank_normalize(pred):

    ranks = rankdata(
        pred,
        method="average",
    )

    return (
        ranks - 1
    ) / (
        len(ranks) - 1
    )


# ============================================================
# PROBABILITY SEARCH
# weight = XGB weight
# ============================================================

def search_probability_blend(
    y,
    xgb_pred,
    lgb_pred,
):

    results = []

    for xgb_weight in WEIGHTS:

        lgb_weight = (
            1.0 - xgb_weight
        )

        blend = (
            xgb_weight * xgb_pred
            +
            lgb_weight * lgb_pred
        )

        auc = roc_auc_score(
            y,
            blend,
        )

        results.append(
            {
                "xgb_weight":
                    xgb_weight,

                "lgb_weight":
                    lgb_weight,

                "auc":
                    auc,
            }
        )

    df = pd.DataFrame(
        results
    )

    df = df.sort_values(
        "auc",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# RANK SEARCH
# weight = XGB weight
# ============================================================

def search_rank_blend(
    y,
    xgb_pred,
    lgb_pred,
):

    xgb_rank = rank_normalize(
        xgb_pred
    )

    lgb_rank = rank_normalize(
        lgb_pred
    )

    results = []

    for xgb_weight in WEIGHTS:

        lgb_weight = (
            1.0 - xgb_weight
        )

        blend = (
            xgb_weight * xgb_rank
            +
            lgb_weight * lgb_rank
        )

        auc = roc_auc_score(
            y,
            blend,
        )

        results.append(
            {
                "xgb_weight":
                    xgb_weight,

                "lgb_weight":
                    lgb_weight,

                "auc":
                    auc,
            }
        )

    df = pd.DataFrame(
        results
    )

    df = df.sort_values(
        "auc",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "ENSEMBLE V8"
    )

    print(
        "XGB V5 + LGB V7"
    )

    print(
        "=" * 100
    )

    # ========================================================
    # LOAD TARGET
    # ========================================================

    raw_train = pd.read_csv(
        RAW_TRAIN_PATH,
        usecols=[
            ID_COL,
            TARGET,
        ],
    )

    y = normalize_target(
        raw_train[TARGET]
    ).to_numpy()

    n_train = len(
        raw_train
    )

    print(
        "\nTrain rows:",
        n_train,
    )

    # ========================================================
    # LOAD OOF
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LOADING OOF"
    )

    print(
        "=" * 100
    )

    xgb_oof = load_prediction(
        XGB_OOF_PATH,
        n_train,
        "XGB_V5",
    )

    lgb_oof = load_prediction(
        LGB_OOF_PATH,
        n_train,
        "LGB_V7",
    )

    # ========================================================
    # INDIVIDUAL AUC
    # ========================================================

    xgb_auc = roc_auc_score(
        y,
        xgb_oof,
    )

    lgb_auc = roc_auc_score(
        y,
        lgb_oof,
    )

    if xgb_auc > 0.99:
        raise ValueError(
            f"Suspicious XGB OOF AUC={xgb_auc:.6f}. "
            "Possible target leakage / wrong prediction column."
        )

    if lgb_auc > 0.99:
        raise ValueError(
            f"Suspicious LGB OOF AUC={lgb_auc:.6f}. "
            "Possible target leakage / wrong prediction column."
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "INDIVIDUAL OOF AUC"
    )

    print(
        "=" * 100
    )

    print(
        f"XGB V5 : {xgb_auc:.6f}"
    )

    print(
        f"LGB V7 : {lgb_auc:.6f}"
    )

    # ========================================================
    # CORRELATION
    # ========================================================

    pearson_corr = pearsonr(
        xgb_oof,
        lgb_oof,
    ).statistic

    spearman_corr = spearmanr(
        xgb_oof,
        lgb_oof,
    ).statistic

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CORRELATION"
    )

    print(
        "=" * 100
    )

    print(
        f"Pearson : "
        f"{pearson_corr:.6f}"
    )

    print(
        f"Spearman: "
        f"{spearman_corr:.6f}"
    )

    # ========================================================
    # PROBABILITY BLEND
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PROBABILITY BLEND SEARCH"
    )

    print(
        "=" * 100
    )

    prob_results = (
        search_probability_blend(
            y,
            xgb_oof,
            lgb_oof,
        )
    )

    print(
        prob_results.head(15).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    best_prob = (
        prob_results.iloc[0]
    )

    prob_gain = (
        best_prob["auc"]
        - xgb_auc
    )

    # ========================================================
    # RANK BLEND
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RANK BLEND SEARCH"
    )

    print(
        "=" * 100
    )

    rank_results = (
        search_rank_blend(
            y,
            xgb_oof,
            lgb_oof,
        )
    )

    print(
        rank_results.head(15).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    best_rank = (
        rank_results.iloc[0]
    )

    rank_gain = (
        best_rank["auc"]
        - xgb_auc
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL V8 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"XGB V5 AUC          : "
        f"{xgb_auc:.6f}"
    )

    print(
        f"LGB V7 AUC          : "
        f"{lgb_auc:.6f}"
    )

    print()

    print(
        f"Pearson correlation : "
        f"{pearson_corr:.6f}"
    )

    print(
        f"Spearman correlation: "
        f"{spearman_corr:.6f}"
    )

    print()

    print(
        "Best probability blend:"
    )

    print(
        f"  XGB weight : "
        f"{best_prob['xgb_weight']:.2f}"
    )

    print(
        f"  LGB weight : "
        f"{best_prob['lgb_weight']:.2f}"
    )

    print(
        f"  OOF AUC    : "
        f"{best_prob['auc']:.6f}"
    )

    print(
        f"  Gain vs XGB: "
        f"{prob_gain:+.6f}"
    )

    print()

    print(
        "Best rank blend:"
    )

    print(
        f"  XGB weight : "
        f"{best_rank['xgb_weight']:.2f}"
    )

    print(
        f"  LGB weight : "
        f"{best_rank['lgb_weight']:.2f}"
    )

    print(
        f"  OOF AUC    : "
        f"{best_rank['auc']:.6f}"
    )

    print(
        f"  Gain vs XGB: "
        f"{rank_gain:+.6f}"
    )

    # ========================================================
    # SAVE SEARCH REPORTS
    # ========================================================

    prob_path = (
        OUTPUT_DIR
        / "probability_search.csv"
    )

    rank_path = (
        OUTPUT_DIR
        / "rank_search.csv"
    )

    prob_results.to_csv(
        prob_path,
        index=False,
    )

    rank_results.to_csv(
        rank_path,
        index=False,
    )

    # ========================================================
    # CHOOSE BEST METHOD
    # ========================================================

    if (
        best_rank["auc"]
        >
        best_prob["auc"]
    ):

        best_method = "rank"

        best_xgb_weight = float(
            best_rank["xgb_weight"]
        )

        best_lgb_weight = float(
            best_rank["lgb_weight"]
        )

        best_auc = float(
            best_rank["auc"]
        )

    else:

        best_method = "probability"

        best_xgb_weight = float(
            best_prob["xgb_weight"]
        )

        best_lgb_weight = float(
            best_prob["lgb_weight"]
        )

        best_auc = float(
            best_prob["auc"]
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SELECTED ENSEMBLE"
    )

    print(
        "=" * 100
    )

    print(
        "Method     :",
        best_method,
    )

    print(
        f"XGB weight : "
        f"{best_xgb_weight:.2f}"
    )

    print(
        f"LGB weight : "
        f"{best_lgb_weight:.2f}"
    )

    print(
        f"OOF AUC    : "
        f"{best_auc:.6f}"
    )

    print(
        f"Gain vs XGB: "
        f"{best_auc - xgb_auc:+.6f}"
    )

    # ========================================================
    # LOAD TEST
    # ========================================================

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    n_test = len(
        sample
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LOADING TEST PREDICTIONS"
    )

    print(
        "=" * 100
    )

    xgb_test = load_prediction(
        XGB_TEST_PATH,
        n_test,
        "XGB_V5_TEST",
    )

    lgb_test = load_prediction(
        LGB_TEST_PATH,
        n_test,
        "LGB_V7_TEST",
    )

    # ========================================================
    # BUILD SELECTED TEST BLEND
    # ========================================================

    if best_method == "rank":

        xgb_test_component = (
            rank_normalize(
                xgb_test
            )
        )

        lgb_test_component = (
            rank_normalize(
                lgb_test
            )
        )

        ensemble_test = (
            best_xgb_weight
            * xgb_test_component
            +
            best_lgb_weight
            * lgb_test_component
        )

    else:

        ensemble_test = (
            best_xgb_weight
            * xgb_test
            +
            best_lgb_weight
            * lgb_test
        )

    # ========================================================
    # SUBMISSION
    # ========================================================

    submission = (
        sample.copy()
    )

    submission[TARGET] = (
        ensemble_test
    )

    submission_path = (
        OUTPUT_DIR
        / "submission_ensemble_v8.csv"
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    # ========================================================
    # SAVE OOF BLEND TOO
    # ========================================================

    if best_method == "rank":

        ensemble_oof = (
            best_xgb_weight
            * rank_normalize(
                xgb_oof
            )
            +
            best_lgb_weight
            * rank_normalize(
                lgb_oof
            )
        )

    else:

        ensemble_oof = (
            best_xgb_weight
            * xgb_oof
            +
            best_lgb_weight
            * lgb_oof
        )

    ensemble_oof_path = (
        OUTPUT_DIR
        / "oof_ensemble_v8.csv"
    )

    ensemble_oof_df = pd.DataFrame(
        {
            ID_COL:
                raw_train[
                    ID_COL
                ].to_numpy(),

            TARGET:
                ensemble_oof,
        }
    )

    ensemble_oof_df.to_csv(
        ensemble_oof_path,
        index=False,
    )

    # ========================================================
    # SUMMARY FILE
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "xgb_v5_auc":
                    xgb_auc,

                "lgb_v7_auc":
                    lgb_auc,

                "pearson":
                    pearson_corr,

                "spearman":
                    spearman_corr,

                "best_method":
                    best_method,

                "xgb_weight":
                    best_xgb_weight,

                "lgb_weight":
                    best_lgb_weight,

                "ensemble_auc":
                    best_auc,

                "gain_vs_xgb":
                    best_auc
                    - xgb_auc,
            }
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # OUTPUT
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
        "Probability search:",
        prob_path,
    )

    print(
        "Rank search       :",
        rank_path,
    )

    print(
        "OOF ensemble      :",
        ensemble_oof_path,
    )

    print(
        "Submission        :",
        submission_path,
    )

    print(
        "Summary           :",
        summary_path,
    )

    print()

    print(
        "Submission shape:",
        submission.shape,
    )

    print(
        "Prediction min:",
        float(
            ensemble_test.min()
        ),
    )

    print(
        "Prediction max:",
        float(
            ensemble_test.max()
        ),
    )

    print(
        "Prediction mean:",
        float(
            ensemble_test.mean()
        ),
    )

    print(
        "\nDONE."
    )


if __name__ == "__main__":
    main()