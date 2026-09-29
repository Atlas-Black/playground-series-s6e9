from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TARGET = "Will_Buy_EV"

V7_PATH = (
    ROOT
    / "outputs"
    / "lgb_v7"
    / "oof_lgb_v7.csv"
)

OOF_DIR = ROOT / "predictions" / "oof"

MODEL_PATHS = {
    "LGB_V5": OOF_DIR / "lgb_v5.csv",
    "LGB_V4": OOF_DIR / "lgb_v4.csv",
    "XGB_V4_TE": OOF_DIR / "xgb_v4_te.csv",
    "XGB_V3": OOF_DIR / "xgb_v3.csv",
    "CAT_FEAT_V5": OOF_DIR / "cat_feat_v5.csv",
}

# ============================================================
# HELPERS
# ============================================================

def detect_prediction_column(df):

    candidates = [
        "prediction",
        "pred",
        "oof",
        "probability",
        "prob",
        "Will_Buy_EV",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    # Last fallback:
    # choose numeric column excluding metadata
    excluded = {
        "id",
        "row_index",
        "fold",
        "target",
        TARGET,
    }

    numeric_cols = [
        col
        for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        "Could not identify prediction column. "
        f"Columns: {df.columns.tolist()}"
    )


def normalize_target(series):

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()

    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
    }

    mapped = cleaned.map(mapping)

    if mapped.isna().any():

        raise ValueError(
            "Unknown target values: "
            f"{series[mapped.isna()].unique()}"
        )

    return mapped.astype(int).to_numpy()


def to_rank(pred):

    ranks = rankdata(
        pred,
        method="average",
    )

    return (
        ranks - 1
    ) / max(
        len(ranks) - 1,
        1,
    )


def auc(y, pred):

    return roc_auc_score(
        y,
        pred,
    )


# ============================================================
# LOAD V7
# ============================================================

def load_v7():

    if not V7_PATH.exists():

        raise FileNotFoundError(
            f"V7 OOF not found:\n{V7_PATH}"
        )

    df = pd.read_csv(
        V7_PATH
    )

    if TARGET not in df.columns:

        raise ValueError(
            f"{TARGET} not found in V7 OOF."
        )

    pred_col = detect_prediction_column(
        df
    )

    y = normalize_target(
        df[TARGET]
    )

    pred = (
        df[pred_col]
        .to_numpy(dtype=float)
    )

    return df, y, pred


# ============================================================
# LOAD OLD MODELS
# ============================================================

def load_old_models(expected_length):

    predictions = {}
    paths = {}

    print("\nLoading old OOF predictions...")

    for model_name, path in MODEL_PATHS.items():

        if not path.exists():

            print(
                f"[NOT FOUND] {model_name}: {path}"
            )

            continue

        df = pd.read_csv(path)

        pred_col = detect_prediction_column(df)

        pred = df[pred_col].to_numpy(
            dtype=float
        )

        if len(pred) != expected_length:

            print(
                f"[SKIP] {model_name}: "
                f"length={len(pred)}, "
                f"expected={expected_length}"
            )

            continue

        predictions[model_name] = pred
        paths[model_name] = path

        print(
            f"[FOUND] {model_name:<12} "
            f"rows={len(pred)} "
            f"-> {path}"
        )

    return predictions, paths

# ============================================================
# TWO MODEL SEARCH
# ============================================================

def two_model_search(
    y,
    base_pred,
    candidate_pred,
    rank=False,
):

    if rank:

        p1 = to_rank(
            base_pred
        )

        p2 = to_rank(
            candidate_pred
        )

    else:

        p1 = base_pred
        p2 = candidate_pred

    rows = []

    # Candidate receives 0% ... 50%
    # Fine grid = 1%
    for candidate_weight in (
        np.arange(
            0.0,
            0.501,
            0.01,
        )
    ):

        base_weight = (
            1.0
            - candidate_weight
        )

        blend = (
            base_weight * p1
            + candidate_weight * p2
        )

        score = auc(
            y,
            blend,
        )

        rows.append({
            "candidate_weight":
                candidate_weight,

            "v7_weight":
                base_weight,

            "auc":
                score,
        })

    result = pd.DataFrame(
        rows
    )

    return (
        result
        .sort_values(
            "auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# GREEDY MULTI MODEL SEARCH
# ============================================================

def greedy_rank_blend(
    y,
    predictions,
    start_model="LGB_V7",
):

    ranked = {
        name: to_rank(pred)
        for name, pred
        in predictions.items()
    }

    current = (
        ranked[start_model]
        .copy()
    )

    current_auc = auc(
        y,
        current,
    )

    selected = [
        start_model
    ]

    history = []

    remaining = [
        name
        for name in ranked
        if name != start_model
    ]

    print(
        "\nGreedy rank ensemble"
    )

    print(
        f"Start {start_model}: "
        f"{current_auc:.6f}"
    )

    while remaining:

        best_candidate = None
        best_auc = current_auc
        best_weight = None
        best_pred = None

        for name in remaining:

            candidate = (
                ranked[name]
            )

            # Add candidate with up to 30%
            for w in np.arange(
                0.02,
                0.301,
                0.02,
            ):

                blend = (
                    (1 - w) * current
                    + w * candidate
                )

                score = auc(
                    y,
                    blend,
                )

                if score > best_auc:

                    best_auc = score
                    best_candidate = name
                    best_weight = w
                    best_pred = blend

        if best_candidate is None:
            break

        gain = (
            best_auc
            - current_auc
        )

        # Don't add meaningless noise
        if gain < 0.000005:
            break

        history.append({
            "added_model":
                best_candidate,

            "weight_at_step":
                best_weight,

            "auc":
                best_auc,

            "gain":
                gain,
        })

        print(
            f"+ {best_candidate:<12} "
            f"w={best_weight:.2f} | "
            f"AUC={best_auc:.6f} | "
            f"gain={gain:+.6f}"
        )

        current = best_pred
        current_auc = best_auc

        selected.append(
            best_candidate
        )

        remaining.remove(
            best_candidate
        )

    return (
        current,
        current_auc,
        selected,
        pd.DataFrame(history),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("ENSEMBLE V7")
    print("=" * 100)

    # --------------------------------------------------------
    # V7
    # --------------------------------------------------------

    v7_df, y, v7_pred = (
        load_v7()
    )

    print(
        "\nLoaded V7:"
    )

    print(
        "Rows:",
        len(v7_pred)
    )

    print(
        "OOF AUC:",
        f"{auc(y, v7_pred):.6f}"
    )

    # --------------------------------------------------------
    # Old models
    # --------------------------------------------------------

    old_predictions, paths = (
        load_old_models(
            expected_length=len(y)
        )
    )

    predictions = {
        "LGB_V7": v7_pred,
        **old_predictions,
    }

    print(
        "\nModels available:",
        list(predictions.keys())
    )

    # --------------------------------------------------------
    # Individual AUC
    # --------------------------------------------------------

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

    individual_rows = []

    for name, pred in (
        predictions.items()
    ):

        score = auc(
            y,
            pred,
        )

        individual_rows.append({
            "model": name,
            "auc": score,
        })

    individual_df = (
        pd.DataFrame(
            individual_rows
        )
        .sort_values(
            "auc",
            ascending=False,
        )
    )

    print(
        individual_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------------

    pred_df = pd.DataFrame(
        predictions
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PEARSON CORRELATION"
    )

    print(
        "=" * 100
    )

    print(
        pred_df.corr(
            method="pearson"
        ).round(6)
    )

    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SPEARMAN / RANK CORRELATION"
    )

    print(
        "=" * 100
    )

    print(
        pred_df.corr(
            method="spearman"
        ).round(6)
    )

    # --------------------------------------------------------
    # V7 + each model
    # --------------------------------------------------------

    pair_rows = []

    print(
        "\n"
        + "=" * 100
    )

    print(
        "V7 + EACH MODEL"
    )

    print(
        "=" * 100
    )

    base_auc = auc(
        y,
        v7_pred,
    )

    for name, pred in (
        old_predictions.items()
    ):

        # Probability blend
        prob_result = (
            two_model_search(
                y,
                v7_pred,
                pred,
                rank=False,
            )
        )

        best_prob = (
            prob_result.iloc[0]
        )

        # Rank blend
        rank_result = (
            two_model_search(
                y,
                v7_pred,
                pred,
                rank=True,
            )
        )

        best_rank = (
            rank_result.iloc[0]
        )

        print(
            f"\n{name}"
        )

        print(
            "Probability:"
            f" AUC={best_prob['auc']:.6f}"
            f" | old_w="
            f"{best_prob['candidate_weight']:.2f}"
            f" | gain="
            f"{best_prob['auc'] - base_auc:+.6f}"
        )

        print(
            "Rank       :"
            f" AUC={best_rank['auc']:.6f}"
            f" | old_w="
            f"{best_rank['candidate_weight']:.2f}"
            f" | gain="
            f"{best_rank['auc'] - base_auc:+.6f}"
        )

        pair_rows.append({
            "candidate":
                name,

            "prob_auc":
                best_prob["auc"],

            "prob_old_weight":
                best_prob[
                    "candidate_weight"
                ],

            "prob_gain":
                best_prob["auc"]
                - base_auc,

            "rank_auc":
                best_rank["auc"],

            "rank_old_weight":
                best_rank[
                    "candidate_weight"
                ],

            "rank_gain":
                best_rank["auc"]
                - base_auc,
        })

    pair_df = (
        pd.DataFrame(
            pair_rows
        )
    )

    if not pair_df.empty:

        pair_df = (
            pair_df
            .sort_values(
                "rank_auc",
                ascending=False,
            )
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            "PAIR SUMMARY"
        )

        print(
            "=" * 100
        )

        print(
            pair_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

    # --------------------------------------------------------
    # Greedy rank blend
    # --------------------------------------------------------

    (
        greedy_pred,
        greedy_auc,
        selected,
        history,
    ) = greedy_rank_blend(
        y=y,
        predictions=predictions,
        start_model="LGB_V7",
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL GREEDY RANK RESULT"
    )

    print(
        "=" * 100
    )

    print(
        "Selected:",
        selected,
    )

    print(
        f"V7 AUC       : "
        f"{base_auc:.6f}"
    )

    print(
        f"Ensemble AUC : "
        f"{greedy_auc:.6f}"
    )

    print(
        f"Gain         : "
        f"{greedy_auc - base_auc:+.6f}"
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "outputs"
        / "ensemble_v7"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    individual_df.to_csv(
        output_dir
        / "individual_auc.csv",
        index=False,
    )

    pred_df.corr(
        method="pearson"
    ).to_csv(
        output_dir
        / "pearson_correlation.csv"
    )

    pred_df.corr(
        method="spearman"
    ).to_csv(
        output_dir
        / "spearman_correlation.csv"
    )

    if not pair_df.empty:

        pair_df.to_csv(
            output_dir
            / "pair_search.csv",
            index=False,
        )

    history.to_csv(
        output_dir
        / "greedy_rank_history.csv",
        index=False,
    )

    print(
        "\nSaved reports to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()