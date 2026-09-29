from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.build_features_v3 import V3_FEATURES


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "processed" / "train_engineered_v3.parquet"
)

TARGET = "Will_Buy_EV"

SEED = 42

# 用一个较难 + 一个较容易的 fold 做快速筛选
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
# Parameter candidates
# =========================================================

# V3 baseline:
# num_leaves=31
# min_child_samples=40
# colsample_bytree=0.85
# reg_alpha=0.1
# reg_lambda=1.0

PARAM_SETS = [
    {
        "name": "baseline",
        "num_leaves": 31,
        "min_child_samples": 40,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # -----------------------------------------------------
    # Leaves
    # -----------------------------------------------------

    {
        "name": "leaves_20",
        "num_leaves": 20,
        "min_child_samples": 40,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "leaves_24",
        "num_leaves": 24,
        "min_child_samples": 40,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "leaves_40",
        "num_leaves": 40,
        "min_child_samples": 40,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "leaves_48",
        "num_leaves": 48,
        "min_child_samples": 40,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # -----------------------------------------------------
    # min_child_samples
    # -----------------------------------------------------

    {
        "name": "child_20",
        "num_leaves": 31,
        "min_child_samples": 20,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "child_70",
        "num_leaves": 31,
        "min_child_samples": 70,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "child_120",
        "num_leaves": 31,
        "min_child_samples": 120,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # -----------------------------------------------------
    # Feature subsampling
    # -----------------------------------------------------

    {
        "name": "colsample_070",
        "num_leaves": 31,
        "min_child_samples": 40,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "colsample_100",
        "num_leaves": 31,
        "min_child_samples": 40,
        "colsample_bytree": 1.00,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # -----------------------------------------------------
    # Regularization
    # -----------------------------------------------------

    {
        "name": "reg_light",
        "num_leaves": 31,
        "min_child_samples": 40,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 0.5,
    },

    {
        "name": "reg_medium",
        "num_leaves": 31,
        "min_child_samples": 40,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
    },

    {
        "name": "reg_strong",
        "num_leaves": 31,
        "min_child_samples": 40,
        "colsample_bytree": 0.85,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    },

    # -----------------------------------------------------
    # A few combined candidates
    # -----------------------------------------------------

    {
        "name": "leaves24_child70",
        "num_leaves": 24,
        "min_child_samples": 70,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "leaves40_child70",
        "num_leaves": 40,
        "min_child_samples": 70,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
]


# =========================================================
# Prepare categorical
# =========================================================

def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in CATEGORICAL_COLS:

        if col not in df.columns:
            raise ValueError(
                f"Missing categorical column: {col}"
            )

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .astype("category")
        )

    return df


# =========================================================
# Evaluate one parameter set
# =========================================================

def evaluate_params(
    X: pd.DataFrame,
    y: pd.Series,
    folds: np.ndarray,
    config: dict,
):

    scores = []
    best_iterations = []

    print("\n" + "=" * 70)
    print(f"Testing: {config['name']}")
    print("=" * 70)

    for fold in TUNE_FOLDS:

        train_idx = np.where(
            folds != fold
        )[0]

        valid_idx = np.where(
            folds == fold
        )[0]

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = lgb.LGBMClassifier(

            objective="binary",

            n_estimators=4000,
            learning_rate=0.03,

            num_leaves=config[
                "num_leaves"
            ],

            max_depth=-1,

            min_child_samples=config[
                "min_child_samples"
            ],

            subsample=0.85,
            subsample_freq=1,

            colsample_bytree=config[
                "colsample_bytree"
            ],

            reg_alpha=config[
                "reg_alpha"
            ],

            reg_lambda=config[
                "reg_lambda"
            ],

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

            categorical_feature=CATEGORICAL_COLS,

            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=150,
                    verbose=False,
                ),
            ],
        )

        pred = model.predict_proba(
            X_valid,
            num_iteration=model.best_iteration_,
        )[:, 1]

        auc = roc_auc_score(
            y_valid,
            pred,
        )

        scores.append(
            auc
        )

        best_iterations.append(
            model.best_iteration_
        )

        print(
            f"Fold {fold}: "
            f"AUC={auc:.6f} | "
            f"best_iter={model.best_iteration_}"
        )

    mean_auc = float(
        np.mean(scores)
    )

    std_auc = float(
        np.std(scores)
    )

    mean_iteration = float(
        np.mean(best_iterations)
    )

    print(
        f"Mean AUC: {mean_auc:.6f}"
    )

    return {
        "name": config["name"],

        "mean_auc": mean_auc,
        "std_auc": std_auc,

        "fold_0_auc": scores[0],
        "fold_2_auc": scores[1],

        "avg_best_iteration":
            mean_iteration,

        "num_leaves":
            config["num_leaves"],

        "min_child_samples":
            config["min_child_samples"],

        "colsample_bytree":
            config["colsample_bytree"],

        "reg_alpha":
            config["reg_alpha"],

        "reg_lambda":
            config["reg_lambda"],
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("LightGBM V4 Parameter Screening")
    print("=" * 70)

    print(
        f"\nTune folds: {TUNE_FOLDS}"
    )

    print(
        f"Parameter sets: {len(PARAM_SETS)}"
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    train = pd.read_parquet(
        TRAIN_PATH
    )

    print(
        "\nTrain shape:",
        train.shape
    )

    if TARGET not in train.columns:
        raise ValueError(
            f"Missing target: {TARGET}"
        )

    if "fold" not in train.columns:
        raise ValueError(
            "Missing fold."
        )

    missing = [
        col
        for col in V3_FEATURES
        if col not in train.columns
    ]

    if missing:
        raise ValueError(
            f"Missing V3 features: {missing}"
        )

    # -----------------------------------------------------
    # Prepare
    # -----------------------------------------------------

    train = prepare_categorical(
        train
    )

    X = train[
        V3_FEATURES
    ].copy()

    y = train[
        TARGET
    ].copy()

    folds = (
        train["fold"]
        .to_numpy()
    )

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    results = []

    start_time = time.time()

    for i, config in enumerate(
        PARAM_SETS,
        start=1,
    ):

        print(
            f"\n[{i}/{len(PARAM_SETS)}]"
        )

        result = evaluate_params(
            X,
            y,
            folds,
            config,
        )

        results.append(
            result
        )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "mean_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RANKING"
    )

    print(
        "=" * 70
    )

    display_cols = [
        "name",
        "mean_auc",
        "fold_0_auc",
        "fold_2_auc",
        "avg_best_iteration",
        "num_leaves",
        "min_child_samples",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ]

    print(
        results_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nTop 3:"
    )

    for i in range(
        min(
            3,
            len(results_df),
        )
    ):

        row = (
            results_df
            .iloc[i]
        )

        print(
            f"{i + 1}. "
            f"{row['name']} | "
            f"AUC={row['mean_auc']:.6f}"
        )

    # -----------------------------------------------------
    # Compare baseline
    # -----------------------------------------------------

    baseline_auc = float(
        results_df.loc[
            results_df["name"]
            == "baseline",
            "mean_auc",
        ].iloc[0]
    )

    best_auc = float(
        results_df.iloc[0][
            "mean_auc"
        ]
    )

    print(
        "\nBest improvement "
        "over screening baseline:"
    )

    print(
        f"{best_auc - baseline_auc:+.6f}"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = (
        ROOT
        / "logs"
        / "lgb_v4_tuning.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
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
        f"\nRuntime: "
        f"{elapsed / 60:.1f} minutes"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()