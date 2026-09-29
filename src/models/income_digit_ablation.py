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
# Income digit features
# =========================================================

DIGIT_FEATURES = [
    "Income_Ones",
    "Income_Tens",
    "Income_Hundreds",
    "Income_Thousands",
    "Income_TenThousands",
]


# =========================================================
# Prepare categorical
# =========================================================

def prepare_categorical(df):

    df = df.copy()

    for col in CATEGORICAL_COLS:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .astype("category")
        )

    return df


# =========================================================
# Evaluate one feature set
# =========================================================

def evaluate_feature_set(
    train,
    features,
    experiment_name,
):

    folds = train["fold"].to_numpy()
    y = train[TARGET]

    scores = []
    iterations = []

    print("\n" + "=" * 70)
    print(experiment_name)
    print("=" * 70)

    print(
        "Number of features:",
        len(features)
    )

    for fold in TUNE_FOLDS:

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

        y_train = (
            y.iloc[train_idx]
        )

        X_valid = (
            train
            .iloc[valid_idx][features]
        )

        y_valid = (
            y.iloc[valid_idx]
        )

        active_cat_cols = [
            col
            for col in CATEGORICAL_COLS
            if col in features
        ]

        # =================================================
        # LGB V4 parameters
        # =================================================

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

            categorical_feature=active_cat_cols,

            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=150,
                    verbose=False,
                )
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

        iterations.append(
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

    print(
        f"Mean AUC: {mean_auc:.6f}"
    )

    return {
        "experiment": experiment_name,
        "n_features": len(features),
        "fold_0_auc": scores[0],
        "fold_2_auc": scores[1],
        "mean_auc": mean_auc,
        "avg_best_iteration": float(
            np.mean(iterations)
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("Income Digit Feature Ablation")
    print("=" * 70)

    print(
        "Tune folds:",
        TUNE_FOLDS
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

    # -----------------------------------------------------
    # Checks
    # -----------------------------------------------------

    if TARGET not in train.columns:
        raise ValueError(
            f"Missing target: {TARGET}"
        )

    if "fold" not in train.columns:
        raise ValueError(
            "Missing fold column."
        )

    for feature in DIGIT_FEATURES:

        if feature not in V3_FEATURES:

            raise ValueError(
                f"{feature} is not "
                f"in V3_FEATURES"
            )

    # -----------------------------------------------------
    # Prepare categorical
    # -----------------------------------------------------

    train = prepare_categorical(
        train
    )

    results = []

    start_time = time.time()

    # =====================================================
    # Baseline: all V3 features
    # =====================================================

    baseline_result = (
        evaluate_feature_set(
            train,
            V3_FEATURES,
            "ALL_V3",
        )
    )

    results.append(
        baseline_result
    )

    baseline_auc = (
        baseline_result[
            "mean_auc"
        ]
    )

    # =====================================================
    # Remove ONE digit at a time
    # =====================================================

    for digit_feature in DIGIT_FEATURES:

        features = [
            feature
            for feature in V3_FEATURES
            if feature != digit_feature
        ]

        result = (
            evaluate_feature_set(
                train,
                features,
                f"REMOVE_{digit_feature}",
            )
        )

        result["removed_feature"] = (
            digit_feature
        )

        result["delta_vs_all"] = (
            result["mean_auc"]
            - baseline_auc
        )

        results.append(
            result
        )

    # =====================================================
    # Remove all digits as sanity check
    # =====================================================

    no_digits = [
        feature
        for feature in V3_FEATURES
        if feature not in DIGIT_FEATURES
    ]

    result = (
        evaluate_feature_set(
            train,
            no_digits,
            "REMOVE_ALL_DIGITS",
        )
    )

    result["removed_feature"] = (
        "ALL_DIGITS"
    )

    result["delta_vs_all"] = (
        result["mean_auc"]
        - baseline_auc
    )

    results.append(
        result
    )

    # =====================================================
    # Results
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.loc[
        results_df["experiment"]
        == "ALL_V3",
        "delta_vs_all",
    ] = 0.0

    # 最负 = 删除后伤害最大 = 最重要
    results_df = (
        results_df
        .sort_values(
            "delta_vs_all",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n"
        + "=" * 85
    )

    print(
        "INCOME DIGIT ABLATION RANKING"
    )

    print(
        "=" * 85
    )

    print(
        results_df[
            [
                "experiment",
                "n_features",
                "fold_0_auc",
                "fold_2_auc",
                "mean_auc",
                "delta_vs_all",
                "avg_best_iteration",
            ]
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # Individual digit importance
    # =====================================================

    print(
        "\n"
        + "=" * 85
    )

    print(
        "INDIVIDUAL DIGIT IMPORTANCE"
    )

    print(
        "=" * 85
    )

    individual = results_df[
        results_df[
            "removed_feature"
        ].isin(
            DIGIT_FEATURES
        )
    ].copy()

    # importance_loss > 0 means removing feature hurts
    individual[
        "importance_loss"
    ] = (
        -individual[
            "delta_vs_all"
        ]
    )

    individual = (
        individual
        .sort_values(
            "importance_loss",
            ascending=False,
        )
    )

    for _, row in individual.iterrows():

        feature = row[
            "removed_feature"
        ]

        loss = row[
            "importance_loss"
        ]

        if loss > 0.00030:

            verdict = (
                "VERY IMPORTANT"
            )

        elif loss > 0.00010:

            verdict = (
                "IMPORTANT"
            )

        elif loss > 0:

            verdict = (
                "slightly useful"
            )

        else:

            verdict = (
                "possibly unnecessary"
            )

        print(
            f"{feature:<25} "
            f"loss={loss:+.6f} | "
            f"{verdict}"
        )

    # =====================================================
    # Save
    # =====================================================

    output_path = (
        ROOT
        / "logs"
        / "income_digit_ablation.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
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
        "=" * 85
    )


if __name__ == "__main__":
    main()