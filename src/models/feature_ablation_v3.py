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

# 和上一轮参数筛选保持一致
TUNE_FOLDS = [0, 2]


# =========================================================
# Categorical features
# =========================================================

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
# Feature groups
# =========================================================

FEATURE_GROUPS = {

    # -----------------------------------------------------
    # V1 engineered
    # -----------------------------------------------------

    "V1_ENGINEERED": [
        "Total_Charging_Stations",
        "Charging_Per_Commute",
        "Has_Home_Charging",
        "Income_Per_Age",
        "Range_Anxiety_Num",
        "Anxiety_Commute_Ratio",
        "Has_Multiple_Cars",
        "Subsidy_Flag",
        "Eco_Subsidy_Score",
    ],

    # -----------------------------------------------------
    # V2 interaction features
    # -----------------------------------------------------

    "V2_INTERACTIONS": [
        "Income_Per_Car",
        "Income_Subsidy_Interaction",
        "HomeCharging_Commute",
        "Charging_Anxiety_Interaction",
        "Commute_Charging_Pressure",
        "Eco_Income_Interaction",
        "Eco_HomeCharging_Score",
    ],

    # -----------------------------------------------------
    # Income digit decomposition
    # -----------------------------------------------------

    "INCOME_DIGITS": [
        "Income_Ones",
        "Income_Tens",
        "Income_Hundreds",
        "Income_Thousands",
        "Income_TenThousands",
    ],

    # -----------------------------------------------------
    # Income logarithm
    # -----------------------------------------------------

    "INCOME_LOG": [
        "Income_Log",
    ],

    # -----------------------------------------------------
    # Income bins
    # -----------------------------------------------------

    "INCOME_BINS": [
        "Income_1k_Bin",
        "Income_5k_Bin",
        "Income_10k_Bin",
    ],

    # -----------------------------------------------------
    # Frequency encoding
    # -----------------------------------------------------

    "FREQUENCY": [
        "Gender_Freq",
        "City_Type_Freq",
        "Current_Car_Type_Freq",
        "Home_Charging_Possible_Freq",
        "Subsidy_Available_Freq",
        "Range_Anxiety_Level_Freq",
        "Number_of_Cars_Owned_Freq",
        "Charging_Stations_Near_Home_Freq",
        "Charging_Stations_Near_Work_Freq",
        "Environmental_Concern_Level_Freq",
    ],
}


# =========================================================
# Prepare categorical
# =========================================================

def prepare_categorical(df):

    df = df.copy()

    for col in CATEGORICAL_COLS:

        values = (
            df[col]
            .astype(str)
            .str.strip()
        )

        df[col] = values.astype(
            "category"
        )

    return df


# =========================================================
# Evaluate feature set
# =========================================================

def evaluate_feature_set(
    train,
    features,
    name,
):

    folds = train["fold"].to_numpy()

    y = train[TARGET]

    scores = []
    best_iterations = []

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Number of features: {len(features)}"
    )

    for fold in TUNE_FOLDS:

        train_idx = np.where(
            folds != fold
        )[0]

        valid_idx = np.where(
            folds == fold
        )[0]

        X_train = (
            train.iloc[train_idx][features]
        )

        y_train = (
            y.iloc[train_idx]
        )

        X_valid = (
            train.iloc[valid_idx][features]
        )

        y_valid = (
            y.iloc[valid_idx]
        )

        # ---------------------------------------------
        # Only categorical columns that still exist
        # ---------------------------------------------

        active_cat_cols = [
            col
            for col in CATEGORICAL_COLS
            if col in features
        ]

        # ---------------------------------------------
        # LGB V4 parameters
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

            # LGB V4 reg_light
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

        scores.append(auc)

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

    print(
        f"Mean AUC: {mean_auc:.6f}"
    )

    return {
        "experiment": name,
        "n_features": len(features),
        "fold_0_auc": scores[0],
        "fold_2_auc": scores[1],
        "mean_auc": mean_auc,
        "avg_best_iteration": float(
            np.mean(best_iterations)
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("V3 Feature Group Ablation")
    print("=" * 70)

    print(
        f"Tune folds: {TUNE_FOLDS}"
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
            "Missing fold column."
        )

    missing_features = [
        f
        for f in V3_FEATURES
        if f not in train.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing V3 features: "
            f"{missing_features}"
        )

    # -----------------------------------------------------
    # Verify groups
    # -----------------------------------------------------

    grouped_features = []

    for group_name, features in FEATURE_GROUPS.items():

        print(
            f"\n{group_name}: "
            f"{len(features)} features"
        )

        for feature in features:

            if feature not in V3_FEATURES:

                raise ValueError(
                    f"{feature} from "
                    f"{group_name} "
                    f"is not in V3_FEATURES"
                )

            grouped_features.append(
                feature
            )

    duplicates = (
        pd.Series(grouped_features)
        .value_counts()
    )

    duplicates = duplicates[
        duplicates > 1
    ]

    if len(duplicates) > 0:

        raise ValueError(
            "Features appear in multiple groups:\n"
            + str(duplicates)
        )

    # -----------------------------------------------------
    # Prepare categories
    # -----------------------------------------------------

    train = prepare_categorical(
        train
    )

    results = []

    start_time = time.time()

    # =====================================================
    # Baseline
    # =====================================================

    baseline_result = (
        evaluate_feature_set(
            train=train,
            features=V3_FEATURES,
            name="ALL_V3",
        )
    )

    results.append(
        baseline_result
    )

    baseline_auc = (
        baseline_result["mean_auc"]
    )

    # =====================================================
    # Remove each feature group
    # =====================================================

    for group_name, group_features in FEATURE_GROUPS.items():

        remaining_features = [
            feature
            for feature in V3_FEATURES
            if feature not in group_features
        ]

        result = evaluate_feature_set(
            train=train,
            features=remaining_features,
            name=f"REMOVE_{group_name}",
        )

        result["removed_group"] = (
            group_name
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

    # baseline delta
    results_df.loc[
        results_df["experiment"] == "ALL_V3",
        "delta_vs_all",
    ] = 0.0

    results_df = (
        results_df
        .sort_values(
            "delta_vs_all",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FEATURE ABLATION RANKING"
    )

    print(
        "=" * 80
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
    # Interpretation
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 80
    )

    ablations = results_df[
        results_df["experiment"]
        != "ALL_V3"
    ]

    for _, row in ablations.iterrows():

        delta = row[
            "delta_vs_all"
        ]

        name = row[
            "experiment"
        ]

        if delta > 0.00010:

            verdict = (
                "REMOVE looks promising"
            )

        elif delta > 0:

            verdict = (
                "slightly better without group"
            )

        elif delta > -0.00010:

            verdict = (
                "almost neutral"
            )

        else:

            verdict = (
                "group appears useful"
            )

        print(
            f"{name:<30} "
            f"{delta:+.6f} | "
            f"{verdict}"
        )

    # =====================================================
    # Save
    # =====================================================

    output_path = (
        ROOT
        / "logs"
        / "feature_ablation_v3.csv"
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
        "=" * 80
    )


if __name__ == "__main__":
    main()