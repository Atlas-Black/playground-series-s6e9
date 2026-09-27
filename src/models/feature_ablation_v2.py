from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "processed" / "train_engineered_v2.parquet"
)

TARGET = "Will_Buy_EV"
FOLD_COL = "fold"

SEED = 42

# 先固定 Fold 2 做快速筛选
VALID_FOLD = 2


# =========================================================
# V2 new features
# =========================================================

V2_NEW_FEATURES = [
    "Income_Per_Car",
    "Income_Subsidy_Interaction",
    "HomeCharging_Commute",
    "Charging_Anxiety_Interaction",
    "Commute_Charging_Pressure",
    "Eco_Income_Interaction",
    "Eco_HomeCharging_Score",
]


# =========================================================
# Load data
# =========================================================

def load_data():

    train = pd.read_parquet(TRAIN_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"Missing target: {TARGET}")

    if FOLD_COL not in train.columns:
        raise ValueError(f"Missing fold column: {FOLD_COL}")

    missing = [
        col
        for col in V2_NEW_FEATURES
        if col not in train.columns
    ]

    if missing:
        raise ValueError(
            f"Missing V2 features: {missing}"
        )

    return train


# =========================================================
# Detect categorical features
# =========================================================

def get_categorical_features(df, features):

    categorical = []

    for col in features:

        if (
            df[col].dtype == "object"
            or isinstance(
                df[col].dtype,
                pd.CategoricalDtype
            )
        ):
            categorical.append(col)

    return categorical


# =========================================================
# Train one experiment
# =========================================================

def run_experiment(
    train,
    features,
    experiment_name,
):

    train_mask = train[FOLD_COL] != VALID_FOLD
    valid_mask = train[FOLD_COL] == VALID_FOLD

    X_train = train.loc[
        train_mask,
        features
    ].copy()

    y_train = train.loc[
        train_mask,
        TARGET
    ]

    X_valid = train.loc[
        valid_mask,
        features
    ].copy()

    y_valid = train.loc[
        valid_mask,
        TARGET
    ]

    categorical_cols = get_categorical_features(
        X_train,
        features,
    )

    # CatBoost categorical columns不能有 NaN，
    # 统一转换成字符串
    for col in categorical_cols:

        X_train[col] = (
            X_train[col]
            .fillna("__MISSING__")
            .astype(str)
        )

        X_valid[col] = (
            X_valid[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=False,
    )

    pred = model.predict_proba(
        X_valid
    )[:, 1]

    auc = roc_auc_score(
        y_valid,
        pred
    )

    best_iteration = model.get_best_iteration()

    print(
        f"{experiment_name:<35} "
        f"AUC={auc:.6f} "
        f"best_iter={best_iteration}"
    )

    return auc


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 75)
    print("V2 Feature Ablation - Fast Screening")
    print(f"Validation Fold = {VALID_FOLD}")
    print("=" * 75)

    train = load_data()

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    excluded_cols = {
        TARGET,
        FOLD_COL,
        "id",
    }

    all_features = [
        col
        for col in train.columns
        if col not in excluded_cols
    ]

    print("\nTrain shape:", train.shape)
    print("Total features:", len(all_features))

    print("\nV2 features:")
    for feature in V2_NEW_FEATURES:
        print(" -", feature)

    # -----------------------------------------------------
    # Baseline
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("Baseline")
    print("=" * 75)

    baseline_auc = run_experiment(
        train,
        all_features,
        "ALL V2 FEATURES",
    )

    # -----------------------------------------------------
    # Ablation
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("Ablation")
    print("=" * 75)

    results = []

    for removed_feature in V2_NEW_FEATURES:

        features = [
            col
            for col in all_features
            if col != removed_feature
        ]

        auc = run_experiment(
            train,
            features,
            f"Remove {removed_feature}",
        )

        delta = auc - baseline_auc

        results.append({
            "removed_feature": removed_feature,
            "auc": auc,
            "delta_vs_baseline": delta,
        })

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    # 删除后提升最大的排最前面
    results_df = results_df.sort_values(
        "delta_vs_baseline",
        ascending=False,
    )

    print("\n" + "=" * 75)
    print("Ablation Results")
    print("=" * 75)

    print(
        f"\nBaseline AUC: "
        f"{baseline_auc:.6f}"
    )

    print(
        "\n"
        f"{'Removed feature':<35}"
        f"{'AUC':>12}"
        f"{'Delta':>14}"
    )

    print("-" * 61)

    for _, row in results_df.iterrows():

        print(
            f"{row['removed_feature']:<35}"
            f"{row['auc']:>12.6f}"
            f"{row['delta_vs_baseline']:>+14.6f}"
        )

    # -----------------------------------------------------
    # Interpretation
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("Interpretation")
    print("=" * 75)

    print(
        "\nDelta > 0:"
        " removing the feature improved AUC"
    )

    print(
        "Delta < 0:"
        " removing the feature reduced AUC"
    )

    print("\nPossible weak/noisy features:")

    noisy = results_df[
        results_df["delta_vs_baseline"] > 0
    ]

    if len(noisy) == 0:

        print(
            "None - every V2 feature helped "
            "on this fold."
        )

    else:

        for _, row in noisy.iterrows():

            print(
                f" - {row['removed_feature']}: "
                f"{row['delta_vs_baseline']:+.6f}"
            )

    print("\n" + "=" * 75)
    print("Done")
    print("=" * 75)


if __name__ == "__main__":
    main()