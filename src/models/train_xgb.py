from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train_folds.parquet"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_FOLDS = 5
SEED = 42


# =========================================================
# Features
# 与 LGB_V1 / CAT_V1 保持相同的 13 个原始特征
# =========================================================

NUMERICAL_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
]

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

FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS


# =========================================================
# Align categorical dtypes
# =========================================================

def prepare_categorical(train, test):

    for col in CATEGORICAL_COLS:

        # 统一成字符串，避免 train/test 类型不一致
        train_col = train[col].astype(str).str.strip()
        test_col = test[col].astype(str).str.strip()

        # train + test 使用完全相同的类别集合
        categories = sorted(
            set(train_col.unique())
            | set(test_col.unique())
        )

        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False
        )

        train[col] = train_col.astype(dtype)
        test[col] = test_col.astype(dtype)

    return train, test


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("XGBoost V1 - Baseline")
    print("=" * 60)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

    print("\nFold distribution:")
    print(train["fold"].value_counts().sort_index())

    # -----------------------------------------------------
    # Categorical preprocessing
    # -----------------------------------------------------

    train, test = prepare_categorical(train, test)

    X = train[FEATURES]
    y = train[TARGET]

    X_test = test[FEATURES]

    folds = train["fold"].to_numpy()

    # -----------------------------------------------------
    # Prediction arrays
    # -----------------------------------------------------

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    fold_scores = []
    best_iterations = []

    # -----------------------------------------------------
    # 5-fold
    # -----------------------------------------------------

    for fold in range(N_FOLDS):

        print(f"\n========== Fold {fold} ==========")

        train_idx = np.where(folds != fold)[0]
        valid_idx = np.where(folds == fold)[0]

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",

            n_estimators=2000,
            learning_rate=0.05,

            max_depth=7,
            min_child_weight=1,

            subsample=0.8,
            colsample_bytree=0.8,

            reg_alpha=0.0,
            reg_lambda=1.0,

            tree_method="hist",
            enable_categorical=True,

            random_state=SEED,
            n_jobs=-1,

            early_stopping_rounds=100,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=100,
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        valid_pred = model.predict_proba(X_valid)[:, 1]

        oof[valid_idx] = valid_pred

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred
        )

        fold_scores.append(fold_auc)

        best_iteration = model.best_iteration
        best_iterations.append(best_iteration)

        print(f"\nFold {fold} AUC: {fold_auc:.6f}")
        print(f"Best iteration: {best_iteration}")

        # -------------------------------------------------
        # Test
        # -------------------------------------------------

        test_pred += (
            model.predict_proba(X_test)[:, 1]
            / N_FOLDS
        )

    # =====================================================
    # Overall OOF
    # =====================================================

    overall_auc = roc_auc_score(y, oof)

    print("\n" + "=" * 60)
    print("Fold AUC:")

    for i, score in enumerate(fold_scores):
        print(f"Fold {i}: {score:.6f}")

    print("-" * 60)
    print(f"OOF ROC-AUC: {overall_auc:.6f}")

    print(
        "Average best iteration:",
        round(np.mean(best_iterations), 1)
    )

    # =====================================================
    # Save predictions
    # =====================================================

    oof_dir = ROOT / "predictions" / "oof"
    test_dir = ROOT / "predictions" / "test"

    oof_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.DataFrame({
        ID_COL: train[ID_COL],
        TARGET: y,
        "prediction": oof
    })

    oof_path = oof_dir / "xgb_v1.csv"
    oof_df.to_csv(oof_path, index=False)

    test_df = pd.DataFrame({
        ID_COL: test[ID_COL],
        "prediction": test_pred
    })

    test_path = test_dir / "xgb_v1.csv"
    test_df.to_csv(test_path, index=False)

    print("\nPredictions saved:")
    print(oof_path)
    print(test_path)

    print("=" * 60)


if __name__ == "__main__":
    main()