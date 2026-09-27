#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Engineering V3.

基于 V2 的特征继续增加：
1. Annual_Income_USD digit decomposition
2. Frequency Encoding
3. Income transformation / bins

注意：
- 不覆盖 V1 / V2
- 不在这里做 Target Encoding
- Target Encoding 后续在 CV fold 内完成，避免 leakage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    TARGET,
    ID,
    DATA_PROCESSED,
)

# 直接复用 V2 的特征定义
from src.features.build_features_v2 import (
    V2_FEATURES,
)


# =========================================================
# V3 Feature Definitions
# =========================================================

# ---------------------------------------------------------
# 1. Income digit decomposition
# ---------------------------------------------------------

DIGIT_FEATURES = [
    "Income_Ones",
    "Income_Tens",
    "Income_Hundreds",
    "Income_Thousands",
    "Income_TenThousands",
]


# ---------------------------------------------------------
# 2. Income transformations
# ---------------------------------------------------------

INCOME_FEATURES = [
    "Income_Log",
    "Income_1k_Bin",
    "Income_5k_Bin",
    "Income_10k_Bin",
]


# ---------------------------------------------------------
# 3. Frequency Encoding
# ---------------------------------------------------------

FREQ_SOURCE_COLUMNS = [
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

FREQ_FEATURES = [
    f"{col}_Freq"
    for col in FREQ_SOURCE_COLUMNS
]


V3_NEW_FEATURES = (
    DIGIT_FEATURES
    + INCOME_FEATURES
    + FREQ_FEATURES
)

V3_FEATURES = (
    V2_FEATURES
    + V3_NEW_FEATURES
)


# =========================================================
# Digit decomposition
# =========================================================

def add_digit_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    if "Annual_Income_USD" not in df.columns:
        raise ValueError(
            "Annual_Income_USD not found."
        )

    # 收入应该是非负数。
    # round 后转 int，避免浮点表示问题。
    income = (
        df["Annual_Income_USD"]
        .fillna(0)
        .round()
        .astype("int64")
        .abs()
    )

    df["Income_Ones"] = (
        income % 10
    )

    df["Income_Tens"] = (
        (income // 10) % 10
    )

    df["Income_Hundreds"] = (
        (income // 100) % 10
    )

    df["Income_Thousands"] = (
        (income // 1000) % 10
    )

    df["Income_TenThousands"] = (
        (income // 10000) % 10
    )

    return df


# =========================================================
# Income transformations
# =========================================================

def add_income_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    income = (
        df["Annual_Income_USD"]
        .astype(float)
        .clip(lower=0)
    )

    # Log transform
    df["Income_Log"] = np.log1p(income)

    # Fixed-width bins
    #
    # 注意这里不用 qcut。
    # 因为 train/test 如果分别 qcut，
    # bin 的边界可能不一致。
    df["Income_1k_Bin"] = (
        np.floor(income / 1000.0)
    ).astype("int32")

    df["Income_5k_Bin"] = (
        np.floor(income / 5000.0)
    ).astype("int32")

    df["Income_10k_Bin"] = (
        np.floor(income / 10000.0)
    ).astype("int32")

    return df


# =========================================================
# Frequency Encoding
# =========================================================

def add_frequency_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train = train.copy()
    test = test.copy()

    # 使用 train + test 的无标签特征分布计算 frequency。
    # 不使用 TARGET，因此不存在 target leakage。
    combined = pd.concat(
        [
            train[FREQ_SOURCE_COLUMNS],
            test[FREQ_SOURCE_COLUMNS],
        ],
        axis=0,
        ignore_index=True,
    )

    total_rows = len(combined)

    for col in FREQ_SOURCE_COLUMNS:

        counts = (
            combined[col]
            .value_counts(
                dropna=False
            )
        )

        frequency = (
            counts / total_rows
        )

        train[f"{col}_Freq"] = (
            train[col]
            .map(frequency)
            .fillna(0.0)
            .astype("float32")
        )

        test[f"{col}_Freq"] = (
            test[col]
            .map(frequency)
            .fillna(0.0)
            .astype("float32")
        )

    return train, test


# =========================================================
# Validation helper
# =========================================================

def validate_no_inf(
    df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
) -> None:

    numeric_df = (
        df[feature_cols]
        .select_dtypes(
            include=[np.number]
        )
    )

    if np.isinf(
        numeric_df.to_numpy()
    ).any():

        raise ValueError(
            f"{name} contains inf values."
        )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 65)
    print("Feature Engineering V3")
    print("=" * 65)

    # -----------------------------------------------------
    # Load V2
    # -----------------------------------------------------

    train_path = (
        DATA_PROCESSED
        / "train_engineered_v2.parquet"
    )

    test_path = (
        DATA_PROCESSED
        / "test_engineered_v2.parquet"
    )

    print("\nLoading V2 data...")

    train = pd.read_parquet(
        train_path
    )

    test = pd.read_parquet(
        test_path
    )

    print(
        "Train shape:",
        train.shape
    )

    print(
        "Test shape :",
        test.shape
    )

    # -----------------------------------------------------
    # Basic checks
    # -----------------------------------------------------

    missing_train = [
        c for c in V2_FEATURES
        if c not in train.columns
    ]

    missing_test = [
        c for c in V2_FEATURES
        if c not in test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Train missing V2 features: "
            f"{missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Test missing V2 features: "
            f"{missing_test}"
        )

    if TARGET not in train.columns:
        raise ValueError(
            f"Train missing target: {TARGET}"
        )

    if "fold" not in train.columns:
        raise ValueError(
            "Train missing fold column."
        )

    if ID not in test.columns:
        raise ValueError(
            f"Test missing ID: {ID}"
        )

    # -----------------------------------------------------
    # 1. Digit decomposition
    # -----------------------------------------------------

    print(
        "\n[1/3] Adding income digit features..."
    )

    train = add_digit_features(
        train
    )

    test = add_digit_features(
        test
    )

    # -----------------------------------------------------
    # 2. Income transforms
    # -----------------------------------------------------

    print(
        "[2/3] Adding income transforms..."
    )

    train = add_income_features(
        train
    )

    test = add_income_features(
        test
    )

    # -----------------------------------------------------
    # 3. Frequency Encoding
    # -----------------------------------------------------

    print(
        "[3/3] Adding frequency encoding..."
    )

    train, test = (
        add_frequency_features(
            train,
            test,
        )
    )

    # -----------------------------------------------------
    # Feature checks
    # -----------------------------------------------------

    missing_train_v3 = [
        c for c in V3_FEATURES
        if c not in train.columns
    ]

    missing_test_v3 = [
        c for c in V3_FEATURES
        if c not in test.columns
    ]

    if missing_train_v3:
        raise ValueError(
            f"Train missing V3 features: "
            f"{missing_train_v3}"
        )

    if missing_test_v3:
        raise ValueError(
            f"Test missing V3 features: "
            f"{missing_test_v3}"
        )

    # -----------------------------------------------------
    # Final columns
    # -----------------------------------------------------

    train_output_cols = (
        V3_FEATURES
        + [
            TARGET,
            "fold",
        ]
    )

    test_output_cols = (
        [ID]
        + V3_FEATURES
    )

    train_v3 = train[
        train_output_cols
    ].copy()

    test_v3 = test[
        test_output_cols
    ].copy()

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    assert TARGET in train_v3.columns
    assert "fold" in train_v3.columns
    assert ID not in train_v3.columns

    assert ID in test_v3.columns
    assert TARGET not in test_v3.columns

    # NaN
    train_nan = (
        train_v3[V3_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    test_nan = (
        test_v3[V3_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    if train_nan != 0:
        raise ValueError(
            f"Train contains {train_nan} NaNs."
        )

    if test_nan != 0:
        raise ValueError(
            f"Test contains {test_nan} NaNs."
        )

    # Inf
    validate_no_inf(
        train_v3,
        V3_FEATURES,
        "Train",
    )

    validate_no_inf(
        test_v3,
        V3_FEATURES,
        "Test",
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_train_path = (
        DATA_PROCESSED
        / "train_engineered_v3.parquet"
    )

    output_test_path = (
        DATA_PROCESSED
        / "test_engineered_v3.parquet"
    )

    train_v3.to_parquet(
        output_train_path,
        index=False,
    )

    test_v3.to_parquet(
        output_test_path,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print(
        "Feature Engineering V3 Complete"
    )
    print("=" * 65)

    print("\nFeature count:")

    print(
        f"V2     = {len(V2_FEATURES)}"
    )

    print(
        f"V3 new = {len(V3_NEW_FEATURES)}"
    )

    print(
        f"Total  = {len(V3_FEATURES)}"
    )

    print("\nV3 new features:")

    for feature in V3_NEW_FEATURES:
        print(
            " -",
            feature
        )

    print("\nFeature groups:")

    print(
        "Digit features     =",
        len(DIGIT_FEATURES)
    )

    print(
        "Income features    =",
        len(INCOME_FEATURES)
    )

    print(
        "Frequency features =",
        len(FREQ_FEATURES)
    )

    print("\nFinal data:")

    print(
        "Train shape:",
        train_v3.shape
    )

    print(
        "Test shape :",
        test_v3.shape
    )

    print("\nFold distribution:")

    print(
        train_v3["fold"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved:")

    print(
        output_train_path
    )

    print(
        output_test_path
    )

    print("=" * 65)


if __name__ == "__main__":
    main()