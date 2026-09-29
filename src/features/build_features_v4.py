#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Engineering V4.

基于 V3，针对 Annual_Income_USD 的强 digit signal
进一步构造 income structure features。

V3 消融结果表明：
- Income_Hundreds 非常重要
- Income_Tens 非常重要
- Income_Thousands / Ones 也有明显贡献
- 删除全部 income digits 会导致明显 AUC 下降

因此 V4 专门探索收入数字结构。

注意：
- 不覆盖 V1 / V2 / V3
- V4 完整继承 V3
- 暂时不删除任何旧特征
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    TARGET,
    ID,
    DATA_PROCESSED,
)

from src.features.build_features_v3 import (
    V3_FEATURES,
)


# =========================================================
# V4 Feature Definitions
# =========================================================

INCOME_STRUCTURE_FEATURES = [
    "Income_Last2",
    "Income_Last3",
    "Income_Last4",
    "Income_First2",
    "Income_100_Bin",
    "Income_500_Bin",
    "Income_2k_Bin",
    "Income_Hundreds_Tens",
]


V4_NEW_FEATURES = (
    INCOME_STRUCTURE_FEATURES
)

V4_FEATURES = (
    V3_FEATURES
    + V4_NEW_FEATURES
)


# =========================================================
# Income structure features
# =========================================================

def add_income_structure_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    if "Annual_Income_USD" not in df.columns:
        raise ValueError(
            "Annual_Income_USD not found."
        )

    # 与 V3 digit decomposition 保持完全一致
    income = (
        df["Annual_Income_USD"]
        .fillna(0)
        .round()
        .astype("int64")
        .abs()
    )

    # -----------------------------------------------------
    # Suffix features
    # -----------------------------------------------------

    # 最后两位
    df["Income_Last2"] = (
        income % 100
    ).astype("int32")

    # 最后三位
    df["Income_Last3"] = (
        income % 1000
    ).astype("int32")

    # 最后四位
    df["Income_Last4"] = (
        income % 10000
    ).astype("int32")

    # -----------------------------------------------------
    # Prefix feature
    # -----------------------------------------------------

    # 前两位数字。
    #
    # 例如：
    # 78436 -> 78
    # 9234  -> 92
    # 800   -> 80
    #
    # 不使用字符串，保持纯数值计算。
    safe_income = np.maximum(
        income.to_numpy(),
        1,
    )

    digits = (
        np.floor(
            np.log10(safe_income)
        ).astype(int)
        + 1
    )

    divisor_power = np.maximum(
        digits - 2,
        0,
    )

    divisor = np.power(
        10,
        divisor_power,
    )

    first2 = (
        income.to_numpy()
        // divisor
    )

    # 对于 0 收入保持为 0
    first2 = np.where(
        income.to_numpy() == 0,
        0,
        first2,
    )

    df["Income_First2"] = (
        first2.astype("int32")
    )

    # -----------------------------------------------------
    # Alternative income bins
    # -----------------------------------------------------

    df["Income_100_Bin"] = (
        income // 100
    ).astype("int32")

    df["Income_500_Bin"] = (
        income // 500
    ).astype("int32")

    df["Income_2k_Bin"] = (
        income // 2000
    ).astype("int32")

    # -----------------------------------------------------
    # Hundreds + Tens
    # -----------------------------------------------------

    # 78436:
    # hundreds = 4
    # tens     = 3
    # result   = 43
    #
    # 消融中 Hundreds 和 Tens 是最强的两个 digit。
    hundreds = (
        (income // 100) % 10
    )

    tens = (
        (income // 10) % 10
    )

    df["Income_Hundreds_Tens"] = (
        hundreds * 10
        + tens
    ).astype("int32")

    return df


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

    print("=" * 70)
    print("Feature Engineering V4")
    print("=" * 70)

    # -----------------------------------------------------
    # Load V3
    # -----------------------------------------------------

    train_path = (
        DATA_PROCESSED
        / "train_engineered_v3.parquet"
    )

    test_path = (
        DATA_PROCESSED
        / "test_engineered_v3.parquet"
    )

    print("\nLoading V3 data...")

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
        c
        for c in V3_FEATURES
        if c not in train.columns
    ]

    missing_test = [
        c
        for c in V3_FEATURES
        if c not in test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Train missing V3 features: "
            f"{missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Test missing V3 features: "
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
    # Add V4 features
    # -----------------------------------------------------

    print(
        "\nAdding income structure features..."
    )

    train = (
        add_income_structure_features(
            train
        )
    )

    test = (
        add_income_structure_features(
            test
        )
    )

    # -----------------------------------------------------
    # Feature checks
    # -----------------------------------------------------

    missing_train_v4 = [
        c
        for c in V4_FEATURES
        if c not in train.columns
    ]

    missing_test_v4 = [
        c
        for c in V4_FEATURES
        if c not in test.columns
    ]

    if missing_train_v4:
        raise ValueError(
            f"Train missing V4 features: "
            f"{missing_train_v4}"
        )

    if missing_test_v4:
        raise ValueError(
            f"Test missing V4 features: "
            f"{missing_test_v4}"
        )

    # -----------------------------------------------------
    # Final columns
    # -----------------------------------------------------

    train_output_cols = (
        V4_FEATURES
        + [
            TARGET,
            "fold",
        ]
    )

    test_output_cols = (
        [ID]
        + V4_FEATURES
    )

    train_v4 = train[
        train_output_cols
    ].copy()

    test_v4 = test[
        test_output_cols
    ].copy()

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    assert TARGET in train_v4.columns
    assert "fold" in train_v4.columns
    assert ID not in train_v4.columns

    assert ID in test_v4.columns
    assert TARGET not in test_v4.columns

    train_nan = (
        train_v4[V4_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    test_nan = (
        test_v4[V4_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    if train_nan != 0:
        raise ValueError(
            f"Train contains "
            f"{train_nan} NaNs."
        )

    if test_nan != 0:
        raise ValueError(
            f"Test contains "
            f"{test_nan} NaNs."
        )

    validate_no_inf(
        train_v4,
        V4_FEATURES,
        "Train",
    )

    validate_no_inf(
        test_v4,
        V4_FEATURES,
        "Test",
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_train_path = (
        DATA_PROCESSED
        / "train_engineered_v4.parquet"
    )

    output_test_path = (
        DATA_PROCESSED
        / "test_engineered_v4.parquet"
    )

    train_v4.to_parquet(
        output_train_path,
        index=False,
    )

    test_v4.to_parquet(
        output_test_path,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("Feature Engineering V4 Complete")
    print("=" * 70)

    print("\nFeature count:")

    print(
        f"V3     = {len(V3_FEATURES)}"
    )

    print(
        f"V4 new = {len(V4_NEW_FEATURES)}"
    )

    print(
        f"Total  = {len(V4_FEATURES)}"
    )

    print(
        "\nV4 new features:"
    )

    for feature in V4_NEW_FEATURES:
        print(
            " -",
            feature
        )

    print(
        "\nFinal data:"
    )

    print(
        "Train shape:",
        train_v4.shape
    )

    print(
        "Test shape :",
        test_v4.shape
    )

    print(
        "\nFold distribution:"
    )

    print(
        train_v4["fold"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSaved:"
    )

    print(
        output_train_path
    )

    print(
        output_test_path
    )

    print("=" * 70)


if __name__ == "__main__":
    main()