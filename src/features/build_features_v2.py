#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Feature Engineering V2.

在 V1 的 22 个特征基础上增加 7 个交互特征。
保持 V1 数据不变，单独输出 V2 parquet。
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    TARGET,
    ID,
    DATA_PROCESSED,
    ENGINEERED_FEATURES,
)


# =========================================================
# V2 Features
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

V2_FEATURES = ENGINEERED_FEATURES + V2_NEW_FEATURES


# =========================================================
# Add V2 features
# =========================================================

def add_v2_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    required_cols = [
        "Annual_Income_USD",
        "Number_of_Cars_Owned",
        "Daily_Commute_km",
        "Environmental_Concern_Level",
        "Subsidy_Flag",
        "Has_Home_Charging",
        "Total_Charging_Stations",
        "Range_Anxiety_Num",
    ]

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns required for V2 features: {missing}"
        )

    income = df["Annual_Income_USD"].astype(float)
    cars = df["Number_of_Cars_Owned"].astype(float)
    commute = df["Daily_Commute_km"].astype(float)
    eco = df["Environmental_Concern_Level"].astype(float)

    # 1. 收入 / 车辆负担
    df["Income_Per_Car"] = (
        income / (cars + 1.0)
    )

    # 2. 收入 × 补贴
    df["Income_Subsidy_Interaction"] = (
        income * (1.0 + df["Subsidy_Flag"])
    )

    # 3. 家庭充电 × 通勤
    df["HomeCharging_Commute"] = (
        df["Has_Home_Charging"] * commute
    )

    # 4. 充电条件 / 里程焦虑
    df["Charging_Anxiety_Interaction"] = (
        df["Total_Charging_Stations"]
        / (df["Range_Anxiety_Num"] + 1.0)
    )

    # 5. 通勤压力 / 充电条件
    df["Commute_Charging_Pressure"] = (
        commute
        / (df["Total_Charging_Stations"] + 1.0)
    )

    # 6. 环保意识 × 收入
    df["Eco_Income_Interaction"] = (
        eco * income
    )

    # 7. 环保意识 × 家庭充电
    df["Eco_HomeCharging_Score"] = (
        eco * df["Has_Home_Charging"]
    )

    return df


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("Feature Engineering V2")
    print("=" * 60)

    # -----------------------------------------------------
    # Load V1 engineered data
    # -----------------------------------------------------

    train = pd.read_parquet(
        DATA_PROCESSED / "train_engineered.parquet"
    )

    test = pd.read_parquet(
        DATA_PROCESSED / "test_engineered.parquet"
    )

    print("\nOriginal:")
    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

    # -----------------------------------------------------
    # Add V2 features
    # -----------------------------------------------------

    train = add_v2_features(train)
    test = add_v2_features(test)

    # -----------------------------------------------------
    # Check model features
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
            f"Train missing V2 features: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Test missing V2 features: {missing_test}"
        )

    # -----------------------------------------------------
    # Check metadata columns
    #
    # V1 train has:
    #   features + TARGET + fold
    #
    # V1 test has:
    #   features + ID
    # -----------------------------------------------------

    if TARGET not in train.columns:
        raise ValueError(
            f"Train does not contain target column: {TARGET}"
        )

    if "fold" not in train.columns:
        raise ValueError(
            "Train does not contain fold column"
        )

    if ID not in test.columns:
        raise ValueError(
            f"Test does not contain ID column: {ID}"
        )

    # -----------------------------------------------------
    # Select final columns
    # -----------------------------------------------------

    # Train 不需要 ID。
    # 保留 TARGET 和 fold，供后续 CV / OOF 训练使用。
    train_output_cols = (
        V2_FEATURES
        + [TARGET, "fold"]
    )

    # Test 必须保留 ID，之后生成 submission 时使用。
    test_output_cols = (
        [ID]
        + V2_FEATURES
    )

    train_v2 = train[
        train_output_cols
    ].copy()

    test_v2 = test[
        test_output_cols
    ].copy()

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    assert TARGET in train_v2.columns
    assert "fold" in train_v2.columns
    assert ID not in train_v2.columns

    assert ID in test_v2.columns
    assert TARGET not in test_v2.columns

    assert train_v2[V2_FEATURES].isna().sum().sum() == 0
    assert test_v2[V2_FEATURES].isna().sum().sum() == 0

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    train_path = (
        DATA_PROCESSED
        / "train_engineered_v2.parquet"
    )

    test_path = (
        DATA_PROCESSED
        / "test_engineered_v2.parquet"
    )

    train_v2.to_parquet(
        train_path,
        index=False
    )

    test_v2.to_parquet(
        test_path,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("Feature Engineering V2 Complete")
    print("=" * 60)

    print("\nFeature count:")
    print(
        f"V1     = {len(ENGINEERED_FEATURES)}"
    )
    print(
        f"V2 new = {len(V2_NEW_FEATURES)}"
    )
    print(
        f"Total  = {len(V2_FEATURES)}"
    )

    print("\nV2 new features:")

    for feature in V2_NEW_FEATURES:
        print(" -", feature)

    print("\nFinal data:")
    print("Train shape:", train_v2.shape)
    print("Test shape :", test_v2.shape)

    print("\nTrain metadata:")
    print("Target:", TARGET)
    print(
        "Fold distribution:"
    )
    print(
        train_v2["fold"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved:")
    print(train_path)
    print(test_path)

    print("=" * 60)


if __name__ == "__main__":
    main()