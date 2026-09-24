#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特征工程：类别列对齐 + 电动车业务衍生特征。

零泄漏约束：
    - 衍生特征仅使用特征列（不触碰目标列），对 train/test 施加完全相同的变换；
    - 类别标签集仅为特征词汇对齐（非目标统计）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import (
    TARGET,
    ID,
    DATA_PROCESSED,
    NOMINAL_CAT,
    BOOLEAN_CAT,
    ORDINAL_STR_CAT,
    ORDINAL_NUM_CAT,
    RANGE_ANXIETY_MAP,
    BASE_FEATURES,
    NEW_FEATURES,
    ENGINEERED_FEATURES,
)


def align_categorical(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """把类别列转为 pandas category dtype，并保证 train/test 类别集与顺序一致。"""
    str_cats = NOMINAL_CAT + BOOLEAN_CAT + ORDINAL_STR_CAT
    num_cats = ORDINAL_NUM_CAT

    for c in str_cats:
        t = train[c].astype(str).str.strip()
        s = test[c].astype(str).str.strip()
        cats = sorted(set(t.unique()) | set(s.unique()))
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        train[c] = t.astype(dtype)
        test[c] = s.astype(dtype)

    for c in num_cats:
        t = train[c].astype(float).round().astype(int)
        s = test[c].astype(float).round().astype(int)
        cats = sorted(set(t.unique()) | set(s.unique()))
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        train[c] = t.astype(dtype)
        test[c] = s.astype(dtype)

    return train, test


def add_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """为 train/test 施加相同的电动车业务衍生特征。"""
    def derive(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 1) 充电便利度
        df["Total_Charging_Stations"] = (
            df["Charging_Stations_Near_Home"].astype(float)
            + df["Charging_Stations_Near_Work"].astype(float)
        )
        df["Charging_Per_Commute"] = (
            df["Total_Charging_Stations"] / (df["Daily_Commute_km"].astype(float) + 1.0)
        )
        df["Has_Home_Charging"] = (df["Home_Charging_Possible"].astype(str) == "Yes").astype(int)
        # 2) 经济与购买力
        df["Income_Per_Age"] = df["Annual_Income_USD"].astype(float) / (df["Age"].astype(float) + 1.0)
        # 3) 焦虑与出行矛盾
        df["Range_Anxiety_Num"] = df["Range_Anxiety_Level"].astype(str).map(RANGE_ANXIETY_MAP).astype(float)
        df["Anxiety_Commute_Ratio"] = df["Range_Anxiety_Num"] * df["Daily_Commute_km"].astype(float)
        # 4) 多车与环保偏好
        df["Has_Multiple_Cars"] = (df["Number_of_Cars_Owned"].astype(float) > 1).astype(int)
        df["Subsidy_Flag"] = (df["Subsidy_Available"].astype(str) == "Yes").astype(int)
        df["Eco_Subsidy_Score"] = df["Environmental_Concern_Level"].astype(float) * df["Subsidy_Flag"].astype(float)
        return df

    return derive(train), derive(test)


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对齐类别 + 衍生特征，返回带全部（新老）特征的 train/test。"""
    train, test = align_categorical(train, test)
    train, test = add_features(train, test)
    return train, test


if __name__ == "__main__":
    train = pd.read_csv(DATA_PROCESSED / "train_clean.csv", low_memory=False)
    test = pd.read_csv(DATA_PROCESSED / "test_clean.csv", low_memory=False)
    train, test = build_features(train, test)
    train[[ID, TARGET] + ENGINEERED_FEATURES].to_parquet(DATA_PROCESSED / "train_engineered.parquet", index=False)
    test[[ID] + ENGINEERED_FEATURES].to_parquet(DATA_PROCESSED / "test_engineered.parquet", index=False)
    print(f"[build_features] train_engineered={train.shape} test_engineered={test.shape}")
    print(f"[build_features] 特征数={len(ENGINEERED_FEATURES)}（基线 {len(BASE_FEATURES)} + 衍生 {len(NEW_FEATURES)}）")
