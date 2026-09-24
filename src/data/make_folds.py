#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固化分层 5 折划分（StratifiedKFold, shuffle, seed=42），写入 train_folds.parquet。

该 fold 列是团队唯一的线下评估标尺：所有建模/调参/融合都必须复用同一划分，
否则不同实验的 OOF 分数不可比。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import TARGET, N_FOLDS, SEED, DATA_PROCESSED


def assign_folds(
    train: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.Series:
    """返回整型 fold 列（0~n_folds-1），按目标列分层。"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = pd.Series(-1, index=train.index, dtype="int8")
    for fold, (_, va_idx) in enumerate(skf.split(train, y)):
        folds.iloc[va_idx] = fold
    return folds


def make_folds(
    train_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """读取清洗后训练集，固化 5 折，保存 train_folds.parquet，返回带 fold 的数据。"""
    train_path = train_path or DATA_PROCESSED / "train_clean.csv"
    out_path = out_path or DATA_PROCESSED / "train_folds.parquet"

    train = pd.read_csv(train_path, low_memory=False)
    train["fold"] = assign_folds(train, train[TARGET])

    global_rate = train[TARGET].mean()
    fold_rates = train.groupby("fold")[TARGET].agg(["mean", "count"])
    # 正样本总数无法被 5 整除，各折允许 ±1 正样本的固有离散，容差取 1e-4。
    assert np.allclose(fold_rates["mean"], global_rate, atol=1e-4), "各折正样本率与全局不一致"

    train.to_parquet(out_path, index=False)
    print(f"[make_folds] 全局正样本率={global_rate:.4%}")
    print(f"[make_folds] 各折正样本率={dict((fold_rates['mean'] * 100).round(4))}")
    print(f"[make_folds] 已保存 {out_path}")
    return train


if __name__ == "__main__":
    make_folds()
