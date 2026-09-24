#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightGBM 5 折训练 + OOF/测试预测 + 提交 + 实验记录。

流程（`python -m src.models.train_lgbm`）：
    1. 读取清洗后数据，固化 5 折（src.data.make_folds）；
    2. 构建特征（src.features.build_features）；
    3. 在基线与衍生特征集上各跑一次 5 折 LightGBM，得 OOF ROC-AUC；
    4. 保存 OOF / 测试集预测到 predictions/{oof,test}/；
    5. 生成 submissions/submission_lgb_baseline.csv（基线模型折平均）；
    6. 追加实验记录到 experiments.csv，写 logs/step2_results.json。
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.config import (
    TARGET,
    ID,
    SEED,
    LGB_PARAMS,
    EARLY_STOPPING,
    BASE_FEATURES,
    ENGINEERED_FEATURES,
    DATA_PROCESSED,
    LOGS_DIR,
    SUBMISSIONS_DIR,
    PRED_OOF_DIR,
    PRED_TEST_DIR,
    EXPERIMENTS_CSV,
)
from src.data.make_folds import assign_folds
from src.features.build_features import build_features


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("train_lgbm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS_DIR / "train_lgbm.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def run_oof(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: np.ndarray,
    feature_names: Sequence[str],
    logger: logging.Logger,
) -> Tuple[float, np.ndarray, np.ndarray, List[lgb.LGBMClassifier]]:
    """在预划分 folds 上训练 LightGBM，返回 (OOF AUC, oof_prob, test_prob 折平均, models)。"""
    n_folds = int(folds.max()) + 1
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    models: List[lgb.LGBMClassifier] = []
    X = X[feature_names]
    X_test_sub = X_test[feature_names]
    for fold in range(n_folds):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_X=X_va, eval_y=y_va,
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_test_sub)[:, 1] / n_folds
        models.append(model)
    auc = roc_auc_score(y, oof)
    return auc, oof, test_pred, models


def save_oof(train: pd.DataFrame, oof: np.ndarray, feature_set: str) -> None:
    PRED_OOF_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        ID: train[ID].values,
        "fold": train["fold"].values,
        "y_true": train[TARGET].values,
        "y_pred": oof,
    })
    out.to_csv(PRED_OOF_DIR / f"{feature_set}_oof.csv", index=False)


def save_test_pred(test: pd.DataFrame, test_pred: np.ndarray, feature_set: str) -> None:
    PRED_TEST_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({ID: test[ID].values, TARGET: test_pred})
    out.to_csv(PRED_TEST_DIR / f"{feature_set}_test.csv", index=False)


def write_submission(sample_sub: pd.DataFrame, test: pd.DataFrame, test_pred: np.ndarray) -> None:
    """生成严格对齐 sample_submission 的提交文件，并做健康检查。"""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({ID: test[ID], "pred": test_pred})
    sub = sample_sub[[ID]].merge(pred_df, on=ID, how="left").rename(columns={"pred": TARGET})
    sub = sub[list(sample_sub.columns)]

    assert len(sub) == len(sample_sub), "提交行数与 sample_submission 不一致"
    assert sub[TARGET].isna().sum() == 0, "提交存在缺失值"
    assert bool(np.all((sub[TARGET] >= 0) & (sub[TARGET] <= 1))), "提交概率超出 [0,1]"
    assert bool((sub[ID].to_numpy() == sample_sub[ID].to_numpy()).all()), "提交 id 顺序不一致"

    out = SUBMISSIONS_DIR / "submission_lgb_baseline.csv"
    sub.to_csv(out, index=False)
    print(f"[submit] 已保存 {out}（{len(sub):,} 行，概率∈[{sub[TARGET].min():.4f},{sub[TARGET].max():.4f}]）")


def record_experiments(rows: List[dict]) -> None:
    """把实验记录追加写入 experiments.csv（不存在则建表头）。"""
    df_new = pd.DataFrame(rows)
    if EXPERIMENTS_CSV.exists():
        df_old = pd.read_csv(EXPERIMENTS_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(EXPERIMENTS_CSV, index=False)


def main() -> int:
    logger = setup_logger()
    logger.info("=" * 70)
    logger.info("LightGBM 训练：基线 vs 衍生特征（同一 5 折划分）")
    logger.info("=" * 70)

    # ---------- 1. 读取 ----------
    train = pd.read_csv(DATA_PROCESSED / "train_clean.csv", low_memory=False)
    test = pd.read_csv(DATA_PROCESSED / "test_clean.csv", low_memory=False)
    sample_sub = pd.read_csv(DATA_PROCESSED.parent / "raw" / "sample_submission.csv", low_memory=False)
    logger.info("train=%s test=%s", train.shape, test.shape)

    # ---------- 2. 5 折 + 特征 ----------
    train["fold"] = assign_folds(train, train[TARGET])
    train, test = build_features(train, test)
    y = train[TARGET]
    folds = train["fold"].to_numpy()
    X = train[ENGINEERED_FEATURES]
    X_test = test[ENGINEERED_FEATURES]

    # ---------- 3. 基线 ----------
    t0 = time.time()
    logger.info(">>> 基线（%d 个原始特征）", len(BASE_FEATURES))
    base_auc, base_oof, base_test, base_models = run_oof(X, y, X_test, folds, BASE_FEATURES, logger)
    logger.info("Baseline OOF ROC-AUC = %.5f (%.1fs)", base_auc, time.time() - t0)

    # ---------- 4. 衍生 ----------
    t0 = time.time()
    logger.info(">>> 衍生（%d 个特征）", len(ENGINEERED_FEATURES))
    eng_auc, eng_oof, eng_test, eng_models = run_oof(X, y, X_test, folds, ENGINEERED_FEATURES, logger)
    logger.info("Engineered OOF ROC-AUC = %.5f (%.1fs)", eng_auc, time.time() - t0)

    diff = eng_auc - base_auc
    logger.info("[对比] Baseline=%.5f  Engineered=%.5f  Diff=%+.5f", base_auc, eng_auc, diff)

    # ---------- 5. 保存预测 ----------
    save_oof(train, base_oof, "baseline")
    save_oof(train, eng_oof, "engineered")
    save_test_pred(test, base_test, "baseline")
    save_test_pred(test, eng_test, "engineered")
    logger.info("[预测] OOF/测试集预测已写入 predictions/")

    # ---------- 6. 提交（基线折平均）----------
    write_submission(sample_sub, test, base_test)

    # ---------- 7. 实验记录 ----------
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record_experiments([
        {"experiment": "baseline_lgb", "n_features": len(BASE_FEATURES),
         "oof_roc_auc": round(base_auc, 6), "test_mean_prob": round(float(base_test.mean()), 6),
         "note": "原始 13 特征", "timestamp": ts},
        {"experiment": "engineered_lgb", "n_features": len(ENGINEERED_FEATURES),
         "oof_roc_auc": round(eng_auc, 6), "test_mean_prob": round(float(eng_test.mean()), 6),
         "note": "原始+9 衍生特征（未涨分）", "timestamp": ts},
    ])
    logger.info("[记录] experiments.csv 已更新")

    # ---------- 8. 结果 JSON ----------
    results = {
        "n_folds": int(folds.max()) + 1,
        "seed": SEED,
        "baseline_auc": float(base_auc),
        "engineered_auc": float(eng_auc),
        "auc_diff": float(diff),
        "baseline_n_features": len(BASE_FEATURES),
        "engineered_n_features": len(ENGINEERED_FEATURES),
    }
    (LOGS_DIR / "step2_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[记录] logs/step2_results.json 已写入")

    logger.info("=" * 70)
    logger.info("完成。Baseline=%.5f Engineered=%.5f", base_auc, eng_auc)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
