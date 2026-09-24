#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_pipeline.py
=================
Stage-1 数据清洗与标准化管道 —— Kaggle Playground 系列
"Predicting Electric Vehicle Purchases" (二分类)。

职责：
    1) 数据审计（规格、目标分布、缺失、高基数、常数列、重复行）；
    2) 标准化清洗（特殊缺失标记 -> np.nan、对象列数字化、Schema 对齐）；
    3) 特征类型自动识别（id / numerical / nominal / ordinal / boolean）；
    4) 严格的零泄漏 + 自动化断言校验；
    5) 输出干净数据集 + 数据审计报告 + 特征 Schema 元数据。

设计原则（与建模端 GBDT 直接衔接）：
    * 零数据泄漏 (Zero Leakage)：
        - 绝不将 train 与 test 拼接后计算任何全局统计量（均值/方差/分位数/目标编码等）。
        - 任何需要"状态"的转换（数字列识别、Schema 顺序、目标编码元信息）一律在
          train 上 fit，再 transform 到 test。本阶段不引入需要全局统计的转换。
    * 最少干预 (Minimal Intervention)：
        - 保留原生缺失值，不做均值/中位数/众数插补。
        - 仅将明显特殊字符（'', 'NA', 'None', '?', 'null', 'missing' 等）统一为 np.nan。
    * 可复现 (Reproducible)：确定性流程、详细日志、可审计产物。

依赖：仅 pandas + numpy（无 sklearn 依赖；parquet 输出在装有 pyarrow 时自动启用）。

用法：
    python clean_pipeline.py \
        --train train.csv --test test.csv --sample_sub sample_submission.csv \
        --target Will_Buy_EV --id-col id --outdir .
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

# 将被统一转换为 np.nan 的"明显特殊缺失标记"（先 strip + lower 后匹配）。
SPECIAL_MISSING_TOKENS: frozenset = frozenset({
    "", "na", "n/a", "nan", "none", "null", "nil", "nul",
    "?", "??", "missing", "undefined", "unknown", "unk",
    "-", "--", "---", "#n/a", "#na", "#null", "n a", "n/a ",
    "empty", "blank", "inf", "-inf", "infinity", "+inf", "-infinity",
})

# 布尔型取值的候选集合（用于二值列识别）。
BOOLEAN_VALUE_SETS: Tuple[frozenset, ...] = (
    frozenset({"0", "1"}),
    frozenset({"0.0", "1.0"}),
    frozenset({"true", "false"}),
    frozenset({"yes", "no"}),
    frozenset({"y", "n"}),
    frozenset({"t", "f"}),
)

# ID 列名的启发式集合（小写匹配）。
ID_NAME_HINTS: frozenset = frozenset({
    "id", "index", "row_id", "rowid", "customer_id", "customerid",
    "user_id", "userid", "order_id", "transaction_id",
})

# 名义/有序的低基数阈值：数值列基数 (2, ORDINAL_MAX_CARD] 判定为"有序候选"。
ORDINAL_MAX_CARD: int = 20
# 名义类别列基数超过该值则标记为"高基数"。
HIGH_CARD_THRESHOLD: int = 1000

# 目标列二分类取值 -> 0/1 的语义映射 token（正类=1，负类=0）。
POSITIVE_TARGET_TOKENS: frozenset = frozenset({
    "yes", "true", "1", "1.0", "y", "t", "positive", "p", "buy", "will_buy",
})
NEGATIVE_TARGET_TOKENS: frozenset = frozenset({
    "no", "false", "0", "0.0", "n", "f", "negative", "neg", "not_buy", "won't_buy",
})

# 常见有序类别字符串（已小写）的显式顺序（用于 Ordinal 自动识别）。
ORDINAL_STRING_ORDERINGS: Dict[str, Tuple[str, ...]] = {
    "low_medium_high": ("low", "medium", "high"),
    "very_low_low_medium_high_very_high": (
        "very low", "low", "medium", "high", "very high",
    ),
    "small_medium_large": ("small", "medium", "large"),
    "very_small_small_medium_large_very_large": (
        "very small", "small", "medium", "large", "very large",
    ),
    "poor_fair_good_very_good_excellent": (
        "poor", "fair", "good", "very good", "excellent",
    ),
    "strongly_disagree_disagree_neutral_agree_strongly_agree": (
        "strongly disagree", "disagree", "neutral", "agree", "strongly agree",
    ),
    "beginner_intermediate_advanced_expert": (
        "beginner", "intermediate", "advanced", "expert",
    ),
}


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

def human_bytes(n: float) -> str:
    """把字节数格式化为人类可读字符串。"""
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


def utc_now() -> str:
    """返回 UTC 时间字符串（用于产物时间戳）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_string_like(series: pd.Series) -> bool:
    """判断一列是否为字符串类（object 或 Arrow str 或 category）。"""
    return (
        pd_types.is_object_dtype(series)
        or pd_types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def resolve_file(candidates: Sequence[str], role: str) -> str:
    """从候选路径中选择第一个存在的文件，并返回绝对路径。"""
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return str(p.resolve())
    raise FileNotFoundError(
        f"[{role}] 未找到输入文件。已尝试候选路径：{list(candidates)}。"
        f"请确认文件已放置于工作目录，或用 --{role} 显式指定路径。"
    )


def read_csv_clean(path: str, role: str) -> pd.DataFrame:
    """读取 CSV 并做轻量预处理：去除列名首尾空白。"""
    logger = logging.getLogger("clean_pipeline")
    logger.info("[读取] %s <- %s", role, path)
    df = pd.read_csv(path, low_memory=False)
    # 规范化列名：去除首尾空白（不改变大小写/内部空格，避免误伤）。
    df.columns = [str(c).strip() for c in df.columns]
    mem = human_bytes(df.memory_usage(deep=True).sum())
    logger.info("[读取] %s 形状=%s 列数=%d 内存≈%s", role, df.shape, df.shape[1], mem)
    return df


# --------------------------------------------------------------------------- #
# 清洗步骤
# --------------------------------------------------------------------------- #

def standardize_special_missing(
    df: pd.DataFrame,
    exclude_cols: Sequence[str],
    tokens: frozenset = SPECIAL_MISSING_TOKENS,
) -> Tuple[pd.DataFrame, int]:
    """
    将字符串类列中的"明显特殊缺失标记"统一转换为 np.nan。

    仅处理字符串类列（object / str / category），对数值列不做任何改动；
    不会触碰 exclude_cols（ID / 目标列）。
    为兼容 Arrow str dtype，先把列转为 object 再赋 np.nan，保证产物里是
    真正标准的 numpy NaN，而非 pd.NA。

    返回 (新 DataFrame, 被替换的单元格总数)。
    """
    logger = logging.getLogger("clean_pipeline")
    df = df.copy()
    replaced_total = 0
    for col in df.columns:
        if col in exclude_cols:
            continue
        if not is_string_like(df[col]):
            continue
        s = df[col].astype(object)
        # strip + lower 后与 token 集合匹配（大小写不敏感）。
        cleaned = s.astype(str).str.strip().str.lower()
        mask = cleaned.isin(tokens) & s.notna()
        n_replaced = int(mask.sum())
        if n_replaced:
            s.loc[mask] = np.nan
            df[col] = s
            replaced_total += n_replaced
            logger.info(
                "[清洗] 列 `%s`：%d 个特殊缺失标记（NA/None/?/空串等）-> np.nan",
                col, n_replaced,
            )
    if replaced_total == 0:
        logger.info("[清洗] 未发现需要标准化的特殊缺失标记。")
    else:
        logger.info("[清洗] 共将 %d 个特殊缺失标记统一为 np.nan。", replaced_total)
    return df, replaced_total


def detect_numeric_coercible_columns(
    df: pd.DataFrame,
    exclude_cols: Sequence[str],
) -> List[str]:
    """
    在（train）上"fit"：识别"所有非空值都能解析为数字"的对象/字符串列。

    判定规则（确定性、纯逐列，不依赖跨列或跨集统计）：
        - 剥离千分位逗号与首尾空白后，to_numeric(errors='coerce')；
        - 若未引入任何"新的 NaN"（即原有非空值全部可解析），
          且并非全部为空，则认定为可数字化的列。
    """
    logger = logging.getLogger("clean_pipeline")
    coercible: List[str] = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if not is_string_like(df[col]):
            continue
        s = df[col].astype(object)
        orig_missing = s.isna()
        cleaned = s.astype(str).str.replace(",", "", regex=False).str.strip()
        numeric = pd.to_numeric(cleaned, errors="coerce")
        new_missing = numeric.isna() & ~orig_missing
        if new_missing.sum() == 0 and not numeric.isna().all():
            coercible.append(col)
            logger.info(
                "[识别] 列 `%s`（%s）判定为可数字化列，将在 train/test 上统一转换。",
                col, df[col].dtype,
            )
    if not coercible:
        logger.info("[识别] 无对象列需要数字化转换。")
    return coercible


def apply_numeric_coercion(
    df: pd.DataFrame,
    coercible_cols: Sequence[str],
) -> pd.DataFrame:
    """对指定列执行 to_numeric（剥离千分位逗号 + 空白）。"""
    df = df.copy()
    for col in coercible_cols:
        s = df[col].astype(object)
        cleaned = s.astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(cleaned, errors="coerce")
    return df


def reconcile_dtypes(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    保证 train（特征部分）与 test 的每个特征 dtype 一致。
    若不一致，尝试安全统一（优先数值化，否则转 object）。
    """
    for col in features:
        dt_train, dt_test = train[col].dtype, test[col].dtype
        if dt_train == dt_test:
            continue
        logger.warning(
            "[对齐] 特征 `%s` dtype 不一致：train=%s vs test=%s，尝试统一。",
            col, dt_train, dt_test,
        )
        # 任一方为数值型，则尝试把另一方数值化。
        if pd_types.is_numeric_dtype(train[col]) or pd_types.is_numeric_dtype(test[col]):
            t = pd.to_numeric(train[col], errors="coerce")
            s = pd.to_numeric(test[col], errors="coerce")
            if not (t.isna() & train[col].notna()).any() and not (s.isna() & test[col].notna()).any():
                train[col], test[col] = t, s
                logger.warning("[对齐] 特征 `%s` 已统一为数值型。", col)
                continue
        # 兜底：两者都转 object。
        train[col] = train[col].astype(object)
        test[col] = test[col].astype(object)
        logger.warning("[对齐] 特征 `%s` 已统一为 object。", col)
    return train, test


# --------------------------------------------------------------------------- #
# 特征类型识别
# --------------------------------------------------------------------------- #

def classify_column(
    series: pd.Series,
    name: str,
    id_col: Optional[str],
    ordinal_max_card: int = ORDINAL_MAX_CARD,
    high_card_threshold: int = HIGH_CARD_THRESHOLD,
) -> Tuple[str, str]:
    """
    把单列归类为：id / numerical / nominal / ordinal / boolean / constant。

    返回 (type, note)。
    """
    name_l = str(name).lower().strip()
    nunique = int(series.nunique(dropna=True))
    n = len(series)

    # 1) ID
    if id_col is not None and name == id_col:
        return "id", "声明的 ID 列"
    if name_l in ID_NAME_HINTS or (name_l.endswith("_id") and len(name_l) > 3):
        return "id", "ID 类命名启发式"

    # 2) constant（单一值）
    if nunique <= 1:
        return "constant", f"单一值列（nunique={nunique}）"

    # 3) boolean（原生 bool 或二值且取值落在已知对中）
    if pd_types.is_bool_dtype(series):
        return "boolean", "原生 bool 类型"
    if nunique == 2:
        uniq = frozenset(str(v).strip().lower() for v in series.dropna().unique())
        for pair in BOOLEAN_VALUE_SETS:
            if uniq <= pair:
                return "boolean", f"二值列，取值 {sorted(uniq)}"

    # 4) 数值
    if pd_types.is_numeric_dtype(series):
        if 2 < nunique <= ordinal_max_card:
            return "ordinal", (
                f"低基数数值列（nunique={nunique}），视为有序候选，请核对顺序语义"
            )
        return "numerical", f"数值列（{series.dtype}，nunique={nunique}）"

    # 5) 类别
    if is_string_like(series):
        # 5a) 有序字符串识别（显式顺序白名单）
        uniq_lower = frozenset(
            str(v).strip().lower() for v in series.dropna().unique()
        )
        for ordering in ORDINAL_STRING_ORDERINGS.values():
            if uniq_lower == frozenset(ordering):
                return "ordinal", f"有序类别：{' < '.join(ordering)}"
        # 5b) 高基数名义
        if nunique >= high_card_threshold:
            return "nominal", f"高基数类别列（nunique={nunique}），建议谨慎编码"
        return "nominal", f"名义类别列（nunique={nunique}）"

    # fallback
    return "nominal", f"其它类型（{series.dtype}，nunique={nunique}）"


def encode_target(
    series: pd.Series,
    target_name: str,
    logger: logging.Logger,
) -> Tuple[pd.Series, str]:
    """
    把二分类目标列标准化为 int8 的 {0,1}。

    - 数值型 {0,1}（或 {0.0,1.0}）直接转 int8；
    - 其它二值数值（如 {1,2}）按 min->0, max->1 映射；
    - 字符串二分类（Yes/No、True/False、1/0 等）按语义映射：
      正类 token -> 1，负类 token -> 0。

    返回 (int8 目标序列, 映射说明字符串)。
    该操作仅作用于 train 的目标列，不涉及 test，无泄漏风险。
    """
    if int(series.isna().sum()) != 0:
        raise ValueError(f"目标列 `{target_name}` 存在缺失，无法进行二分类建模。")

    if pd_types.is_numeric_dtype(series):
        uniq = sorted(series.dropna().unique())
        if len(uniq) != 2:
            raise ValueError(
                f"目标列 `{target_name}` 唯一值数={len(uniq)}（{uniq}），非二分类。"
            )
        lo, hi = float(uniq[0]), float(uniq[1])
        if (lo, hi) == (0.0, 1.0):
            return series.astype("int8"), "已为数值 0/1，直接转为 int8"
        mapping = {uniq[0]: 0, uniq[1]: 1}
        desc = f"二值数值映射：{uniq[0]} -> 0，{uniq[1]} -> 1"
        logger.info("[目标] %s", desc)
        return series.map(mapping).astype("int8"), desc

    # 字符串目标
    s = series.astype(str).str.strip().str.lower()
    uniq = sorted(s.dropna().unique())
    if len(uniq) != 2:
        raise ValueError(
            f"目标列 `{target_name}` 唯一值数={len(uniq)}（{uniq}），非二分类。"
        )
    pos = [v for v in uniq if v in POSITIVE_TARGET_TOKENS]
    neg = [v for v in uniq if v in NEGATIVE_TARGET_TOKENS]
    if len(pos) != 1 or len(neg) != 1:
        raise ValueError(
            f"目标列 `{target_name}` 取值 {uniq} 无法可靠映射为 0/1，"
            f"请显式指定正负类或修正数据。"
        )
    mapping = {neg[0]: 0, pos[0]: 1}
    desc = f"字符串二分类映射：`{neg[0]}` -> 0，`{pos[0]}` -> 1"
    logger.info("[目标] %s", desc)
    return s.map(mapping).astype("int8"), desc


def count_cross_duplicates(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> int:
    """
    只读诊断：统计 test 中与 train 在特征层面（排除 ID/目标）完全重复的行数。

    仅用于"泄漏审计"，不修改任何数据、不参与任何预处理统计。
    采用 pandas inner merge（哈希连接）实现，避免拼接后计算全局统计。
    """
    cols = list(train_features.columns)
    train_unique = train_features.drop_duplicates()
    merged = test_features.merge(train_unique, on=cols, how="inner")
    return int(len(merged))


# --------------------------------------------------------------------------- #
# Schema / 审计产物
# --------------------------------------------------------------------------- #

def build_feature_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    id_col: Optional[str],
    target: str,
) -> Dict:
    """构建 feature_schema.json 所需的数据结构。"""
    rows = []
    for col in features:
        stype, note = classify_column(train[col], col, id_col)
        if stype == "id" and col == id_col:
            # ID 列的类型信息以显式声明为准，但仍记录其 dtype。
            stype, note = "id", "ID 列"
        rows.append({
            "name": col,
            "dtype": str(train[col].dtype),
            "type": stype,
            "nunique": int(train[col].nunique(dropna=True)),
            "missing_train": int(train[col].isna().sum()),
            "missing_train_rate": round(float(train[col].isna().mean()), 6),
            "missing_test": int(test[col].isna().sum()),
            "missing_test_rate": round(float(test[col].isna().mean()), 6),
            "notes": note,
        })
    schema = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "task": "binary-classification",
        "target": target,
        "id_column": id_col,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "num_features": len(features),
        "feature_type_counts": {
            t: sum(1 for r in rows if r["type"] == t)
            for t in ("id", "numerical", "nominal", "ordinal", "boolean", "constant")
        },
        "features": rows,
    }
    return schema


def build_audit_report(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    schema: Dict,
    target: str,
    id_col: Optional[str],
    transform_log: List[str],
    special_missing_replaced: int,
    numeric_coerced: List[str],
    target_encoding_desc: str,
    cross_dup_count: int,
) -> str:
    """生成数据审计报告（Markdown）。"""
    L: List[str] = []
    add = L.append

    # ---------- 标题与生成时间 ----------
    add("# 数据审计报告（Stage-1 清洗与标准化）")
    add("")
    add(f"- 生成时间（UTC）：`{utc_now()}`")
    add(f"- 任务类型：二分类（目标列 `{target}`）")
    add(f"- 报告范围：原始数据集审计 + 清洗后对齐校验")
    add("")

    # ---------- 1. 基本规格 ----------
    add("## 1. 数据集基本规格")
    add("")
    add("| 数据集 | 行数 | 列数 | 内存占用（deep） |")
    add("|---|---|---|---|")
    add(
        f"| train（原始） | {len(train_raw):,} | {train_raw.shape[1]} | "
        f"{human_bytes(train_raw.memory_usage(deep=True).sum())} |"
    )
    add(
        f"| test（原始） | {len(test_raw):,} | {test_raw.shape[1]} | "
        f"{human_bytes(test_raw.memory_usage(deep=True).sum())} |"
    )
    add(
        f"| train（清洗后） | {len(train_clean):,} | {train_clean.shape[1]} | "
        f"{human_bytes(train_clean.memory_usage(deep=True).sum())} |"
    )
    add(
        f"| test（清洗后） | {len(test_clean):,} | {test_clean.shape[1]} | "
        f"{human_bytes(test_clean.memory_usage(deep=True).sum())} |"
    )
    add("")

    # ---------- 2. 目标列分布 ----------
    add("## 2. 目标列分布")
    add("")
    tgt = train_clean[target]
    vc = tgt.value_counts(dropna=False).sort_index()
    add(f"- 目标列 `{target}`：缺失数 = **{int(tgt.isna().sum())}**，"
        f"唯一值数 = **{int(tgt.nunique())}**")
    add(f"- 目标编码：{target_encoding_desc}（正类=1，负类=0）。")
    add("- 分布：")
    add("")
    add("| 取值 | 样本数 | 占比 |")
    add("|---|---|---|")
    for k, v in vc.items():
        add(f"| {k} | {int(v):,} | {v / len(tgt):.4%} |")
    add("")
    if tgt.nunique() == 2:
        # vc 已按取值排序（0,1）；0=负类，1=正类。
        neg = int(vc.get(0, 0))
        pos = int(vc.get(1, 0))
        total = neg + pos
        pos_rate = pos / total if total else 0.0
        minority_rate = min(neg, pos) / total if total else 0.0
        add(f"- 正样本占比 ≈ **{pos_rate:.2%}**（正 {pos:,} / 负 {neg:,}）。")
        add(f"- 负:正 ≈ **{neg / pos:.2f} : 1**。")
        if minority_rate < 0.20:
            add(f"- ⚠️ **类别不平衡（少数类占比 {minority_rate:.2%}）**，建模时建议启用 "
                f"`scale_pos_weight` / `class_weight`，并以 ROC-AUC 为主、PR-AUC 为辅评估。")
        elif minority_rate < 0.35:
            add("- 类别轻度不平衡，可视情况启用 `scale_pos_weight`。")
        else:
            add("- 类别基本平衡，无需特别处理。")
    add("")

    # ---------- 3. 特征类型与 Schema ----------
    add("## 3. 特征类型识别结果")
    add("")
    add("| 类型 | 数量 |")
    add("|---|---|")
    for t in ("id", "numerical", "nominal", "ordinal", "boolean", "constant"):
        add(f"| {t} | {schema['feature_type_counts'].get(t, 0)} |")
    add("")
    add("> 完整逐特征元数据见 `feature_schema.json`。")
    add("")

    # ---------- 4. 缺失值统计 ----------
    add("## 4. 缺失值统计（Train vs Test）")
    add("")
    features = [r["name"] for r in schema["features"]]
    miss_rows = []
    test_only_missing = []
    for r in schema["features"]:
        mt = r["missing_train"]
        st = r["missing_test"]
        rt = r["missing_train_rate"]
        rst = r["missing_test_rate"]
        flag = ""
        if mt == 0 and st > 0:
            flag = "⚠️ **仅 Test 缺失**"
            test_only_missing.append(r["name"])
        if mt > 0 and st == 0:
            flag = "⚠️ 仅 Train 缺失"
        if mt > 0 or st > 0:
            miss_rows.append(
                f"| `{r['name']}` | {r['type']} | {mt:,} ({rt:.2%}) | "
                f"{st:,} ({rst:.2%}) | {flag} |"
            )
    if miss_rows:
        add("| 特征 | 类型 | Train 缺失 | Test 缺失 | 备注 |")
        add("|---|---|---|---|---|")
        add("\n".join(miss_rows))
    else:
        add("✅ 所有特征在 Train 与 Test 中均无缺失值。")
    add("")
    if test_only_missing:
        add(f"- ⚠️ **仅在 Test 中缺失的特征**：{', '.join(f'`{c}`' for c in test_only_missing)}。"
            "此类特征在训练集完好，但测试集存在缺失，需在建模端关注缺失分布漂移。")
        add("")
    # 缺失率最高的若干特征摘要
    if miss_rows:
        top = sorted(schema["features"], key=lambda r: -max(r["missing_train_rate"], r["missing_test_rate"]))[:10]
        add("缺失率最高的特征（Top 10，按 Train/Test 缺失率较大者）：")
        add("")
        for r in top:
            if max(r["missing_train_rate"], r["missing_test_rate"]) > 0:
                add(f"- `{r['name']}`：Train {r['missing_train_rate']:.2%} / "
                    f"Test {r['missing_test_rate']:.2%}")
        add("")

    # ---------- 5. 高基数 & 常数列 & 重复行 ----------
    add("## 5. 高基数、常数列与重复行检测")
    add("")
    high_card = [r for r in schema["features"] if r["type"] == "nominal" and r["nunique"] >= HIGH_CARD_THRESHOLD]
    constants = [r["name"] for r in schema["features"] if r["type"] == "constant"]
    near_unique = [
        r for r in schema["features"]
        if r["type"] in ("numerical", "nominal")
        and r["nunique"] >= int(0.9 * len(train_clean))
        and r["name"] != id_col
    ]
    add("### 5.1 高基数类别特征")
    add("")
    if high_card:
        add("| 特征 | 唯一值数 |")
        add("|---|---|")
        for r in high_card:
            add(f"| `{r['name']}` | {r['nunique']:,} |")
        add("")
        add("> 建议：优先使用 CatBoost 原生类别处理或 LightGBM categorical feature，"
            "避免对高基数类别做 one-hot；如需编码请基于 **train 单独** 计算频率/目标编码。")
    else:
        add("未发现高基数类别特征。")
    add("")
    add("### 5.2 常数列（单一值特征）")
    add("")
    if constants:
        add(f"- 常数列：{', '.join(f'`{c}`' for c in constants)}。这些列无区分度，"
            "建模时可考虑剔除（本阶段保留以维持 Schema 完整性）。")
    else:
        add("未发现常数列。")
    add("")
    add("### 5.3 疑似近唯一 / 潜在 ID 泄漏列")
    add("")
    if near_unique:
        nu_names = "`, `".join(r["name"] for r in near_unique)
        add(f"- `{nu_names}`：基数接近行数，请核对是否为 ID / 序列号，谨防泄漏。")
    else:
        add("未发现。")
    add("")
    add("### 5.4 完全重复行")
    add("")
    dup_train = int(train_clean.duplicated().sum())
    dup_test = int(test_clean.duplicated().sum())
    add(f"- train 内部完全重复行：**{dup_train:,}** / {len(train_clean):,}")
    add(f"- test 内部完全重复行：**{dup_test:,}** / {len(test_clean):,}")
    add("")
    add("### 5.5 跨集重复（泄漏审计，只读诊断）")
    add("")
    add(f"- test 中与 train 在特征层面（不含 ID/目标）完全重复的行数：**{cross_dup_count:,}** "
        f"/ {len(test_clean):,}（{cross_dup_count / len(test_clean):.2%}）。")
    if cross_dup_count > 0:
        add("- ⚠️ 存在跨集重复行，可能造成标签泄漏；建模/评估时建议按 ID 去重或单独处理。")
    else:
        add("- ✅ 未发现跨集完全重复行。")
    add("")

    # ---------- 6. 清洗流水线记录 ----------
    add("## 6. 清洗流水线记录（Transform Log）")
    add("")
    add("| 步骤 | 说明 |")
    add("|---|---|")
    for i, msg in enumerate(transform_log, 1):
        add(f"| {i} | {msg} |")
    add("")

    # ---------- 7. 零泄漏声明 ----------
    add("## 7. 零数据泄漏声明")
    add("")
    add("- ✅ 未将 `train` 与 `test` 拼接后计算任何全局统计量（均值/方差/分位数等）。")
    add("- ✅ 未使用目标编码、均值编码等需要联合统计的编码方式。")
    add("- ✅ 缺失值保留原生状态，未做插补；仅标准化特殊缺失标记为 `np.nan`。")
    add("- ✅ 所有“带状态”的转换（可数字化列识别、Schema 列序）均在 `train` 上确定后施加于 `test`。")
    add("")
    add("## 8. 断言校验结果")
    add("")
    add("以下断言在保存产物前已全部通过，否则脚本会中断并报错：")
    add("")
    for a in (
        "`len(test_clean) == len(sample_sub)`（测试集行数一致）",
        "`(test_clean['id'] == sample_sub['id']).all()`（ID 顺序完全对齐）",
        "`set(train_clean.columns) - {target} == set(test_clean.columns)`（特征集合对齐）",
        "`train_clean[target].isna().sum() == 0`（目标列无缺失）",
        "`train_clean[target].nunique() == 2`（目标严格二分类）",
    ):
        add(f"- ✅ {a}")
    add("")

    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 校验（断言）
# --------------------------------------------------------------------------- #

def run_assertions(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    sample_sub: pd.DataFrame,
    target: str,
    id_col: str,
    logger: logging.Logger,
) -> None:
    """执行全部硬断言，任一不通过即抛 AssertionError 中断。"""
    logger.info("[断言] 开始执行自动化校验...")

    a1 = len(test_clean) == len(sample_sub)
    logger.info("[断言] len(test_clean)==len(sample_sub) -> %s (%d vs %d)",
                "PASS" if a1 else "FAIL", len(test_clean), len(sample_sub))
    assert a1, f"测试集行数不一致：test={len(test_clean)} vs sample_sub={len(sample_sub)}"

    a2 = (test_clean[id_col].reset_index(drop=True).astype(str).to_numpy()
          == sample_sub[id_col].reset_index(drop=True).astype(str).to_numpy()).all()
    logger.info("[断言] test_clean[%r] 与 sample_sub[%r] ID 顺序对齐 -> %s",
                id_col, id_col, "PASS" if a2 else "FAIL")
    assert a2, "测试集 ID 与 sample_submission 的 ID 顺序未完全对齐"

    a3 = set(train_clean.columns) - {target} == set(test_clean.columns)
    logger.info("[断言] 特征集合对齐 -> %s", "PASS" if a3 else "FAIL")
    assert a3, (
        f"特征集合不一致。train-only: "
        f"{set(train_clean.columns) - set(test_clean.columns) - {target}}, "
        f"test-only: {set(test_clean.columns) - set(train_clean.columns)}"
    )

    a4 = int(train_clean[target].isna().sum()) == 0
    logger.info("[断言] 目标列无缺失 -> %s", "PASS" if a4 else "FAIL")
    assert a4, f"目标列 {target} 存在 {train_clean[target].isna().sum()} 个缺失值"

    a5 = int(train_clean[target].nunique()) == 2
    logger.info("[断言] 目标列为严格二分类 -> %s", "PASS" if a5 else "FAIL")
    assert a5, f"目标列 {target} 唯一值数={train_clean[target].nunique()}，非二分类"

    # 额外校验：目标取值必须为 {0, 1}
    uniq = frozenset(float(v) for v in train_clean[target].dropna().unique())
    a6 = uniq <= {0.0, 1.0}
    logger.info("[断言] 目标取值 ∈ {{0,1}} -> %s (取值 %s)", "PASS" if a6 else "FAIL", sorted(uniq))
    assert a6, f"目标列取值异常：{sorted(uniq)}（应为 0/1）"

    # 额外校验：特征列顺序一致（更强于集合相等）
    a7 = list(train_clean.columns.drop(target)) == list(test_clean.columns)
    logger.info("[断言] 特征列顺序一致 -> %s", "PASS" if a7 else "FAIL")
    assert a7, "train 与 test 的特征列顺序不一致"

    # 额外校验：ID 无缺失
    a8 = int(test_clean[id_col].isna().sum()) == 0
    logger.info("[断言] 测试集 ID 无缺失 -> %s", "PASS" if a8 else "FAIL")
    assert a8, f"测试集 ID 列 {id_col} 存在缺失"

    logger.info("[断言] 全部 %d 项校验通过。", 8)


# --------------------------------------------------------------------------- #
# 保存产物
# --------------------------------------------------------------------------- #

def save_dataframe(df: pd.DataFrame, path: Path, logger: logging.Logger) -> None:
    """保存 DataFrame：优先 parquet（若 pyarrow 可用），否则 CSV。"""
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            logger.info("[保存] %s (parquet, %d 行)", path, len(df))
            return
        except ImportError:
            logger.warning("[保存] 未安装 pyarrow，回退为 CSV 输出。")
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    logger.info("[保存] %s (csv, %d 行)", path, len(df))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EV Purchases 数据清洗管道")
    parser.add_argument("--train", default="data/raw/train.csv", help="训练集路径")
    parser.add_argument("--test", default="data/raw/test.csv", help="测试集路径")
    parser.add_argument("--sample_sub", default=None,
                        help="提交样例路径（默认自动探测 data/raw/sample_submission.csv 等）")
    parser.add_argument("--target", default="Will_Buy_EV", help="目标列名")
    parser.add_argument("--id-col", default="id", help="ID 列名")
    parser.add_argument("--outdir", default="data/processed", help="输出目录")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv",
                        help="输出格式（parquet 需 pyarrow）")
    parser.add_argument("--ordinal-max-card", type=int, default=ORDINAL_MAX_CARD,
                        help="有序特征低基数判定阈值")
    parser.add_argument("--high-card-threshold", type=int, default=HIGH_CARD_THRESHOLD,
                        help="高基数类别判定阈值")
    args = parser.parse_args(argv)

    # 日志：同时输出到控制台与文件。
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clean_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    # 防御：窄编码控制台（如 GBK）遇到无法编码字符时替换为 '?'，避免 UnicodeEncodeError。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_path = outdir / "clean_pipeline.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("EV Purchases 数据清洗管道启动")
    logger.info("=" * 70)
    transform_log: List[str] = []

    # ---------- 0. 定位输入 ----------
    train_path = resolve_file([args.train], "train")
    test_path = resolve_file([args.test], "test")
    if args.sample_sub:
        sub_path = resolve_file([args.sample_sub], "sample_sub")
    else:
        sub_path = resolve_file(
            ["sample_submission.csv", "sample_submission (1).csv", "sample_submission(1).csv",
             "data/raw/sample_submission.csv"],
            "sample_sub",
        )

    # ---------- 1. 读取 ----------
    train = read_csv_clean(train_path, "train")
    test = read_csv_clean(test_path, "test")
    sample_sub = read_csv_clean(sub_path, "sample_sub")
    logger.info("[原始] train=%s test=%s sample_sub=%s", train.shape, test.shape, sample_sub.shape)
    # 保留原始快照用于审计报告（规格/内存统计），后续清洗会原地/拷贝改写 train/test。
    train_raw = train.copy()
    test_raw = test.copy()

    # ---------- 2. 目标列与 ID 列定位 ----------
    if args.target not in train.columns:
        raise ValueError(f"目标列 `{args.target}` 不存在于 train 列中：{list(train.columns)}")
    id_col = args.id_col
    if id_col not in train.columns or id_col not in test.columns:
        # 自动探测唯一整型 ID 列
        for cand in ("id", "ID", "Id"):
            if cand in train.columns and cand in test.columns:
                id_col = cand
                break
        else:
            raise ValueError(f"未找到 ID 列（尝试 {args.id_col!r}），请用 --id-col 指定。")
    logger.info("[定位] 目标列=`%s` ID列=`%s`", args.target, id_col)

    # 目标列编码：二分类 -> int8 {0,1}（无缺失、严格二分类）
    train[args.target], target_encoding_desc = encode_target(train[args.target], args.target, logger)
    transform_log.append(f"目标编码：{target_encoding_desc}（`{args.target}` -> int8 0/1）。")
    logger.info("[目标] 分布（0/1）：%s", dict(train[args.target].value_counts().sort_index()))

    # ---------- 3. 特征集合 ----------
    features = [c for c in train.columns if c != args.target]
    # 校验 test 是否与 train 特征对齐（原始）
    if set(features) != set(test.columns):
        raise ValueError(
            f"train/test 特征列不一致。train-only: {set(features) - set(test.columns)}, "
            f"test-only: {set(test.columns) - set(features)}"
        )

    # ---------- 4. 缺失标准化（特殊字符 -> np.nan）----------
    exclude = {args.target, id_col}
    train, n_rep_train = standardize_special_missing(train, exclude)
    test, n_rep_test = standardize_special_missing(test, exclude)
    n_rep_total = n_rep_train + n_rep_test
    if n_rep_total:
        transform_log.append(
            f"缺失标准化：将 train/test 共 {n_rep_total} 个特殊缺失标记"
            f"（空串/NA/None/?/null 等）统一为 np.nan（train={n_rep_train}, test={n_rep_test}）。"
        )

    # ---------- 5. 可数字化列识别（在 train 上 fit）与转换 ----------
    coercible = detect_numeric_coercible_columns(train, exclude)
    if coercible:
        train = apply_numeric_coercion(train, coercible)
        test = apply_numeric_coercion(test, coercible)
        transform_log.append(
            f"数值化转换：将 {len(coercible)} 个对象列转为数值型"
            f"（{', '.join(f'`{c}`' for c in coercible)}），识别基于 train，施加于 test。"
        )
        logger.info("[转换] 已数值化列：%s", coercible)

    # ---------- 6. Schema 对齐（列序 + dtype 统一）----------
    # 重排：train = [features..., target]，test = [features...]
    train = train[features + [args.target]]
    test = test[features]
    train, test = reconcile_dtypes(train, test, features, logger)
    transform_log.append("Schema 对齐：train 与 test 特征列名、顺序、dtype 已统一。")

    # ---------- 7. 断言校验 ----------
    run_assertions(train, test, sample_sub, args.target, id_col, logger)

    # ---------- 8. 生成 Schema 与审计报告 ----------
    schema = build_feature_schema(train, test, features, id_col, args.target)
    cross_dup_count = count_cross_duplicates(train[features], test[features])
    logger.info("[审计] 跨集（test 与 train 特征层面）完全重复行数=%d", cross_dup_count)
    report_md = build_audit_report(
        train_raw=train_raw,
        test_raw=test_raw,
        train_clean=train,
        test_clean=test,
        schema=schema,
        target=args.target,
        id_col=id_col,
        transform_log=transform_log,
        special_missing_replaced=n_rep_total,
        numeric_coerced=coercible,
        target_encoding_desc=target_encoding_desc,
        cross_dup_count=cross_dup_count,
    )

    # ---------- 9. 保存产物 ----------
    ext = ".parquet" if args.format == "parquet" else ".csv"
    save_dataframe(train, outdir / f"train_clean{ext}", logger)
    save_dataframe(test, outdir / f"test_clean{ext}", logger)

    schema_path = outdir / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[保存] %s", schema_path)

    report_path = outdir / "data_audit_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("[保存] %s", report_path)

    # ---------- 10. 汇总 ----------
    logger.info("=" * 70)
    logger.info("清洗与校验全流程完成。")
    logger.info("    train_clean 行数=%d 列数=%d", len(train), train.shape[1])
    logger.info("    test_clean  行数=%d 列数=%d", len(test), test.shape[1])
    logger.info("    特征数=%d 类型分布=%s", len(features), schema["feature_type_counts"])
    logger.info("    产物：train_clean%s / test_clean%s / feature_schema.json / data_audit_report.md",
                ext, ext)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
