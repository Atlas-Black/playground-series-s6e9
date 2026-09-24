#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局配置：路径、目标列、特征分组、建模超参。

所有模块共享本文件，保证划分、特征、超参的单一事实来源（可复现）。
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# 目录（repo 根 = src 的上一级）
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "logs"
SUBMISSIONS_DIR = ROOT / "submissions"
PRED_OOF_DIR = ROOT / "predictions" / "oof"
PRED_TEST_DIR = ROOT / "predictions" / "test"
EXPERIMENTS_CSV = ROOT / "experiments.csv"

# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
TARGET = "Will_Buy_EV"
ID = "id"

# --------------------------------------------------------------------------- #
# 特征分组（来自 Stage-1 的类型识别）
# --------------------------------------------------------------------------- #
NOMINAL_CAT = ["Gender", "City_Type", "Current_Car_Type"]                 # 名义
BOOLEAN_CAT = ["Home_Charging_Possible", "Subsidy_Available"]             # 布尔
ORDINAL_STR_CAT = ["Range_Anxiety_Level"]                                 # 有序（字符串）
ORDINAL_NUM_CAT = [                                                       # 有序（低基数数值）
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
NUMERICAL_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km"]         # 连续数值

# 转为 category 的全部列（名义 + 布尔 + 有序）
CATEGORICAL_COLS = NOMINAL_CAT + BOOLEAN_CAT + ORDINAL_STR_CAT + ORDINAL_NUM_CAT

# 基线特征集（13 个原始特征）
BASE_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS

# 有序字符串 -> 数值的固定映射（业务语义，非数据统计）
RANGE_ANXIETY_MAP = {"Low": 1, "Medium": 2, "High": 3}

# 第一批衍生特征（Stage-2，与原始特征冗余、未涨分）
NEW_FEATURES = [
    "Total_Charging_Stations",
    "Charging_Per_Commute",
    "Has_Home_Charging",
    "Income_Per_Age",
    "Range_Anxiety_Num",
    "Anxiety_Commute_Ratio",
    "Has_Multiple_Cars",
    "Subsidy_Flag",
    "Eco_Subsidy_Score",
]
ENGINEERED_FEATURES = BASE_FEATURES + NEW_FEATURES

# --------------------------------------------------------------------------- #
# 建模超参（与 Stage-2 基线一致）
# --------------------------------------------------------------------------- #
N_FOLDS = 5
SEED = 42
N_ESTIMATORS = 1000
EARLY_STOPPING = 50
LEARNING_RATE = 0.1
LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)
