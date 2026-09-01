import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = Path(BASE_DIR) / "data/raw"
MODEL_DIR = Path(BASE_DIR) / "models"

TARGET_COLUMNS = [
    "vibration_rms",
    "temperature_c",
    "current_a",
    "speed_rpm",
    "acoustic_db",
    "pressure_kpa",
]

LOOKBACK = 144
HORIZON = 96

# 训练/验证：按每个 sequence 的时间顺序切分，并留出完整预测窗 gap，
# 避免训练标签进入验证预测区间造成时序泄漏。
VALIDATION_FRACTION = 0.15
VALIDATION_MIN_SAMPLES = 32
VALIDATION_GAP = HORIZON

LGB_PARAMS = {
    "n_estimators": 650,
    "learning_rate": 0.035,
    "num_leaves": 31,
    "min_child_samples": 24,
    "subsample": 0.90,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.15,
    "reg_lambda": 0.25,
    "random_state": 42,
    "verbose": -1,
}

XGB_PARAMS = {
    "n_estimators": 520,
    "learning_rate": 0.04,
    "max_depth": 5,
    "min_child_weight": 3.0,
    "subsample": 0.88,
    "colsample_bytree": 0.78,
    "reg_alpha": 0.12,
    "reg_lambda": 1.2,
    "objective": "reg:squarederror",
    "random_state": 42,
    "verbosity": 0,
}
