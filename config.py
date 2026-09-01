# config.py
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = Path(BASE_DIR) / "data/raw"
MODEL_DIR = Path(BASE_DIR) / "models"

TARGET_COLUMNS = ['vibration_rms', 'temperature_c', 'current_a', 'speed_rpm', 'acoustic_db', 'pressure_kpa']
LOOKBACK = 144
HORIZON = 96

# ==================== 关闭数据增强（当前无效） ====================
DATA_AUGMENTATION = False   # 只使用原始数据

# ==================== LightGBM 最优参数 (PC2 Optuna) ====================
LGB_PARAMS = {
    'n_estimators': 720,
    'learning_rate': 0.0446,
    'num_leaves': 26,
    'min_child_samples': 30,
    'subsample': 0.93,
    'colsample_bytree': 0.68,
    'reg_alpha': 0.09,
    'reg_lambda': 0.09,
    'random_state': 42,
    'verbose': -1
}

TEST_SIZE = 0.2
