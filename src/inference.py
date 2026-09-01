# src/inference.py (集成版)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
from config import MODEL_DIR, HORIZON, TARGET_COLUMNS
from src.feature_engineer import extract_inference_features
from src.data_cleaner import clean_sequence

# ========== 全局变量 ==========
_model_lgb = None
_model_xgb = None
_scalers_lgb = None
_scalers_xgb = None

# 默认权重（由 find_best_weight.py 给出）
DEFAULT_WEIGHT_LGB = 0.70   # 请根据搜索结果修改

# ========== 加载函数 ==========
def load_models():
    global _model_lgb, _model_xgb, _scalers_lgb, _scalers_xgb
    if _model_lgb is None:
        with open(MODEL_DIR / "model_lgb.pkl", "rb") as f:
            _model_lgb = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            _scalers_lgb = pickle.load(f)
    if _model_xgb is None:
        with open(MODEL_DIR / "model_xgb.pkl", "rb") as f:
            _model_xgb = pickle.load(f)
        with open(MODEL_DIR / "scaler_xgb.pkl", "rb") as f:
            _scalers_xgb = pickle.load(f)
    return _model_lgb, _model_xgb, _scalers_lgb, _scalers_xgb

def predict_future(history_df: pd.DataFrame, weight_lgb=DEFAULT_WEIGHT_LGB) -> np.ndarray:
    """
    预测未来 96 步绝对值
    weight_lgb: LightGBM 权重，XGBoost 权重为 1-weight_lgb
    """
    # 1. 清洗
    history_clean = clean_sequence(history_df)
    
    # 2. 特征
    features = extract_inference_features(history_clean)  # (996,)
    
    # 3. 加载模型
    model_lgb, model_xgb, scalers_lgb, scalers_xgb = load_models()
    
    # 4. LightGBM 预测差分
    X_lgb = scalers_lgb["scaler_X"].transform(features.reshape(1, -1))
    delta_scaled_lgb = model_lgb.predict(X_lgb)[0]
    delta_lgb = scalers_lgb["scaler_y"].inverse_transform(delta_scaled_lgb.reshape(1, -1))[0]
    
    # 5. XGBoost 预测差分
    X_xgb = scalers_xgb["scaler_X"].transform(features.reshape(1, -1))
    delta_scaled_xgb = model_xgb.predict(X_xgb)[0]
    delta_xgb = scalers_xgb["scaler_y"].inverse_transform(delta_scaled_xgb.reshape(1, -1))[0]
    
    # 6. 加权集成
    delta_ens = weight_lgb * delta_lgb + (1 - weight_lgb) * delta_xgb
    
    # 7. 还原绝对值
    last_hist = history_clean.iloc[-1][TARGET_COLUMNS].values
    pred_abs = delta_ens.reshape(HORIZON, len(TARGET_COLUMNS)) + np.tile(last_hist, (HORIZON, 1))
    
    return pred_abs
