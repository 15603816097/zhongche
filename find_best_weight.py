# find_best_weight_extended.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from config import MODEL_DIR, TEST_SIZE
from src.trainer import load_all_data  # 复用数据加载函数

# 1. 加载验证集数据
print("加载验证集数据...")
X, y = load_all_data()  # 原始特征 + 原始差分（未标准化）
split_idx = int(len(X) * (1 - TEST_SIZE))
X_val, y_val = X[split_idx:], y[split_idx:]
print(f"验证集样本数: {X_val.shape[0]}")

# 2. 加载两个模型的 scaler 和模型
print("加载模型...")
with open(MODEL_DIR / "scaler.pkl", "rb") as f:
    scalers_lgb = pickle.load(f)
with open(MODEL_DIR / "scaler_xgb.pkl", "rb") as f:
    scalers_xgb = pickle.load(f)

with open(MODEL_DIR / "model_lgb.pkl", "rb") as f:
    model_lgb = pickle.load(f)
with open(MODEL_DIR / "model_xgb.pkl", "rb") as f:
    model_xgb = pickle.load(f)

# 3. 预测验证集
print("预测验证集...")
# LightGBM 预测
X_val_scaled_lgb = scalers_lgb["scaler_X"].transform(X_val)
y_pred_scaled_lgb = model_lgb.predict(X_val_scaled_lgb)
y_pred_lgb = scalers_lgb["scaler_y"].inverse_transform(y_pred_scaled_lgb)

# XGBoost 预测
X_val_scaled_xgb = scalers_xgb["scaler_X"].transform(X_val)
y_pred_scaled_xgb = model_xgb.predict(X_val_scaled_xgb)
y_pred_xgb = scalers_xgb["scaler_y"].inverse_transform(y_pred_scaled_xgb)

# y_val 已经是原始尺度差分，无需变换

# 4. 搜索最佳权重（扩展范围 0.30 ~ 0.95，步长 0.05）
print("\n测试不同权重...")
print("-" * 60)
best_rmse = float('inf')
best_weight = 0.5

for w_lgb in np.arange(0.30, 0.96, 0.05):
    w_lgb = round(w_lgb, 2)
    w_xgb = 1 - w_lgb
    y_ens = w_lgb * y_pred_lgb + w_xgb * y_pred_xgb
    rmse = np.sqrt(mean_squared_error(y_val, y_ens))
    print(f"权重 LightGBM={w_lgb:.2f}, XGBoost={w_xgb:.2f} -> RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w_lgb

print("-" * 60)
print(f"✅ 最佳权重: LightGBM={best_weight:.2f}, XGBoost={1-best_weight:.2f}")
print(f"   最佳 RMSE (差分尺度): {best_rmse:.4f}")
print(f"   当前 LightGBM 单独 RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_lgb)):.4f}")
print(f"   当前 XGBoost 单独 RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_xgb)):.4f}")
print("\n请将以下参数更新到 src/inference.py 中：")
print(f"   默认权重 weight_lgb = {best_weight:.2f}")
