# src/trainer.py (差分目标预测版)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import DATA_DIR, MODEL_DIR, TARGET_COLUMNS, LOOKBACK, HORIZON, LGB_PARAMS, TEST_SIZE, DATA_AUGMENTATION
from src.feature_engineer import extract_features_from_window
from src.data_cleaner import clean_sequence


# ==================== 数据加载（只使用原始数据） ====================
def load_all_data():
    """
    加载所有 sequence，只使用原始数据（关闭增强）
    """
    all_X, all_y = [], []
    seq_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("sequence")])
    print(f"找到 {len(seq_dirs)} 个序列文件夹: {[d.name for d in seq_dirs]}")
    
    for seq_dir in seq_dirs:
        hist_path = seq_dir / "history.csv"
        fut_path = seq_dir / "future.csv"
        if not hist_path.exists() or not fut_path.exists():
            print(f"⚠️ 跳过 {seq_dir.name}：缺少文件")
            continue
        
        history = pd.read_csv(hist_path)
        future = pd.read_csv(fut_path)
        history_clean = clean_sequence(history)
        future_clean = clean_sequence(future)
        
        X, y = build_train_data(history_clean, future_clean)
        all_X.append(X)
        all_y.append(y)
        print(f"  加载并清洗 {seq_dir.name}: history={len(history_clean)}行, future={len(future_clean)}行, 生成 {X.shape[0]} 个样本")
    
    X = np.vstack(all_X)
    y = np.vstack(all_y)
    print(f"总样本数: {X.shape[0]}")
    return X, y


def build_train_data(history, future):
    """
    滑动窗口构造样本
    X: 特征 (使用历史窗口提取)
    y: 差分目标 = 未来96步 - 历史窗口的最后一步
    """
    X, y = [], []
    max_start = len(history) - HORIZON - LOOKBACK + 1
    if max_start <= 0:
        print(f"⚠️ 警告：序列长度不足")
        return np.array(X), np.array(y)
    
    for i in range(max_start):
        # 特征：历史窗口
        window = history.iloc[i:i+LOOKBACK][TARGET_COLUMNS]
        features = extract_features_from_window(window)
        X.append(features)
        
        # 标签：差分目标
        last_hist = history.iloc[i+LOOKBACK-1][TARGET_COLUMNS].values  # shape: (6,)
        future_vals = future.iloc[:HORIZON][TARGET_COLUMNS].values.flatten()  # shape: (576,)
        delta = future_vals - np.tile(last_hist, HORIZON)   # shape: (576,)
        y.append(delta)
    
    return np.array(X), np.array(y)


# ==================== 训练主函数 ====================
def train():
    print("=" * 60)
    print("开始训练 LightGBM 多输出预测模型 (差分目标版)")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载并清洗所有数据...")
    X, y = load_all_data()
    print(f"   样本数: {X.shape[0]}, 特征维度: {X.shape[1]}, 目标维度: {y.shape[1]}")
    
    # 2. 标准化（特征和目标都标准化）
    print("\n2. 标准化...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    print("   ✅ 标准化器已保存至 models/scaler.pkl")
    
    # 3. 时序切分 (80/20)
    split_idx = int(len(X_scaled) * (1 - TEST_SIZE))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
    print(f"   训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}")
    
    # 4. 使用多进程模式训练（避免死锁，且训练更快）
    params = LGB_PARAMS.copy()
    params['n_jobs'] = 3   # 使用所有CPU核心
    print("\n3. 使用 MultiOutputRegressor（多进程模式）训练...")
    print(f"   参数: {params}")
    
    model = MultiOutputRegressor(LGBMRegressor(**params), n_jobs=3)
    model.fit(X_train, y_train)
    
    # 5. 验证集评估（在原始尺度上）
    y_pred_scaled = model.predict(X_val)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_val_orig = scaler_y.inverse_transform(y_val)
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
    print(f"\n   验证集整体 RMSE (原始尺度): {rmse:.4f}")
    
    # 6. 保存模型
    model_path = MODEL_DIR / "model_lgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ 模型已保存至: {model_path}")

if __name__ == "__main__":
    train()
