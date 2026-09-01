# src/feature_engineer.py
import numpy as np
import pandas as pd
from config import TARGET_COLUMNS, LOOKBACK

def extract_features_from_window(window_df: pd.DataFrame) -> np.ndarray:
    """
    输入: 包含 LOOKBACK 行历史数据的 DataFrame
    输出: 扁平特征向量（原始序列 + 多尺度统计 + 多阶差分）
    """
    features = []
    
    # 1. 原始序列 (展平) -> 144 * 6 = 864 个特征
    raw_values = window_df[TARGET_COLUMNS].values.flatten()
    features.extend(raw_values)
    
    # 2. 多尺度子窗口统计特征 (Near, Mid, Far)
    n = len(window_df)
    # 定义三个子窗口：最近32步、中段32步、远段64步（如果数据足够）
    segments = {}
    if n >= 96:
        segments['near'] = slice(max(0, n-32), n)               # 最近 32 步
        segments['mid'] = slice(max(0, n-64), max(0, n-32))     # 前 32 步（紧挨着近段）
        segments['far'] = slice(0, max(0, n-64))                # 剩余的前段（长度可变）
    else:
        # 若窗口小于96，则只分两段
        segments['near'] = slice(max(0, n-24), n)
        segments['far'] = slice(0, max(0, n-24))
    
    for seg_name, seg_slice in segments.items():
        seg_df = window_df.iloc[seg_slice]
        if len(seg_df) < 3:
            continue
        for col in TARGET_COLUMNS:
            series = seg_df[col].values
            features.append(np.mean(series))
            features.append(np.std(series))
            features.append(np.max(series) - np.min(series))   # 极差
            # 该段线性斜率（趋势）
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0] if len(series) > 1 else 0.0
            features.append(slope)
    
    # 3. 全局统计特征（保留最重要的，去掉分位数避免冗余）
    for col in TARGET_COLUMNS:
        series = window_df[col].values
        features.append(np.mean(series))
        features.append(np.std(series))
        features.append(np.min(series))
        features.append(np.max(series))
        # 不再添加 25/75 分位数（已被子窗口覆盖）
    
    # 4. 多尺度差分特征（从最后一步向前取 lag=1,5,10,20,30,60）
    last_step = window_df[TARGET_COLUMNS].iloc[-1].values
    for lag in [1, 5, 10, 20, 30, 60]:
        if lag < len(window_df):
            prev_step = window_df[TARGET_COLUMNS].iloc[-lag-1].values
            features.extend(last_step - prev_step)
    
    return np.array(features, dtype=np.float32)

def extract_inference_features(history_df: pd.DataFrame) -> np.ndarray:
    """
    用于 API 推理：从 history.csv (512步) 中取最后 LOOKBACK 步作为窗口
    """
    if len(history_df) < LOOKBACK:
        raise ValueError(f"历史数据不足 {LOOKBACK} 步")
    last_window = history_df.iloc[-LOOKBACK:][TARGET_COLUMNS]
    return extract_features_from_window(last_window)
