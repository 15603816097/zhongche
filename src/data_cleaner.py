# src/data_cleaner.py
import pandas as pd
import numpy as np
from config import TARGET_COLUMNS

def clean_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    对单个序列的 DataFrame (history 或 future) 进行清洗
    处理 Missing, Spike
    """
    df_clean = df.copy()
    
    # 1. 处理缺失值 (Missing): 线性插值 + 前向/后向填充
    df_clean[TARGET_COLUMNS] = df_clean[TARGET_COLUMNS].interpolate(
        method='linear', limit_direction='both', limit=10
    )
    df_clean[TARGET_COLUMNS] = df_clean[TARGET_COLUMNS].ffill().bfill()
    
    # 2. 处理尖峰 (Spike): 滚动中位数 + 3倍标准差截断
    for col in TARGET_COLUMNS:
        series = df_clean[col].values  # 获取 numpy 数组，可写
        if series is None:
            continue
        rolling_median = pd.Series(series).rolling(window=11, center=True, min_periods=3).median().values
        rolling_std = pd.Series(series).rolling(window=11, center=True, min_periods=3).std().values
        
        if np.isnan(rolling_std).all():
            continue
            
        upper_bound = rolling_median + 3 * rolling_std
        lower_bound = rolling_median - 3 * rolling_std
        
        mask_high = series > upper_bound
        mask_low = series < lower_bound
        
        # 使用 numpy 的 where 进行替换，避免视图赋值问题
        series = np.where(mask_high, rolling_median, series)
        series = np.where(mask_low, rolling_median, series)
        
        # 将修改后的数组写回 DataFrame
        df_clean[col] = series
    
    return df_clean
