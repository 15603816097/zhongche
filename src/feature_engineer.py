import numpy as np
import pandas as pd

from config import LOOKBACK, TARGET_COLUMNS


STAT_WINDOWS = (8, 16, 32, 64, LOOKBACK)
LAGS = (1, 2, 3, 5, 10, 20, 30, 60)


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    x = x - x.mean()
    y = values - np.nanmean(values)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def extract_features_from_window(window_df: pd.DataFrame) -> np.ndarray:
    """
    固定长度 LOOKBACK 窗口 -> 特征向量。

    重点：原始序列 + 多尺度统计/斜率 + 工况切换 + 多变量相关性。
    """
    if len(window_df) < LOOKBACK:
        raise ValueError(
            f"窗口长度不足: 实际 {len(window_df)}，要求至少 {LOOKBACK}"
        )

    window = window_df.iloc[-LOOKBACK:][TARGET_COLUMNS]
    arr = window.to_numpy(dtype=np.float64)
    features = []

    # 1) 原始序列
    features.extend(arr.reshape(-1))

    # 2) 多尺度统计和趋势
    for width in STAT_WINDOWS:
        width = min(width, len(arr))
        seg = arr[-width:]

        for j in range(seg.shape[1]):
            v = seg[:, j]
            q25, q75 = np.nanpercentile(v, [25, 75])
            half = max(1, len(v) // 2)
            first_mean = float(np.nanmean(v[:half]))
            second_mean = float(np.nanmean(v[-half:]))

            features.extend([
                float(np.nanmean(v)),
                float(np.nanstd(v)),
                float(np.nanmin(v)),
                float(np.nanmax(v)),
                float(np.nanmedian(v)),
                float(q75 - q25),
                _slope(v),
                float(v[-1] - np.nanmean(v)),
                float(second_mean - first_mean),
            ])

    # 3) 多尺度 lag 差分
    last = arr[-1]
    for lag in LAGS:
        if lag < len(arr):
            features.extend(last - arr[-lag - 1])

    # 4) 工况切换/局部水平变化
    for short, long in ((8, 16), (16, 32), (32, 64)):
        recent = np.nanmean(arr[-short:], axis=0)
        previous = np.nanmean(arr[-long:-short], axis=0)
        features.extend(recent - previous)

    # 5) 最近段与全局波动率变化
    std_recent = np.nanstd(arr[-32:], axis=0)
    std_global = np.nanstd(arr, axis=0)
    features.extend(std_recent - std_global)
    features.extend(std_recent / (std_global + 1e-6))

    # 6) 跨传感器相关性：全窗口 + 最近64步
    for seg in (arr, arr[-64:]):
        for i in range(seg.shape[1]):
            for j in range(i + 1, seg.shape[1]):
                features.append(_safe_corr(seg[:, i], seg[:, j]))

    out = np.asarray(features, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def extract_inference_features(history_df: pd.DataFrame) -> np.ndarray:
    if len(history_df) < LOOKBACK:
        raise ValueError(f"历史数据不足 {LOOKBACK} 步")
    return extract_features_from_window(
        history_df.iloc[-LOOKBACK:][TARGET_COLUMNS]
    )


def robust_trend_forecast(window_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    轻量稳健趋势基线：多尺度斜率 + 信噪比收缩 + 远期阻尼。
    """
    window = window_df.iloc[-LOOKBACK:][TARGET_COLUMNS]
    arr = window.to_numpy(dtype=np.float64)
    n_targets = arr.shape[1]
    result = np.zeros((horizon, n_targets), dtype=np.float64)

    effective_h = 48.0 * (
        1.0 - np.exp(-np.arange(1, horizon + 1, dtype=np.float64) / 48.0)
    )

    for j in range(n_targets):
        values = arr[:, j]
        last = float(values[-1])

        slopes = []
        for width in (16, 32, 64):
            width = min(width, len(values))
            slopes.append(_slope(values[-width:]))
        slope = float(np.median(slopes))

        recent = values[-32:]
        noise = float(np.nanstd(np.diff(recent))) if len(recent) > 2 else 0.0
        trend_span = abs(slope) * max(len(recent) - 1, 1)
        signal = trend_span / (
            noise * np.sqrt(max(len(recent) - 1, 1)) + 1e-8
        )
        shrink = float(np.clip(signal / 2.0, 0.15, 1.0))
        slope *= shrink

        diff = np.diff(recent)
        if len(diff):
            diff_med = float(np.nanmedian(diff))
            diff_mad = float(np.nanmedian(np.abs(diff - diff_med)))
            diff_sigma = 1.4826 * diff_mad
            cap = max(4.0 * diff_sigma, 0.005 * max(abs(last), 1.0))
            slope = float(np.clip(slope, -cap, cap))

        result[:, j] = last + slope * effective_h

    return result
