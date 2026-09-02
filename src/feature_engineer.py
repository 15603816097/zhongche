import numpy as np
import pandas as pd

from config import LOOKBACK, TARGET_COLUMNS


# 修改特征定义时递增，用于训练数据缓存失效判断。
FEATURE_VERSION = 2
STAT_WINDOWS = (8, 16, 32, 64, LOOKBACK)
LAGS = (1, 2, 3, 5, 10, 20, 30, 60)


def _finite_array(values: np.ndarray) -> np.ndarray:
    """把输入转成连续 float64，并补齐极少数残留 NaN/Inf。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(
            f"输入 shape 错误: {arr.shape}，要求 (*, {len(TARGET_COLUMNS)})"
        )

    if np.isfinite(arr).all():
        return np.ascontiguousarray(arr)

    arr = arr.copy()
    safe = np.where(np.isfinite(arr), arr, np.nan)
    med = np.nanmedian(safe, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        rows, cols = np.where(bad)
        arr[rows, cols] = med[cols]
    return np.ascontiguousarray(arr)


def _slopes_matrix(seg: np.ndarray) -> np.ndarray:
    """一次计算所有传感器的线性斜率。"""
    n = len(seg)
    if n < 2:
        return np.zeros(seg.shape[1], dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.zeros(seg.shape[1], dtype=np.float64)
    # x 已中心化，因此无需显式减去每列均值。
    return np.dot(x, seg) / denom


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return float(_slopes_matrix(values)[0])


def _corr_features(seg: np.ndarray) -> np.ndarray:
    """返回 6 个变量两两相关系数，共 15 维。"""
    std = np.std(seg, axis=0)
    valid = std > 1e-12
    corr = np.eye(seg.shape[1], dtype=np.float64)

    if np.count_nonzero(valid) >= 2:
        sub = np.corrcoef(seg[:, valid], rowvar=False)
        valid_idx = np.where(valid)[0]
        corr[np.ix_(valid_idx, valid_idx)] = sub

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    i, j = np.triu_indices(seg.shape[1], k=1)
    return corr[i, j]


def extract_features_from_array(values: np.ndarray) -> np.ndarray:
    """
    NumPy 高速版固定长度窗口特征。

    特征定义与旧版保持一致，总维度仍为 1242：
      864 原始值
      270 多尺度统计/趋势
       48 lag 差分
       18 工况切换
       12 波动率变化
       30 跨传感器相关性
    """
    arr = _finite_array(values)
    if len(arr) < LOOKBACK:
        raise ValueError(f"窗口长度不足: 实际 {len(arr)}，要求至少 {LOOKBACK}")
    arr = arr[-LOOKBACK:]

    parts = [arr.reshape(-1)]

    # 1) 多尺度统计和趋势。一次对 6 列向量化计算，再按“变量优先”展平，
    # 与旧版逐列 append 的顺序完全一致。
    for width in STAT_WINDOWS:
        seg = arr[-min(width, len(arr)):]
        half = max(1, len(seg) // 2)

        mean = np.mean(seg, axis=0)
        std = np.std(seg, axis=0)
        min_v = np.min(seg, axis=0)
        max_v = np.max(seg, axis=0)
        median = np.median(seg, axis=0)
        q25, q75 = np.percentile(seg, [25, 75], axis=0)
        slope = _slopes_matrix(seg)
        first_mean = np.mean(seg[:half], axis=0)
        second_mean = np.mean(seg[-half:], axis=0)

        block = np.column_stack(
            [
                mean,
                std,
                min_v,
                max_v,
                median,
                q75 - q25,
                slope,
                seg[-1] - mean,
                second_mean - first_mean,
            ]
        )
        parts.append(block.reshape(-1))

    # 2) 多尺度 lag 差分
    last = arr[-1]
    for lag in LAGS:
        if lag < len(arr):
            parts.append(last - arr[-lag - 1])

    # 3) 工况切换/局部水平变化
    for short, long in ((8, 16), (16, 32), (32, 64)):
        recent = np.mean(arr[-short:], axis=0)
        previous = np.mean(arr[-long:-short], axis=0)
        parts.append(recent - previous)

    # 4) 最近段与全局波动率变化
    std_recent = np.std(arr[-32:], axis=0)
    std_global = np.std(arr, axis=0)
    parts.append(std_recent - std_global)
    parts.append(std_recent / (std_global + 1e-6))

    # 5) 跨传感器相关性：全窗口 + 最近64步
    parts.append(_corr_features(arr))
    parts.append(_corr_features(arr[-64:]))

    out = np.concatenate(parts).astype(np.float32, copy=False)
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    if out.size != 1242:
        raise RuntimeError(f"特征维度异常: {out.size}，预期 1242")
    return out


def extract_features_from_window(window_df: pd.DataFrame) -> np.ndarray:
    if len(window_df) < LOOKBACK:
        raise ValueError(f"窗口长度不足: 实际 {len(window_df)}，要求至少 {LOOKBACK}")
    arr = window_df.iloc[-LOOKBACK:][TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return extract_features_from_array(arr)


def extract_inference_features(history_df: pd.DataFrame) -> np.ndarray:
    if len(history_df) < LOOKBACK:
        raise ValueError(f"历史数据不足 {LOOKBACK} 步")
    arr = history_df.iloc[-LOOKBACK:][TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return extract_features_from_array(arr)


def robust_trend_forecast_array(values: np.ndarray, horizon: int) -> np.ndarray:
    """NumPy 高速版稳健趋势基线。"""
    arr = _finite_array(values)
    if len(arr) < LOOKBACK:
        raise ValueError(f"窗口长度不足: 实际 {len(arr)}，要求至少 {LOOKBACK}")
    arr = arr[-LOOKBACK:]

    n_targets = arr.shape[1]
    result = np.empty((horizon, n_targets), dtype=np.float64)
    effective_h = 48.0 * (
        1.0 - np.exp(-np.arange(1, horizon + 1, dtype=np.float64) / 48.0)
    )

    # 多尺度斜率一次性计算 6 个变量。
    slope_stack = np.vstack([
        _slopes_matrix(arr[-16:]),
        _slopes_matrix(arr[-32:]),
        _slopes_matrix(arr[-64:]),
    ])
    slopes = np.median(slope_stack, axis=0)

    recent = arr[-32:]
    diff = np.diff(recent, axis=0)
    noise = np.std(diff, axis=0)
    trend_span = np.abs(slopes) * max(len(recent) - 1, 1)
    signal = trend_span / (
        noise * np.sqrt(max(len(recent) - 1, 1)) + 1e-8
    )
    shrink = np.clip(signal / 2.0, 0.15, 1.0)
    slopes = slopes * shrink

    diff_med = np.median(diff, axis=0)
    diff_mad = np.median(np.abs(diff - diff_med), axis=0)
    diff_sigma = 1.4826 * diff_mad
    last = arr[-1]
    cap = np.maximum(4.0 * diff_sigma, 0.005 * np.maximum(np.abs(last), 1.0))
    slopes = np.clip(slopes, -cap, cap)

    result[:] = last[None, :] + effective_h[:, None] * slopes[None, :]
    return result


def robust_trend_forecast(window_df: pd.DataFrame, horizon: int) -> np.ndarray:
    arr = window_df.iloc[-LOOKBACK:][TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return robust_trend_forecast_array(arr, horizon)
