import numpy as np
import pandas as pd

from config import HORIZON, LOOKBACK, TARGET_COLUMNS
from src.feature_engineer import robust_trend_forecast_array


EPS = 1e-12
PATTERN_LAGS = (8, 12, 16, 24, 32, 48)


def _finite_array(values: np.ndarray) -> np.ndarray:
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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) < 3 or len(a) != len(b):
        return 0.0

    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= EPS or sb <= EPS:
        return 0.0

    value = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def _robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    if sigma <= EPS:
        sigma = float(np.std(x))
    return max(sigma, 0.0)


def adaptive_pattern_forecast_array(
    values: np.ndarray,
    horizon: int = HORIZON,
) -> np.ndarray:
    """
    小样本场景的确定性趋势形状基线。

    思路：
      1. 先使用 robust_trend_forecast_array 给出稳定的长期斜率；
      2. 在最近一阶差分里寻找 8/12/16/24/32/48 步的重复形状；
      3. 仅把“重复且可信”的差分残差叠加到稳定斜率上；
      4. 随预测距离指数衰减，并用鲁棒差分尺度限幅。

    这样不会像直接复制历史曲线那样容易发散，同时能给 PeakF1、DiffCorr、
    VolatilityFit 提供比纯线性 baseline 更丰富的峰谷/波动信息。
    """
    arr = _finite_array(values)
    if len(arr) < LOOKBACK:
        raise ValueError(f"窗口长度不足: 实际 {len(arr)}，要求至少 {LOOKBACK}")
    arr = arr[-LOOKBACK:]

    linear = robust_trend_forecast_array(arr, horizon)
    last = arr[-1]
    diffs = np.diff(arr, axis=0)

    linear_extended = np.vstack([last[None, :], linear])
    linear_diff = np.diff(linear_extended, axis=0)
    result = linear.copy()

    decay = np.exp(-np.arange(horizon, dtype=np.float64) / 72.0)

    for j in range(len(TARGET_COLUMNS)):
        series_diff = diffs[:, j]
        best_score = 0.0
        best_lag = None

        for lag in PATTERN_LAGS:
            if 2 * lag > len(series_diff):
                continue

            prev = series_diff[-2 * lag:-lag]
            recent = series_diff[-lag:]
            corr = _safe_corr(prev, recent)
            if corr <= 0.0:
                continue

            prev_std = float(np.std(prev))
            recent_std = float(np.std(recent))
            if prev_std <= EPS or recent_std <= EPS:
                amp_consistency = 0.0
            else:
                ratio = recent_std / prev_std
                amp_consistency = float(np.exp(-abs(np.log(max(ratio, EPS)))))

            score = corr * amp_consistency
            if score > best_score:
                best_score = score
                best_lag = lag

        # 相关性太弱时直接使用原稳健趋势，不凭空制造波动。
        if best_lag is None or best_score < 0.12:
            continue

        recent = series_diff[-best_lag:].copy()
        center = float(np.median(recent))
        residual = recent - center

        sigma = _robust_scale(series_diff[-64:])
        if sigma <= EPS:
            continue

        # 只恢复“形状残差”，长期均值/斜率仍由 robust trend 负责。
        residual = np.clip(residual, -3.5 * sigma, 3.5 * sigma)
        repeated = np.resize(residual, horizon)

        # score 0.12 -> 几乎不注入；score 越接近 1，保留越多周期形状。
        strength = float(np.clip((best_score - 0.12) / 0.55, 0.0, 1.0))
        strength *= 0.85

        future_diff = linear_diff[:, j] + strength * decay * repeated

        # 防止极少数异常历史差分导致未来振幅失控。
        local_med = float(np.median(series_diff[-64:]))
        cap = max(4.0 * sigma, 0.005 * max(abs(float(last[j])), 1.0))
        future_diff = np.clip(future_diff, local_med - cap, local_med + cap)

        result[:, j] = float(last[j]) + np.cumsum(future_diff)

    result = np.asarray(result, dtype=np.float64)
    bad = ~np.isfinite(result)
    if np.any(bad):
        fallback = np.tile(last, (horizon, 1))
        result[bad] = fallback[bad]
    return result


def adaptive_pattern_forecast(
    history_df: pd.DataFrame,
    horizon: int = HORIZON,
) -> np.ndarray:
    arr = history_df.iloc[-LOOKBACK:][TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return adaptive_pattern_forecast_array(arr, horizon)
