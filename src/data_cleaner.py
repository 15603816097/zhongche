import numpy as np
import pandas as pd

from config import TARGET_COLUMNS


def _to_numeric_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TARGET_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out


def clean_target_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    训练标签清洗：只处理缺失/非法值，不删除真实峰值、故障突变或工况切换。
    """
    out = _to_numeric_targets(df)
    out[TARGET_COLUMNS] = out[TARGET_COLUMNS].interpolate(
        method="linear", limit_direction="both"
    )
    out[TARGET_COLUMNS] = out[TARGET_COLUMNS].ffill().bfill().fillna(0.0)
    return out


def _remove_isolated_spikes(values: np.ndarray, window: int = 9) -> np.ndarray:
    """
    只抑制“孤立尖峰”，尽量保留持续性的工况切换和故障阶跃。
    """
    x = np.asarray(values, dtype=np.float64).copy()
    n = len(x)
    if n < 5:
        return x

    s = pd.Series(x)
    rolling_median = s.rolling(
        window=window, center=True, min_periods=3
    ).median().to_numpy()

    abs_dev = np.abs(x - rolling_median)
    rolling_mad = pd.Series(abs_dev).rolling(
        window=window, center=True, min_periods=3
    ).median().to_numpy()
    local_sigma = 1.4826 * rolling_mad

    diffs = np.diff(x)
    if len(diffs):
        diff_med = np.nanmedian(diffs)
        diff_mad = np.nanmedian(np.abs(diffs - diff_med))
        diff_sigma = 1.4826 * diff_mad
    else:
        diff_sigma = 0.0

    value_scale = max(float(np.nanmedian(np.abs(x))), 1.0)
    floor = max(6.0 * diff_sigma, 1e-4 * value_scale, 1e-8)
    threshold = np.maximum(6.0 * np.nan_to_num(local_sigma, nan=0.0), floor)

    candidate = abs_dev > threshold
    replace_mask = np.zeros(n, dtype=bool)

    for i in range(1, n - 1):
        if not candidate[i]:
            continue

        left = x[i - 1]
        right = x[i + 1]
        neighbor_gap = abs(left - right)
        neighbor_limit = max(4.0 * diff_sigma, 0.01 * value_scale, 1e-8)

        if (
            neighbor_gap <= neighbor_limit
            and abs(x[i] - left) > threshold[i]
            and abs(x[i] - right) > threshold[i]
        ):
            replace_mask[i] = True

    if np.any(replace_mask):
        x[replace_mask] = rolling_median[replace_mask]

    return x


def clean_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    推理输入清洗：
    1) 缺失/非法值插值；
    2) 只去除孤立尖峰；
    3) 保留持续性的退化、故障和工况切换。
    """
    out = clean_target_sequence(df)

    for col in TARGET_COLUMNS:
        out[col] = _remove_isolated_spikes(
            out[col].to_numpy(dtype=np.float64)
        )

    return out
