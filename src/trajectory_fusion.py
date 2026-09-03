import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    对最后一个维度做 edge-padding 移动平均。

    支持 shape=(H,) / (N,H) / (...,H)。window 会自动修正为奇数。
    """
    x = np.asarray(values, dtype=np.float64)
    if x.shape[-1] < 2:
        return x.copy()

    window = int(window)
    window = max(1, min(window, x.shape[-1]))
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        return x.copy()

    pad = window // 2
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (pad, pad)
    padded = np.pad(x, pad_width, mode="edge")

    csum = np.cumsum(padded, axis=-1, dtype=np.float64)
    zero_shape = list(csum.shape)
    zero_shape[-1] = 1
    csum = np.concatenate([np.zeros(zero_shape, dtype=np.float64), csum], axis=-1)
    return (csum[..., window:] - csum[..., :-window]) / float(window)


def endpoint_zero_highpass(values: np.ndarray, window: int) -> np.ndarray:
    """
    从轨迹中提取高频/峰谷残差，并强制预测区间首尾残差为 0。

    用途：低秩 PCA 模型负责低频水平/长期趋势，原主模型只补回局部峰谷和波动，
    避免把 PCA 已经改善的终点/整体位移再次拉回原模型。

    输入最后一维必须是 horizon。
    """
    x = np.asarray(values, dtype=np.float64)
    if x.shape[-1] < 2:
        return np.zeros_like(x, dtype=np.float64)

    smooth = moving_average(x, window)
    residual = x - smooth

    horizon = x.shape[-1]
    frac = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    line = (
        residual[..., :1] * (1.0 - frac)
        + residual[..., -1:] * frac
    )
    out = residual - line
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
