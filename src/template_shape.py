import numpy as np

from config import HORIZON, LOOKBACK, TARGET_COLUMNS


EPS = 1e-9
RAW_DIM = LOOKBACK * len(TARGET_COLUMNS)
DEFAULT_MATCH_TEMPERATURE = 0.35


def raw_windows_from_features(X: np.ndarray) -> np.ndarray:
    """从 1242 维特征前部还原原始 144x6 历史窗口。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < RAW_DIM:
        raise ValueError(f"X shape 异常: {X.shape}, 至少需要 {RAW_DIM} 维")
    return X[:, :RAW_DIM].reshape(-1, LOOKBACK, len(TARGET_COLUMNS))


def _window_signature(raw: np.ndarray) -> np.ndarray:
    """
    为 sequence soft matching 构造低维、抗噪声历史签名。

    每变量：最近48步 mean/std/range/last/last-mean/slope/diff_std，共 42 维。
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (LOOKBACK, len(TARGET_COLUMNS)):
        raise ValueError(f"raw window shape 异常: {raw.shape}")

    recent = raw[:, -48:, :]
    n = recent.shape[1]
    x = np.arange(n, dtype=np.float64)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered ** 2)) + EPS

    mean = np.mean(recent, axis=1)
    std = np.std(recent, axis=1)
    span = np.max(recent, axis=1) - np.min(recent, axis=1)
    last = recent[:, -1, :]
    last_minus_mean = last - mean
    centered = recent - mean[:, None, :]
    slope = np.sum(centered * x_centered[None, :, None], axis=1) / denom
    diff_std = np.std(np.diff(recent, axis=1), axis=1)

    return np.concatenate(
        [mean, std, span, last, last_minus_mean, slope, diff_std],
        axis=1,
    )


def endpoint_zero_future_shape(y_abs: np.ndarray, last_values: np.ndarray):
    """
    把未来绝对轨迹分解为“整体位移 + endpoint-zero 局部形状”。

    先去掉从历史最后一点到未来终点的线性位移，再强制预测区间首尾残差为0。
    返回 shape 和每样本每变量的 shape RMS。
    """
    y_abs = np.asarray(y_abs, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)
    if y_abs.ndim != 3:
        raise ValueError(f"y_abs 必须为三维，实际 {y_abs.shape}")

    delta = y_abs - last_values[:, None, :]
    frac = (
        np.arange(1, HORIZON + 1, dtype=np.float64) / float(HORIZON)
    ).reshape(1, HORIZON, 1)
    residual = delta - frac * delta[:, -1:, :]

    edge_frac = np.linspace(0.0, 1.0, HORIZON, dtype=np.float64).reshape(
        1, HORIZON, 1
    )
    edge_line = (
        residual[:, :1, :] * (1.0 - edge_frac)
        + residual[:, -1:, :] * edge_frac
    )
    shape = residual - edge_line
    rms = np.sqrt(np.mean(shape ** 2, axis=1))
    return shape, rms


def _history_diff_scale(raw: np.ndarray) -> np.ndarray:
    recent = np.asarray(raw, dtype=np.float64)[:, -48:, :]
    return np.std(np.diff(recent, axis=1), axis=1)


def build_template_bank(bundle, train_idx: np.ndarray) -> dict:
    """
    只使用 train_idx 构建模板库，避免把验证 future 泄漏进模板。

    每个 sequence 保存：
      - 历史签名中心
      - 未来 endpoint-zero 单位形状模板
      - 历史波动 -> 未来 shape 振幅比例
      - 未来 shape 典型振幅
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    raw_all = raw_windows_from_features(bundle.X)
    signatures = _window_signature(raw_all)

    sig_mean = np.mean(signatures[train_idx], axis=0)
    sig_std = np.std(signatures[train_idx], axis=0)
    sig_std = np.where(sig_std < 1e-6, 1.0, sig_std)
    sig_z = (signatures - sig_mean) / sig_std

    shape_all, shape_rms = endpoint_zero_future_shape(
        bundle.y_abs,
        bundle.last_values,
    )
    hist_scale = _history_diff_scale(raw_all)

    sequences = sorted(np.unique(bundle.sequence_names[train_idx]).tolist())
    centroids = []
    unit_templates = []
    amplitude_ratios = []
    amplitude_medians = []
    sample_counts = []

    global_hist_floor = np.maximum(
        np.median(hist_scale[train_idx], axis=0) * 0.10,
        1e-6,
    )

    for seq in sequences:
        idx = train_idx[bundle.sequence_names[train_idx] == seq]
        if len(idx) == 0:
            raise RuntimeError(f"sequence {seq} 没有训练样本")

        centroids.append(np.mean(sig_z[idx], axis=0))

        amp = np.maximum(shape_rms[idx], 1e-8)
        unit = shape_all[idx] / amp[:, None, :]
        template = np.median(unit, axis=0)

        template_rms = np.sqrt(np.mean(template ** 2, axis=0))
        safe = np.where(template_rms < 1e-6, 1.0, template_rms)
        template = template / safe[None, :]
        unit_templates.append(template)

        denom = np.maximum(hist_scale[idx], global_hist_floor[None, :])
        ratio = shape_rms[idx] / denom
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        amplitude_ratios.append(np.median(ratio, axis=0))
        amplitude_medians.append(np.median(shape_rms[idx], axis=0))
        sample_counts.append(len(idx))

    return {
        "version": 1,
        "sequences": sequences,
        "sig_mean": sig_mean,
        "sig_std": sig_std,
        "centroids": np.asarray(centroids, dtype=np.float64),
        "unit_templates": np.asarray(unit_templates, dtype=np.float64),
        "amplitude_ratios": np.asarray(amplitude_ratios, dtype=np.float64),
        "amplitude_medians": np.asarray(amplitude_medians, dtype=np.float64),
        "hist_floor": global_hist_floor,
        "sample_counts": sample_counts,
        "match_temperature": DEFAULT_MATCH_TEMPERATURE,
        "target_columns": list(TARGET_COLUMNS),
        "horizon": HORIZON,
    }


def predict_template_shapes_from_features(
    X: np.ndarray,
    bank: dict,
    temperature: float | None = None,
):
    """仅依赖历史特征做 soft sequence matching 并生成未来形状残差。"""
    raw = raw_windows_from_features(X)
    sig = _window_signature(raw)
    sig_z = (sig - bank["sig_mean"]) / bank["sig_std"]

    centroids = np.asarray(bank["centroids"], dtype=np.float64)
    distances = np.mean(
        (sig_z[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )

    temp = float(
        bank.get("match_temperature", DEFAULT_MATCH_TEMPERATURE)
        if temperature is None
        else temperature
    )
    temp = max(temp, 1e-3)
    logits = -distances / temp
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), EPS)

    unit_templates = np.asarray(bank["unit_templates"], dtype=np.float64)
    ratios = np.asarray(bank["amplitude_ratios"], dtype=np.float64)
    amp_medians = np.asarray(bank["amplitude_medians"], dtype=np.float64)

    unit = np.einsum("ns,shv->nhv", weights, unit_templates)
    ratio = np.einsum("ns,sv->nv", weights, ratios)
    typical_amp = np.einsum("ns,sv->nv", weights, amp_medians)

    hist_scale = _history_diff_scale(raw)
    floor = np.asarray(bank["hist_floor"], dtype=np.float64)
    hist_scale = np.maximum(hist_scale, floor[None, :])
    dynamic_amp = hist_scale * ratio

    # 动态振幅和 sequence 典型振幅各占一半，降低噪声/缺失造成的振幅爆炸。
    amplitude = 0.5 * dynamic_amp + 0.5 * typical_amp
    upper = np.maximum(typical_amp * 2.5, floor[None, :])
    amplitude = np.clip(amplitude, 0.0, upper)

    shape = unit * amplitude[:, None, :]
    return np.nan_to_num(shape), weights, distances


def sequence_match_accuracy(weights: np.ndarray, true_sequence_names, bank: dict) -> float:
    seqs = np.asarray(bank["sequences"], dtype=object)
    pred = seqs[np.argmax(weights, axis=1)]
    true = np.asarray(true_sequence_names, dtype=object)
    return float(np.mean(pred == true))
