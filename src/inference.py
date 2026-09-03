import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.feature_engineer import (
    extract_inference_features,
    robust_trend_forecast,
)
from src.trend_pattern import adaptive_pattern_forecast


_model_lgb = None
_model_xgb = None
_scalers_lgb = None
_scalers_xgb = None
_ensemble_config = None
_model_trend_xgb = None
_scalers_trend_xgb = None

DEFAULT_LGB_WEIGHTS = np.array(
    [0.65] * len(TARGET_COLUMNS),
    dtype=np.float64,
)
DEFAULT_BASELINE_WEIGHTS = np.array(
    [0.15] * len(TARGET_COLUMNS),
    dtype=np.float64,
)

# LightGBM 训练模型为 576 个独立输出。在线预测直接调用所需 estimator，
# 避免 MultiOutputRegressor.predict() 的 joblib 进程调度开销。
LGB_INFER_THREADS = max(
    1,
    int(
        os.getenv(
            "LGB_INFER_THREADS",
            str(max(1, min(8, os.cpu_count() or 1))),
        )
    ),
)
LGB_INFER_EXECUTOR = ThreadPoolExecutor(
    max_workers=LGB_INFER_THREADS,
    thread_name_prefix="lgb-infer",
)


def load_models():
    global _model_lgb, _model_xgb
    global _scalers_lgb, _scalers_xgb, _ensemble_config

    if _model_lgb is None:
        with open(MODEL_DIR / "model_lgb.pkl", "rb") as f:
            _model_lgb = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            _scalers_lgb = pickle.load(f)

    if _model_xgb is None:
        with open(MODEL_DIR / "model_xgb.pkl", "rb") as f:
            _model_xgb = pickle.load(f)
        with open(MODEL_DIR / "scaler_xgb.pkl", "rb") as f:
            _scalers_xgb = pickle.load(f)

    if _ensemble_config is None:
        config_path = MODEL_DIR / "ensemble_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                _ensemble_config = pickle.load(f)
        else:
            _ensemble_config = {
                "version": 1,
                "lgb_weights": DEFAULT_LGB_WEIGHTS.tolist(),
                "baseline_weights": DEFAULT_BASELINE_WEIGHTS.tolist(),
                "target_columns": list(TARGET_COLUMNS),
            }

    return (
        _model_lgb,
        _model_xgb,
        _scalers_lgb,
        _scalers_xgb,
        _ensemble_config,
    )


def load_trend_model():
    """V5 被接受后才会调用；旧配置完全不会产生额外模型加载/推理开销。"""
    global _model_trend_xgb, _scalers_trend_xgb

    if _model_trend_xgb is None:
        model_path = MODEL_DIR / "model_trend_xgb.pkl"
        scaler_path = MODEL_DIR / "scaler_trend_xgb.pkl"
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                "V5 配置已启用，但缺少 model_trend_xgb.pkl 或 scaler_trend_xgb.pkl"
            )
        with open(model_path, "rb") as f:
            _model_trend_xgb = pickle.load(f)
        with open(scaler_path, "rb") as f:
            _scalers_trend_xgb = pickle.load(f)

    return _model_trend_xgb, _scalers_trend_xgb


def _stepwise_parameters_from_config(config):
    """
    统一把 V1/V2/V3 的主融合配置展开成 (HORIZON, 6) 逐步参数。

    V1/V2: 每个变量一套静态 LGB/BASE 权重
    V3+:   1-32 / 33-64 / 65-96 三段独立 LGB/BASE + delta gain
    """
    n_targets = len(TARGET_COLUMNS)

    old_lgb = np.asarray(
        config.get("lgb_weights", DEFAULT_LGB_WEIGHTS),
        dtype=np.float64,
    )
    old_base = np.asarray(
        config.get("baseline_weights", DEFAULT_BASELINE_WEIGHTS),
        dtype=np.float64,
    )
    if old_lgb.shape != (n_targets,):
        old_lgb = DEFAULT_LGB_WEIGHTS.copy()
    if old_base.shape != (n_targets,):
        old_base = DEFAULT_BASELINE_WEIGHTS.copy()

    lgb_step = np.repeat(old_lgb[None, :], HORIZON, axis=0)
    base_step = np.repeat(old_base[None, :], HORIZON, axis=0)
    gain_step = np.ones((HORIZON, n_targets), dtype=np.float64)

    segments = config.get("horizon_segments")
    lgb_seg = np.asarray(
        config.get("lgb_weights_by_segment", []),
        dtype=np.float64,
    )
    base_seg = np.asarray(
        config.get("baseline_weights_by_segment", []),
        dtype=np.float64,
    )
    gain_seg = np.asarray(
        config.get("delta_gain_by_segment", []),
        dtype=np.float64,
    )

    if isinstance(segments, (list, tuple)) and len(segments) > 0:
        expected_shape = (len(segments), n_targets)
        if (
            lgb_seg.shape == expected_shape
            and base_seg.shape == expected_shape
            and gain_seg.shape == expected_shape
        ):
            for k, pair in enumerate(segments):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                start, end = int(pair[0]), int(pair[1])
                start = max(0, min(HORIZON, start))
                end = max(start, min(HORIZON, end))
                if end <= start:
                    continue
                lgb_step[start:end] = lgb_seg[k]
                base_step[start:end] = base_seg[k]
                gain_step[start:end] = gain_seg[k]

    return (
        np.clip(lgb_step, 0.0, 1.0),
        np.clip(base_step, 0.0, 0.8),
        np.clip(gain_step, 0.75, 1.35),
    )


def _pattern_weights_from_config(config):
    """展开 V4 pattern 注入权重；旧配置默认全 0，完全兼容。"""
    weights = np.zeros(
        (HORIZON, len(TARGET_COLUMNS)),
        dtype=np.float64,
    )

    segments = config.get("horizon_segments_v4")
    by_segment = np.asarray(
        config.get("pattern_weights_by_segment", []),
        dtype=np.float64,
    )

    if not isinstance(segments, (list, tuple)) or len(segments) == 0:
        return weights

    expected = (len(segments), len(TARGET_COLUMNS))
    if by_segment.shape != expected:
        return weights

    for k, pair in enumerate(segments):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        start = max(0, min(HORIZON, start))
        end = max(start, min(HORIZON, end))
        if end <= start:
            continue
        weights[start:end] = by_segment[k]

    return np.clip(weights, 0.0, 0.8)


def _trend_parameters_from_config(config):
    """V5 监督趋势参数；未启用时 alpha/beta 全 0。"""
    n_targets = len(TARGET_COLUMNS)
    alpha = np.asarray(
        config.get("trend_shape_alpha", [0.0] * n_targets),
        dtype=np.float64,
    )
    beta = np.asarray(
        config.get("trend_level_beta", [0.0] * n_targets),
        dtype=np.float64,
    )
    if alpha.shape != (n_targets,):
        alpha = np.zeros(n_targets, dtype=np.float64)
    if beta.shape != (n_targets,):
        beta = np.zeros(n_targets, dtype=np.float64)
    return (
        np.clip(alpha, 0.0, 1.5),
        np.clip(beta, 0.0, 0.5),
    )


def _required_lgb_output_indices(lgb_step: np.ndarray):
    """
    y 展平顺序为 [step0六变量, step1六变量, ...]。
    可按“具体步 + 具体变量”跳过 LGB 权重为 0 的输出。
    """
    n_targets = len(TARGET_COLUMNS)
    required = []
    for step in range(HORIZON):
        for j in range(n_targets):
            if float(lgb_step[step, j]) > 1e-12:
                required.append(step * n_targets + j)
    return required


def _predict_lgb_sparse_scaled(
    model_lgb,
    X_lgb: np.ndarray,
    output_dim: int,
    required_indices,
) -> np.ndarray:
    estimators = getattr(model_lgb, "estimators_", None)
    if estimators is None or len(estimators) != output_dim:
        pred = np.asarray(model_lgb.predict(X_lgb), dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        return pred

    pred_scaled = np.zeros((1, output_dim), dtype=np.float64)

    def predict_one(index):
        value = estimators[index].predict(X_lgb)
        return index, float(np.asarray(value).reshape(-1)[0])

    for index, value in LGB_INFER_EXECUTOR.map(
        predict_one,
        required_indices,
    ):
        pred_scaled[0, index] = value

    return pred_scaled


def _predict_supervised_trend(features, config):
    """
    返回 V5 的 (linear_displacement, zero_endpoint_shape, seconds)。
    趋势模型直接预测逐步一阶差分，再把累计位移拆成线性终点和局部形状。
    """
    alpha, beta = _trend_parameters_from_config(config)
    if not (np.any(alpha > 1e-12) or np.any(beta > 1e-12)):
        zeros = np.zeros((HORIZON, len(TARGET_COLUMNS)), dtype=np.float64)
        return zeros, zeros, 0.0, 0

    started = time.perf_counter()
    model, scalers = load_trend_model()
    X = scalers["scaler_X"].transform(features.reshape(1, -1))
    pred_scaled = np.asarray(model.predict(X))
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(1, -1)
    step_diff = scalers["scaler_y"].inverse_transform(pred_scaled)[0].reshape(
        HORIZON, len(TARGET_COLUMNS)
    )

    cumulative = np.cumsum(step_diff, axis=0)
    frac = (
        np.arange(1, HORIZON + 1, dtype=np.float64) / float(HORIZON)
    ).reshape(HORIZON, 1)
    linear = frac * cumulative[-1:].copy()
    shape = cumulative - linear

    return (
        linear,
        shape,
        time.perf_counter() - started,
        int(np.count_nonzero(alpha > 1e-12) + np.count_nonzero(beta > 1e-12)),
    )


def predict_future(
    history_df: pd.DataFrame,
    return_timings: bool = False,
):
    """
    预测未来 96 步绝对值。

    V3 主干：LGB/XGB 分段融合 + robust trend baseline + delta gain。
    V4（若配置接受）：保守 adaptive pattern 注入。
    V5（若配置接受）：监督式一阶差分 XGBoost 只补局部峰谷/波动和少量整体位移。

    return_timings=False 时仅返回 ndarray。
    return_timings=True 时返回 (pred, timings)。
    """
    total_started = time.perf_counter()

    clean_started = time.perf_counter()
    history_clean = clean_sequence(history_df)
    clean_seconds = time.perf_counter() - clean_started

    feature_started = time.perf_counter()
    features = extract_inference_features(history_clean)
    feature_seconds = time.perf_counter() - feature_started

    (
        model_lgb,
        model_xgb,
        scalers_lgb,
        scalers_xgb,
        ensemble_config,
    ) = load_models()

    lgb_step, base_step, gain_step = _stepwise_parameters_from_config(
        ensemble_config
    )
    pattern_step = _pattern_weights_from_config(ensemble_config)
    trend_alpha, trend_beta = _trend_parameters_from_config(ensemble_config)

    # ------------------------------------------------------------------
    # LightGBM：只计算真正参与融合的 step-target 输出。
    # ------------------------------------------------------------------
    lgb_started = time.perf_counter()
    X_lgb = scalers_lgb["scaler_X"].transform(
        features.reshape(1, -1)
    )
    output_dim = HORIZON * len(TARGET_COLUMNS)
    required_lgb_indices = _required_lgb_output_indices(lgb_step)

    delta_lgb_scaled = _predict_lgb_sparse_scaled(
        model_lgb,
        X_lgb,
        output_dim,
        required_lgb_indices,
    )
    delta_lgb = scalers_lgb["scaler_y"].inverse_transform(
        delta_lgb_scaled
    )[0].reshape(HORIZON, len(TARGET_COLUMNS))
    lgb_seconds = time.perf_counter() - lgb_started

    # ------------------------------------------------------------------
    # XGBoost：原生 576 维多输出，一次 predict。
    # ------------------------------------------------------------------
    xgb_started = time.perf_counter()
    X_xgb = scalers_xgb["scaler_X"].transform(
        features.reshape(1, -1)
    )
    delta_xgb_scaled = np.asarray(model_xgb.predict(X_xgb))
    if delta_xgb_scaled.ndim == 1:
        delta_xgb_scaled = delta_xgb_scaled.reshape(1, -1)
    delta_xgb = scalers_xgb["scaler_y"].inverse_transform(
        delta_xgb_scaled
    )[0].reshape(HORIZON, len(TARGET_COLUMNS))
    xgb_seconds = time.perf_counter() - xgb_started

    last = history_clean.iloc[-1][TARGET_COLUMNS].to_numpy(dtype=np.float64)
    pred_lgb = delta_lgb + last.reshape(1, -1)
    pred_xgb = delta_xgb + last.reshape(1, -1)

    ml_pred = (
        lgb_step * pred_lgb
        + (1.0 - lgb_step) * pred_xgb
    )

    baseline_started = time.perf_counter()
    baseline = robust_trend_forecast(history_clean, HORIZON)
    baseline_seconds = time.perf_counter() - baseline_started

    fuse_started = time.perf_counter()
    pred = (
        (1.0 - base_step) * ml_pred
        + base_step * baseline
    )
    pred = last.reshape(1, -1) + gain_step * (
        pred - last.reshape(1, -1)
    )
    fuse_seconds = time.perf_counter() - fuse_started

    # ------------------------------------------------------------------
    # V4 pattern trend：只有配置里实际使用 pattern 时才计算。
    # ------------------------------------------------------------------
    pattern_seconds = 0.0
    pattern_outputs = int(np.count_nonzero(pattern_step > 1e-12))
    if pattern_outputs > 0:
        pattern_started = time.perf_counter()
        pattern_pred = adaptive_pattern_forecast(history_clean, HORIZON)
        pred = (
            (1.0 - pattern_step) * pred
            + pattern_step * pattern_pred
        )
        pattern_seconds = time.perf_counter() - pattern_started

    # ------------------------------------------------------------------
    # V5 supervised trend：alpha 只注入零终点 shape，beta 只修正整体位移。
    # ------------------------------------------------------------------
    trend_linear, trend_shape, trend_seconds, trend_outputs = _predict_supervised_trend(
        features,
        ensemble_config,
    )
    if trend_outputs > 0:
        base_delta = pred - last.reshape(1, -1)
        pred = last.reshape(1, -1) + (
            (1.0 - trend_beta.reshape(1, -1)) * base_delta
            + trend_beta.reshape(1, -1) * trend_linear
            + trend_alpha.reshape(1, -1) * trend_shape
        )

    pred = np.asarray(pred, dtype=np.float64)
    bad = ~np.isfinite(pred)
    if np.any(bad):
        fallback = np.tile(last, (HORIZON, 1))
        pred[bad] = fallback[bad]

    total_seconds = time.perf_counter() - total_started

    if not return_timings:
        return pred

    timings = {
        "clean": clean_seconds,
        "feature": feature_seconds,
        "lgb": lgb_seconds,
        "xgb": xgb_seconds,
        "baseline": baseline_seconds,
        "fuse": fuse_seconds,
        "pattern": pattern_seconds,
        "trend": trend_seconds,
        "total": total_seconds,
        "lgb_outputs": len(required_lgb_indices),
        "pattern_outputs": pattern_outputs,
        "trend_outputs": trend_outputs,
        "lgb_infer_threads": LGB_INFER_THREADS,
        "ensemble_version": ensemble_config.get("version"),
    }
    return pred, timings
