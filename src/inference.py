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


_model_lgb = None
_model_xgb = None
_scalers_lgb = None
_scalers_xgb = None
_ensemble_config = None

DEFAULT_LGB_WEIGHTS = np.array(
    [0.65] * len(TARGET_COLUMNS),
    dtype=np.float64,
)
DEFAULT_BASELINE_WEIGHTS = np.array(
    [0.15] * len(TARGET_COLUMNS),
    dtype=np.float64,
)

# LightGBM 训练时的 MultiOutputRegressor 使用进程级 joblib 并行，
# 适合训练但不适合“单样本在线预测”：576 个小模型会产生明显进程调度/IPC 开销。
# 在线推理改为共享线程池，直接调用真正需要的子模型。
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


def _weights_from_config(config):
    lgb_weights = np.asarray(
        config.get("lgb_weights", DEFAULT_LGB_WEIGHTS),
        dtype=np.float64,
    )
    baseline_weights = np.asarray(
        config.get("baseline_weights", DEFAULT_BASELINE_WEIGHTS),
        dtype=np.float64,
    )

    if lgb_weights.shape != (len(TARGET_COLUMNS),):
        lgb_weights = DEFAULT_LGB_WEIGHTS.copy()
    if baseline_weights.shape != (len(TARGET_COLUMNS),):
        baseline_weights = DEFAULT_BASELINE_WEIGHTS.copy()

    return (
        np.clip(lgb_weights, 0.0, 1.0),
        np.clip(baseline_weights, 0.0, 0.8),
    )


def _required_lgb_output_indices(lgb_weights: np.ndarray):
    """
    y 的展平顺序是 [step0六变量, step1六变量, ...]。

    某个变量的 LGB 融合权重为 0 时，该变量全部 96 个 LGB 输出都不会参与
    最终结果，因此在线推理可以安全跳过这些子模型。
    """
    n_targets = len(TARGET_COLUMNS)
    needed_targets = [
        j for j, weight in enumerate(lgb_weights)
        if float(weight) > 1e-12
    ]
    return [
        step * n_targets + j
        for step in range(HORIZON)
        for j in needed_targets
    ]


def _predict_lgb_sparse_scaled(
    model_lgb,
    X_lgb: np.ndarray,
    output_dim: int,
    required_indices,
) -> np.ndarray:
    """
    对 MultiOutputRegressor 做低开销在线预测。

    不调用 model_lgb.predict()，因为保存的模型 n_jobs=6，单样本时会触发
    joblib 进程并行；这里直接在线程池中调用所需 estimator，避免进程 IPC。
    """
    estimators = getattr(model_lgb, "estimators_", None)
    if estimators is None or len(estimators) != output_dim:
        # 兼容未来模型结构变化：无法识别时退回标准 predict。
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


def predict_future(
    history_df: pd.DataFrame,
    return_timings: bool = False,
):
    """
    预测未来 96 步绝对值。

    最终预测 =
      (LGB/XGB 逐变量加权) 与 稳健趋势基线 再融合。

    return_timings=False 时保持原接口，只返回 ndarray。
    return_timings=True 时返回 (pred, timings)，用于 API 性能诊断。
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

    lgb_weights, baseline_weights = _weights_from_config(
        ensemble_config
    )

    # ------------------------------------------------------------------
    # LightGBM：仅计算最终融合真正需要的输出。
    # 当前权重 current_a=0、speed_rpm=0，因此可跳过 192/576 个子模型。
    # ------------------------------------------------------------------
    lgb_started = time.perf_counter()
    X_lgb = scalers_lgb["scaler_X"].transform(
        features.reshape(1, -1)
    )
    output_dim = HORIZON * len(TARGET_COLUMNS)
    required_lgb_indices = _required_lgb_output_indices(lgb_weights)

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
    # XGBoost：原生 576 维多输出模型，一次 GPU predict 全部输出。
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

    last = history_clean.iloc[-1][TARGET_COLUMNS].to_numpy(
        dtype=np.float64
    )
    pred_lgb = delta_lgb + last.reshape(1, -1)
    pred_xgb = delta_xgb + last.reshape(1, -1)

    ml_pred = (
        lgb_weights.reshape(1, -1) * pred_lgb
        + (1.0 - lgb_weights.reshape(1, -1)) * pred_xgb
    )

    baseline_started = time.perf_counter()
    baseline = robust_trend_forecast(
        history_clean,
        HORIZON,
    )
    baseline_seconds = time.perf_counter() - baseline_started

    fuse_started = time.perf_counter()
    pred = (
        (1.0 - baseline_weights.reshape(1, -1)) * ml_pred
        + baseline_weights.reshape(1, -1) * baseline
    )

    pred = np.asarray(pred, dtype=np.float64)
    bad = ~np.isfinite(pred)
    if np.any(bad):
        fallback = np.tile(last, (HORIZON, 1))
        pred[bad] = fallback[bad]
    fuse_seconds = time.perf_counter() - fuse_started

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
        "total": total_seconds,
        "lgb_outputs": len(required_lgb_indices),
        "lgb_infer_threads": LGB_INFER_THREADS,
    }
    return pred, timings
