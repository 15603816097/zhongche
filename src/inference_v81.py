from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd

from src.inference import load_models as load_v8_models
from src.inference import predict_future as predict_future_v8
from src.deep.patchtst_temperature_runtime import (
    PATCHTST_TEMPERATURE_WEIGHT,
    apply_patchtst_temperature_candidate,
    warmup_patchtst_temperature_runtime,
)


V81_TEMPERATURE_ENABLED = os.getenv("V81_TEMPERATURE_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
V81_TEMPERATURE_WEIGHT = float(
    os.getenv("V81_TEMPERATURE_WEIGHT", str(PATCHTST_TEMPERATURE_WEIGHT))
)
V81_PATCHTST_DEVICE = os.getenv("V81_PATCHTST_DEVICE", "auto").strip() or "auto"
V81_STRICT_CANDIDATE = os.getenv("V81_STRICT_CANDIDATE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_XGB_WARMED = False


def _xgb_feature_count(scaler_x) -> int:
    n = getattr(scaler_x, "n_features_in_", None)
    if n is not None:
        return int(n)

    for attr in ("mean_", "scale_", "var_"):
        value = getattr(scaler_x, attr, None)
        if value is not None:
            arr = np.asarray(value)
            if arr.ndim == 1 and arr.size > 0:
                return int(arr.size)

    raise RuntimeError("cannot determine XGBoost feature count from scaler_X")


def warmup_v8_xgb_runtime(*, runs: int = 2) -> float:
    """Warm the current V8 main XGBoost predictor at process startup.

    Loading the pickle is not enough when XGBoost uses CUDA: the first real
    ``predict`` can initialize the CUDA predictor/runtime and was observed to take
    about 10+ seconds on a fresh production process.  The official evaluator should
    never pay that one-time cost on its first request, so V8.1 executes a few
    harmless predictions on an all-zero vector in *scaled feature space* here.

    This does not change any model state, ensemble weights, API fields or callback
    behavior.  It only initializes the already-loaded inference runtime.
    """
    global _XGB_WARMED
    if _XGB_WARMED:
        return 0.0

    _, model_xgb, _, scalers_xgb, _ = load_v8_models()
    scaler_x = scalers_xgb["scaler_X"]
    n_features = _xgb_feature_count(scaler_x)
    x_scaled = np.zeros((1, n_features), dtype=np.float64)

    started = time.perf_counter()
    last = None
    for _ in range(max(1, int(runs))):
        last = np.asarray(model_xgb.predict(x_scaled))

    if last is None or last.size == 0 or not np.isfinite(last).all():
        raise RuntimeError("V8 XGBoost startup warmup produced invalid output")

    _XGB_WARMED = True
    return time.perf_counter() - started


def preload_v81_models() -> dict[str, Any]:
    """Preload and fully warm the V8.1 production candidate.

    Startup now warms both one-time native inference paths:
      1) current V8 main XGBoost predictor (including CUDA predictor init)
      2) frozen PatchTST temperature branch (including Transformer/CUDA kernels)

    Moving both costs into FastAPI startup keeps the first official /predict request
    on the same warm path as all later requests.  app.py, callback payloads and
    models/ensemble_config.pkl remain untouched.
    """
    _, _, _, _, ensemble_config = load_v8_models()

    xgb_warm_seconds = warmup_v8_xgb_runtime(runs=2)
    patch_warm_seconds = 0.0
    if V81_TEMPERATURE_ENABLED:
        patch_warm_seconds = warmup_patchtst_temperature_runtime(
            V81_PATCHTST_DEVICE,
            runs=2,
        )

    print(
        f"[V8.1 WARMUP] xgb={xgb_warm_seconds:.3f}s "
        f"patchtst={patch_warm_seconds:.3f}s"
    )
    return ensemble_config


def predict_future(
    history_df: pd.DataFrame,
    return_timings: bool = False,
    *,
    strict_candidate: bool | None = None,
):
    """Drop-in V8.1 candidate predictor.

    V8.1 = exact current V8 for five targets, while temperature_c is
      0.85 * V8 + 0.15 * frozen PatchTST
    by default.

    Candidate failure is fail-safe: unless strict_candidate=True, any PatchTST error
    returns the exact V8 output instead of failing the API request.
    """
    total_started = time.perf_counter()
    strict = V81_STRICT_CANDIDATE if strict_candidate is None else bool(strict_candidate)

    if return_timings:
        v8_pred, timings = predict_future_v8(history_df, return_timings=True)
        timings = dict(timings)
    else:
        v8_pred = predict_future_v8(history_df, return_timings=False)
        timings = None

    v8_pred = np.asarray(v8_pred, dtype=np.float64)
    patch_seconds = 0.0
    fallback = False
    error_text = ""

    if V81_TEMPERATURE_ENABLED:
        try:
            pred, patch_seconds = apply_patchtst_temperature_candidate(
                history_df,
                v8_pred,
                weight=V81_TEMPERATURE_WEIGHT,
                device=V81_PATCHTST_DEVICE,
            )
        except Exception as exc:
            if strict:
                raise
            pred = v8_pred.copy()
            fallback = True
            error_text = f"{type(exc).__name__}: {exc}"
    else:
        pred = v8_pred.copy()

    pred = np.asarray(pred, dtype=np.float64)
    total_seconds = time.perf_counter() - total_started

    if not return_timings:
        return pred

    timings["v81_enabled"] = bool(V81_TEMPERATURE_ENABLED)
    timings["v81_temperature_weight"] = float(V81_TEMPERATURE_WEIGHT)
    timings["v81_patchtst_device"] = str(V81_PATCHTST_DEVICE)
    timings["v81_patchtst_temperature"] = float(patch_seconds)
    timings["v81_fallback_to_v8"] = bool(fallback)
    timings["v81_error"] = error_text
    timings["v81_total"] = float(total_seconds)
    # Preserve the conventional total key for a future drop-in API integration.
    timings["total"] = float(total_seconds)
    return pred, timings
