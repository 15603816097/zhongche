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


def preload_v81_models() -> dict[str, Any]:
    """Preload V8 and fully warm the frozen PatchTST temperature candidate.

    Loading weights alone is not enough for CUDA request latency: the first real
    Transformer forward initializes kernels and can be much slower than steady state.
    Therefore V8.1 startup performs warmup forward passes here, moving that one-time
    cost out of the first /predict request.

    This remains outside app.py for now and does not alter callback payloads or
    models/ensemble_config.pkl.
    """
    _, _, _, _, ensemble_config = load_v8_models()
    if V81_TEMPERATURE_ENABLED:
        warmup_patchtst_temperature_runtime(V81_PATCHTST_DEVICE, runs=2)
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
