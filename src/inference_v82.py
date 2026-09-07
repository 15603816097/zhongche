from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.inference import predict_future as predict_future_v8
from src.inference_v81 import (
    V81_PATCHTST_DEVICE,
    V81_TEMPERATURE_ENABLED,
    apply_patchtst_temperature_candidate,
)


V82_TREND_GATE = os.getenv("V82_TREND_GATE", "1").lower() not in {"0", "false", "off"}
V82_MIN_WEIGHT = float(os.getenv("V82_MIN_WEIGHT", "0.10"))
V82_MAX_WEIGHT = float(os.getenv("V82_MAX_WEIGHT", "0.30"))


def _dynamic_temperature_weight(history_df: pd.DataFrame) -> float:
    """Increase PatchTST contribution only when temperature trend changes."""
    temp = history_df["temperature_c"].astype(float).values
    if len(temp) < 3:
        return V82_MIN_WEIGHT
    diff = np.diff(temp)
    volatility = float(np.std(diff) / (np.mean(np.abs(temp)) + 1e-6))
    weight = V82_MIN_WEIGHT + volatility * 2.0
    return float(np.clip(weight, V82_MIN_WEIGHT, V82_MAX_WEIGHT))


def predict_future(history_df: pd.DataFrame, return_timings: bool = False):
    if return_timings:
        base, timing = predict_future_v8(history_df, return_timings=True)
        timing = dict(timing)
    else:
        base = predict_future_v8(history_df, return_timings=False)
        timing = None

    pred = np.asarray(base, dtype=np.float64).copy()
    weight = _dynamic_temperature_weight(history_df) if V82_TREND_GATE else 0.15

    if V81_TEMPERATURE_ENABLED:
        try:
            pred, patch_time = apply_patchtst_temperature_candidate(
                history_df,
                pred,
                weight=weight,
                device=V81_PATCHTST_DEVICE,
            )
        except Exception:
            patch_time = 0.0

    if not return_timings:
        return pred

    timing["v82_temperature_weight"] = weight
    timing["v82_patchtst_seconds"] = patch_time
    return pred, timing
