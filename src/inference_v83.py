from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.feature_engineer import robust_trend_forecast
from src.inference import load_models as load_v8_models
from src.inference import predict_future as predict_future_v8
from src.inference_v81 import warmup_v8_xgb_runtime
from src.trajectory_fusion import moving_average
from src.trend_pattern import adaptive_pattern_forecast
from src.v8_runtime import v8_enabled


V83_SAFE_CANDIDATE_PATH = Path(MODEL_DIR) / "v83_final_safe_candidate.json"
V83_RAW_CANDIDATE_PATH = Path(MODEL_DIR) / "v83_final_candidate.json"
_EXPECTED_BASE_VERSION = 8
_CANDIDATE_CACHE: dict[str, Any] | None = None


def _candidate_path() -> Path:
    # Final-submission default: prefer the stricter 5/5-positive shrink candidate.
    if V83_SAFE_CANDIDATE_PATH.is_file():
        return V83_SAFE_CANDIDATE_PATH
    return V83_RAW_CANDIDATE_PATH


def _load_candidate() -> dict[str, Any]:
    global _CANDIDATE_CACHE
    if _CANDIDATE_CACHE is not None:
        return _CANDIDATE_CACHE

    path = _candidate_path()
    if not path.is_file():
        raise FileNotFoundError(
            "missing V8.3 candidate config; run bash run_v83_final_sprint.sh and "
            "bash run_v83_safe_shrink.sh first"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["_runtime_candidate_path"] = str(path)
    if not bool(raw.get("offline_gate_pass", False)):
        raise RuntimeError("V8.3 candidate did not pass offline gate")
    if int(raw.get("base_version", -1)) != _EXPECTED_BASE_VERSION:
        raise RuntimeError(f"unexpected V8.3 base version: {raw.get('base_version')}")

    enabled = list(raw.get("enabled_targets", []))
    if not enabled:
        raise RuntimeError("V8.3 candidate has no enabled targets")
    unknown = sorted(set(enabled) - set(TARGET_COLUMNS))
    if unknown:
        raise RuntimeError(f"unknown V8.3 enabled targets: {unknown}")

    target_cfg = raw.get("targets", {})
    for name in enabled:
        item = target_cfg.get(name, {})
        if not bool(item.get("enabled", False)):
            raise RuntimeError(f"V8.3 enabled target {name} lacks enabled target config")
        consensus = item.get("consensus")
        if not isinstance(consensus, dict):
            raise RuntimeError(f"V8.3 enabled target {name} lacks consensus params")
        required = {
            "pattern_weight",
            "trend_weight",
            "highpass_gain",
            "amplitude_gain",
            "highpass_window",
        }
        missing = required - set(consensus)
        if missing:
            raise RuntimeError(f"V8.3 target {name} missing params: {sorted(missing)}")
        scale = float(item.get("runtime_scale", 1.0))
        if not (0.0 < scale <= 1.0):
            raise RuntimeError(f"invalid V8.3 runtime_scale for {name}: {scale}")

    _CANDIDATE_CACHE = raw
    return raw


def _apply_one(
    v8: np.ndarray,
    pattern: np.ndarray,
    trend: np.ndarray,
    anchor: float,
    params: dict[str, Any],
) -> np.ndarray:
    pw = float(params["pattern_weight"])
    tw = float(params["trend_weight"])
    hp_gain = float(params["highpass_gain"])
    amp_gain = float(params["amplitude_gain"])
    hp_window = int(params["highpass_window"])

    if pw < 0.0 or tw < 0.0 or pw + tw > 0.30 + 1e-12:
        raise RuntimeError(f"unsafe V8.3 blend weights: pattern={pw}, trend={tw}")
    if hp_window < 1:
        raise RuntimeError(f"invalid V8.3 highpass window: {hp_window}")

    mixed = (1.0 - pw - tw) * v8 + pw * pattern + tw * trend
    smooth = moving_average(mixed.reshape(1, -1), hp_window)[0]
    shaped = smooth + hp_gain * (mixed - smooth)
    out = float(anchor) + amp_gain * (shaped - float(anchor))
    return np.nan_to_num(
        out,
        nan=float(anchor),
        posinf=float(anchor),
        neginf=float(anchor),
    )


def apply_v83_postprocessor(
    history_df: pd.DataFrame,
    v8_pred: np.ndarray,
    candidate: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply fixed V8.3 consensus plus optional conservative per-target shrink."""
    cfg = _load_candidate() if candidate is None else candidate
    enabled = list(cfg["enabled_targets"])
    target_cfg = cfg["targets"]

    base = np.asarray(v8_pred, dtype=np.float64)
    pred = base.copy()
    if pred.shape != (HORIZON, len(TARGET_COLUMNS)):
        raise ValueError(f"unexpected V8 prediction shape: {pred.shape}")

    history_clean = clean_sequence(history_df)
    pattern = np.asarray(adaptive_pattern_forecast(history_clean, HORIZON), dtype=np.float64)
    trend = np.asarray(robust_trend_forecast(history_clean, HORIZON), dtype=np.float64)
    anchors = history_clean.iloc[-1][TARGET_COLUMNS].to_numpy(dtype=np.float64)

    for name in enabled:
        j = TARGET_COLUMNS.index(name)
        full = _apply_one(
            base[:, j],
            pattern[:, j],
            trend[:, j],
            float(anchors[j]),
            target_cfg[name]["consensus"],
        )
        runtime_scale = float(target_cfg[name].get("runtime_scale", 1.0))
        pred[:, j] = base[:, j] + runtime_scale * (full - base[:, j])

    return pred


def preload_v83_models() -> dict[str, Any]:
    """Preload exact V8 models, warm XGBoost, and validate V8.3 final config."""
    _, _, _, _, ensemble_config = load_v8_models()
    if not v8_enabled(ensemble_config):
        raise RuntimeError(
            f"online ensemble is not V8-compatible: version={ensemble_config.get('version')}"
        )
    candidate = _load_candidate()
    warm_seconds = warmup_v8_xgb_runtime(runs=2)
    print(
        f"[V8.3 READY] xgb_warm={warm_seconds:.3f}s "
        f"candidate={Path(candidate['_runtime_candidate_path']).name} "
        f"enabled_targets={candidate['enabled_targets']} "
        f"offline_rmse_ratio={float(candidate.get('global_rmse_ratio', float('nan'))):.5f} "
        f"proxy_gain={float(candidate.get('global_proxy_gain_pct', float('nan'))):+.2f}%"
    )
    return ensemble_config


def predict_future(history_df: pd.DataFrame, return_timings: bool = False):
    total_started = time.perf_counter()

    if return_timings:
        base, timings = predict_future_v8(history_df, return_timings=True)
        timings = dict(timings)
    else:
        base = predict_future_v8(history_df, return_timings=False)
        timings = None

    post_started = time.perf_counter()
    pred = apply_v83_postprocessor(history_df, np.asarray(base, dtype=np.float64))
    post_seconds = time.perf_counter() - post_started
    total_seconds = time.perf_counter() - total_started

    if not return_timings:
        return pred

    candidate = _load_candidate()
    timings["v83_postprocess"] = float(post_seconds)
    timings["v83_enabled_targets"] = list(candidate["enabled_targets"])
    timings["v83_candidate_file"] = Path(candidate["_runtime_candidate_path"]).name
    timings["v83_total"] = float(total_seconds)
    timings["total"] = float(total_seconds)
    return pred, timings
