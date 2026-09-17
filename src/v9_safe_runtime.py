from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.inference import predict_future as predict_future_v8
from v9_analog_multiscale_diagnostic import (
    AnalogConfig,
    EPS,
    build_bank,
    multiscale_descriptor,
    robust_scale,
)


TEMP_NAME = "temperature_c"
PRESSURE_NAME = "pressure_kpa"
TEMP_IDX = TARGET_COLUMNS.index(TEMP_NAME)
PRESSURE_IDX = TARGET_COLUMNS.index(PRESSURE_NAME)

# Frozen after V9-Safe offline safety diagnostic.
TEMP_CFG = AnalogConfig(context=96, k=8, mode="target", self_guard=96)
PRESSURE_CFG = AnalogConfig(context=144, k=8, mode="multi", self_guard=32)
TEMP_ALPHA = 0.075
PRESSURE_ALPHA = 0.050

_LOCK = threading.Lock()
_HISTORIES: List[np.ndarray] | None = None
_TEMP_BANK = None
_PRESSURE_BANK = None


def _load_histories_only() -> List[np.ndarray]:
    histories: List[np.ndarray] = []
    for seq_dir in sorted(Path(DATA_DIR).glob("sequence*")):
        hp = seq_dir / "history.csv"
        if not hp.is_file():
            continue
        df = pd.read_csv(hp)
        missing = [c for c in TARGET_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{seq_dir.name}: missing history columns={missing}")
        arr = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
        if len(arr) < PRESSURE_CFG.context + HORIZON + PRESSURE_CFG.self_guard:
            raise RuntimeError(
                f"{seq_dir.name}: history too short for V9-Safe bank: {len(arr)}"
            )
        histories.append(arr)
    if len(histories) < 5:
        raise RuntimeError(f"expected >=5 official history sequences, got {len(histories)}")
    return histories


def preload_v9_safe_runtime() -> None:
    global _HISTORIES, _TEMP_BANK, _PRESSURE_BANK
    if _HISTORIES is not None:
        return
    with _LOCK:
        if _HISTORIES is not None:
            return
        histories = _load_histories_only()
        temp_bank = build_bank(histories, TEMP_CFG.context, TEMP_IDX, TEMP_CFG.mode)
        pressure_bank = build_bank(
            histories, PRESSURE_CFG.context, PRESSURE_IDX, PRESSURE_CFG.mode
        )
        _HISTORIES = histories
        _TEMP_BANK = temp_bank
        _PRESSURE_BANK = pressure_bank


def _prepare_query(history_df: pd.DataFrame) -> np.ndarray:
    raw = history_df.reindex(columns=TARGET_COLUMNS).to_numpy(dtype=np.float64)
    if np.isfinite(raw).all():
        return raw
    clean = clean_sequence(history_df)
    return clean[TARGET_COLUMNS].to_numpy(dtype=np.float64)


def _detect_known_source(
    query_h: np.ndarray,
    histories: Sequence[np.ndarray],
    context: int,
) -> int | None:
    """Detect exact official-history queries so local runtime reproduces diagnostic self-guard."""
    for i, ref in enumerate(histories):
        if len(query_h) != len(ref):
            continue
        if np.allclose(
            query_h[-context:],
            ref[-context:],
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        ):
            return i
    return None


def _predict_from_bank(
    query_h: np.ndarray,
    target_idx: int,
    cfg: AnalogConfig,
    bank,
    histories: Sequence[np.ndarray],
) -> np.ndarray:
    descs, deltas, src_ids, future_ends = bank
    if len(query_h) < cfg.context:
        raise ValueError(
            f"history too short for {cfg.key()}: actual={len(query_h)}"
        )

    query_ctx = query_h[-cfg.context:]
    qdesc = multiscale_descriptor(query_ctx, target_idx, cfg.mode)

    legal = np.ones(len(descs), dtype=bool)
    known_source = _detect_known_source(query_h, histories, cfg.context)
    if known_source is not None:
        self_mask = src_ids == known_source
        legal[self_mask] = (
            future_ends[self_mask] <= len(query_h) - int(cfg.self_guard)
        )

    legal_idx = np.where(legal)[0]
    if len(legal_idx) < cfg.k:
        raise RuntimeError(
            f"not enough legal analogs for {cfg.key()} target={TARGET_COLUMNS[target_idx]}"
        )

    diff = descs[legal_idx] - qdesc.reshape(1, -1)
    dist = np.sqrt(np.mean(diff * diff, axis=1))
    order = np.argsort(dist)[: cfg.k]
    chosen = legal_idx[order]
    chosen_dist = dist[order]

    denom = max(float(np.median(chosen_dist)), 1e-6)
    weights = np.exp(-chosen_dist / denom)
    if not np.isfinite(weights).all() or float(weights.sum()) <= EPS:
        weights = np.ones_like(chosen_dist)
    weights = weights / weights.sum()

    dz = np.sum(weights[:, None] * deltas[chosen], axis=0)
    anchor = float(query_ctx[-1, target_idx])
    qscale = robust_scale(query_ctx[:, target_idx])
    pred = anchor + dz * qscale
    return np.nan_to_num(pred, nan=anchor, posinf=anchor, neginf=anchor)


def apply_v9_safe(history_df: pd.DataFrame, pred_v8: np.ndarray) -> Tuple[np.ndarray, float]:
    preload_v9_safe_runtime()
    assert _HISTORIES is not None
    assert _TEMP_BANK is not None
    assert _PRESSURE_BANK is not None

    started = time.perf_counter()
    query_h = _prepare_query(history_df)

    temp_analog = _predict_from_bank(
        query_h, TEMP_IDX, TEMP_CFG, _TEMP_BANK, _HISTORIES
    )
    pressure_analog = _predict_from_bank(
        query_h, PRESSURE_IDX, PRESSURE_CFG, _PRESSURE_BANK, _HISTORIES
    )

    out = np.asarray(pred_v8, dtype=np.float64).copy()
    if out.shape != (HORIZON, len(TARGET_COLUMNS)):
        raise ValueError(f"V8 prediction shape invalid: {out.shape}")

    out[:, TEMP_IDX] = (
        (1.0 - TEMP_ALPHA) * out[:, TEMP_IDX]
        + TEMP_ALPHA * temp_analog
    )
    out[:, PRESSURE_IDX] = (
        (1.0 - PRESSURE_ALPHA) * out[:, PRESSURE_IDX]
        + PRESSURE_ALPHA * pressure_analog
    )

    bad = ~np.isfinite(out)
    if np.any(bad):
        fallback = np.asarray(pred_v8, dtype=np.float64)
        out[bad] = fallback[bad]

    return out, time.perf_counter() - started


def predict_future(
    history_df: pd.DataFrame,
    return_timings: bool = False,
):
    """Exact V8 plus frozen V9-Safe analog correction on temperature and pressure only."""
    total_started = time.perf_counter()

    if return_timings:
        pred_v8, timings = predict_future_v8(history_df, return_timings=True)
    else:
        pred_v8 = predict_future_v8(history_df, return_timings=False)
        timings = None

    pred, analog_seconds = apply_v9_safe(history_df, pred_v8)

    if not return_timings:
        return pred

    result = dict(timings)
    result["analog"] = float(analog_seconds)
    result["total"] = float(time.perf_counter() - total_started)
    result["candidate"] = "v9_safe_temperature_pressure_analog"
    result["temperature_alpha"] = TEMP_ALPHA
    result["pressure_alpha"] = PRESSURE_ALPHA
    return pred, result
