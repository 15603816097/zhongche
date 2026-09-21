from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.inference import predict_future as predict_future_v8
from v9_analog_multiscale_diagnostic import (
    AnalogConfig,
    EPS,
    ROLLING_STRIDE,
    build_bank,
    multiscale_descriptor,
    robust_scale,
)
from full_arch_stage4_confidence_ood import (
    _calibration_stats,
    _distance_ood,
    _reference_neighbor_distance,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_GATE_CONFIG_PATH = ROOT / "full_arch_frozen_gate_v1.json"
GENERATED_GATE_CONFIG_PATH = MODEL_DIR / "full_arch" / "dynamic_gate_config.json"

_GATE_OVERRIDE = os.getenv("FULL_ARCH_GATE_CONFIG", "").strip()
if _GATE_OVERRIDE:
    override_path = Path(_GATE_OVERRIDE).expanduser()
    if not override_path.is_absolute():
        override_path = ROOT / override_path
    GATE_CONFIG_PATH = override_path
else:
    GATE_CONFIG_PATH = (
        FROZEN_GATE_CONFIG_PATH
        if FROZEN_GATE_CONFIG_PATH.is_file()
        else GENERATED_GATE_CONFIG_PATH
    )

_LOCK = threading.Lock()
_READY = False
_CONFIG: Dict | None = None
_HISTORIES: List[np.ndarray] | None = None
_BANKS: Dict[tuple, tuple] = {}
_CALIBRATIONS: Dict[tuple, dict] = {}


def _load_clean_histories() -> List[np.ndarray]:
    histories: List[np.ndarray] = []
    for seq_dir in sorted(Path(DATA_DIR).glob("sequence*")):
        hp = seq_dir / "history.csv"
        if not hp.is_file():
            continue
        df = pd.read_csv(hp)
        missing = [c for c in TARGET_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{seq_dir.name}: missing columns={missing}")
        clean = clean_sequence(df)
        arr = clean[TARGET_COLUMNS].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{seq_dir.name}: non-finite values after cleaning")
        histories.append(arr)
    if len(histories) < 5:
        raise RuntimeError(f"expected >=5 official histories, got {len(histories)}")
    return histories


def _prepare_query(history_df: pd.DataFrame) -> np.ndarray:
    clean = clean_sequence(history_df)
    arr = clean[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("query history contains non-finite values after cleaning")
    return arr


def _detect_known_source(query_h: np.ndarray, histories: Sequence[np.ndarray]) -> int | None:
    for i, ref in enumerate(histories):
        if len(query_h) != len(ref):
            continue
        if np.allclose(
            query_h,
            ref,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        ):
            return i
    return None


def _build_required_runtime_state(config: Dict, histories: Sequence[np.ndarray]) -> None:
    for target_idx, target in enumerate(TARGET_COLUMNS):
        tcfg = config.get("targets", {}).get(target, {})
        if not tcfg.get("enabled", False):
            continue
        for seg in tcfg.get("segments", []):
            if not seg.get("enabled", False):
                continue
            ac = seg["analog_config"]
            context = int(ac["context"])
            k = int(ac["k"])
            mode = str(ac["mode"])
            bank_key = (context, mode, target_idx)
            if bank_key not in _BANKS:
                _BANKS[bank_key] = build_bank(histories, context, target_idx, mode)
            cal_key = (context, mode, target_idx, k)
            if cal_key not in _CALIBRATIONS:
                descs, _, src_ids, _ = _BANKS[bank_key]
                ref = _reference_neighbor_distance(descs, src_ids, k)
                _CALIBRATIONS[cal_key] = _calibration_stats(ref)


def preload_full_arch_runtime() -> None:
    global _READY, _CONFIG, _HISTORIES
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        if not GATE_CONFIG_PATH.is_file():
            raise FileNotFoundError(f"missing dynamic gate config: {GATE_CONFIG_PATH}")
        config = json.loads(GATE_CONFIG_PATH.read_text(encoding="utf-8"))
        if not config.get("global_gate_pass", False):
            raise RuntimeError("dynamic gate config global_gate_pass is false")
        histories = _load_clean_histories()
        _build_required_runtime_state(config, histories)
        _CONFIG = config
        _HISTORIES = histories
        _READY = True


def _regime_ood(query_h: np.ndarray, target_idx: int, refs: Sequence[np.ndarray]) -> float:
    q = query_h[:, target_idx]
    recent = q[-48:]
    q_level = float(np.median(recent))
    q_vol = float(np.std(np.diff(recent))) if len(recent) > 2 else 0.0
    q_slope = float((recent[-1] - recent[0]) / max(len(recent) - 1, 1))

    levels = []
    vols = []
    slopes = []
    for h in refs:
        x = h[:, target_idx]
        for end in range(48, len(x) - HORIZON + 1, ROLLING_STRIDE):
            seg = x[end - 48:end]
            levels.append(float(np.median(seg)))
            vols.append(float(np.std(np.diff(seg))) if len(seg) > 2 else 0.0)
            slopes.append(float((seg[-1] - seg[0]) / max(len(seg) - 1, 1)))

    def robust_z(value: float, reference) -> float:
        ref = np.asarray(reference, dtype=np.float64)
        med = float(np.median(ref))
        q25, q75 = np.quantile(ref, [0.25, 0.75])
        sigma = max(float((q75 - q25) / 1.349), float(np.std(ref)) * 0.25, 1e-6)
        return float(abs(value - med) / sigma)

    level_z = robust_z(q_level, levels)
    vol_z = robust_z(q_vol, vols)
    slope_z = robust_z(q_slope, slopes)
    return float(
        np.clip(
            0.40 * (1.0 - math.exp(-level_z / 3.0))
            + 0.35 * (1.0 - math.exp(-vol_z / 3.0))
            + 0.25 * (1.0 - math.exp(-slope_z / 3.0)),
            0.0,
            1.0,
        )
    )


def _analog_with_confidence(
    query_h: np.ndarray,
    target_idx: int,
    cfg: AnalogConfig,
    bank,
    calibration: dict,
    histories: Sequence[np.ndarray],
) -> tuple[np.ndarray, float, float]:
    descs, deltas, src_ids, future_ends = bank
    query_ctx = query_h[-cfg.context:]
    qdesc = multiscale_descriptor(query_ctx, target_idx, cfg.mode)

    legal = np.ones(len(descs), dtype=bool)
    known_source = _detect_known_source(query_h, histories)
    if known_source is not None:
        self_mask = src_ids == known_source
        legal[self_mask] = (
            future_ends[self_mask] <= len(query_h) - int(cfg.self_guard)
        )

    legal_idx = np.where(legal)[0]
    if len(legal_idx) < cfg.k:
        raise RuntimeError(
            f"not enough legal analogs target={TARGET_COLUMNS[target_idx]} cfg={cfg}"
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

    chosen_delta = deltas[chosen]
    dz = np.sum(weights[:, None] * chosen_delta, axis=0)
    anchor = float(query_ctx[-1, target_idx])
    qscale = robust_scale(query_ctx[:, target_idx])
    pred = anchor + dz * qscale
    pred = np.nan_to_num(pred, nan=anchor, posinf=anchor, neginf=anchor)

    median_distance = float(np.median(chosen_dist))
    distance_ood = _distance_ood(median_distance, calibration)

    weighted_var = np.sum(
        weights[:, None] * (chosen_delta - dz.reshape(1, -1)) ** 2,
        axis=0,
    )
    trajectory_dispersion = float(np.sqrt(np.mean(weighted_var)))
    diversity = float(
        len(np.unique(src_ids[chosen]))
        / max(1, min(cfg.k, len(histories)))
    )
    endpoint_signs = np.sign(chosen_delta[:, -1])
    endpoint_sign_agreement = float(abs(np.sum(weights * endpoint_signs)))

    distance_conf = 1.0 - distance_ood
    dispersion_conf = float(math.exp(-0.75 * max(0.0, trajectory_dispersion)))
    diversity_factor = 0.60 + 0.40 * diversity
    sign_factor = 0.75 + 0.25 * endpoint_sign_agreement
    confidence = float(
        np.clip(
            distance_conf
            * dispersion_conf
            * diversity_factor
            * sign_factor,
            0.0,
            1.0,
        )
    )

    regime = _regime_ood(query_h, target_idx, histories)
    combined_ood = float(np.clip(0.65 * distance_ood + 0.35 * regime, 0.0, 1.0))
    return pred, confidence, combined_ood


def apply_full_arch_gate(
    history_df: pd.DataFrame,
    pred_v8: np.ndarray,
) -> tuple[np.ndarray, dict]:
    preload_full_arch_runtime()
    assert _CONFIG is not None
    assert _HISTORIES is not None

    started = time.perf_counter()
    query_h = _prepare_query(history_df)
    out = np.asarray(pred_v8, dtype=np.float64).copy()
    if out.shape != (HORIZON, len(TARGET_COLUMNS)):
        raise ValueError(f"V8 prediction shape invalid: {out.shape}")

    detail = {}
    for target_idx, target in enumerate(TARGET_COLUMNS):
        tcfg = _CONFIG["targets"].get(target, {})
        if not tcfg.get("enabled", False):
            detail[target] = {"enabled": False, "reason": "exact_v8"}
            continue

        rows = []
        for seg in tcfg.get("segments", []):
            if not seg.get("enabled", False):
                continue
            ac = seg["analog_config"]
            cfg = AnalogConfig(
                context=int(ac["context"]),
                k=int(ac["k"]),
                mode=str(ac["mode"]),
                self_guard=int(ac["self_guard"]),
            )
            bank = _BANKS[(cfg.context, cfg.mode, target_idx)]
            calibration = _CALIBRATIONS[(cfg.context, cfg.mode, target_idx, cfg.k)]
            analog, confidence, combined_ood = _analog_with_confidence(
                query_h,
                target_idx,
                cfg,
                bank,
                calibration,
                _HISTORIES,
            )

            alpha = float(seg["alpha"])
            ood_power = float(seg["ood_power"])
            weight = float(
                np.clip(
                    alpha
                    * confidence
                    * (max(0.0, 1.0 - combined_ood) ** ood_power),
                    0.0,
                    alpha,
                )
            )
            start, end = map(int, seg["segment"])
            out[start:end, target_idx] = (
                out[start:end, target_idx]
                + weight
                * (analog[start:end] - out[start:end, target_idx])
            )
            rows.append(
                {
                    "segment": [start, end],
                    "confidence": confidence,
                    "combined_ood": combined_ood,
                    "weight": weight,
                    "config": cfg.key(),
                }
            )
        detail[target] = {"enabled": True, "segments": rows}

    bad = ~np.isfinite(out)
    if np.any(bad):
        fallback = np.asarray(pred_v8, dtype=np.float64)
        out[bad] = fallback[bad]

    return out, {
        "gate_seconds": float(time.perf_counter() - started),
        "targets": detail,
    }


def predict_future(
    history_df: pd.DataFrame,
    return_timings: bool = False,
):
    total_started = time.perf_counter()
    if return_timings:
        pred_v8, timings = predict_future_v8(history_df, return_timings=True)
    else:
        pred_v8 = predict_future_v8(history_df, return_timings=False)
        timings = None

    pred, gate_info = apply_full_arch_gate(history_df, pred_v8)

    if not return_timings:
        return pred

    result = dict(timings)
    result["full_arch_gate"] = gate_info["gate_seconds"]
    result["total"] = float(time.perf_counter() - total_started)
    result["candidate"] = str(_CONFIG.get("candidate_name", "full_arch_dynamic_gate_v1")) if _CONFIG else "full_arch_dynamic_gate_v1"
    result["enabled_targets"] = list(_CONFIG.get("enabled_targets", [])) if _CONFIG else []
    result["gate_detail"] = gate_info["targets"]
    return pred, result
