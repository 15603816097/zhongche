from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from config import MODEL_DIR, TARGET_COLUMNS
from src.trajectory_fusion import endpoint_zero_highpass


CONFIG_PATH = MODEL_DIR / "v8_multiscale_fusion_candidate.json"

_CONFIG: Dict | None = None


def load_msf_config() -> Dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"missing multi-scale fusion config: {CONFIG_PATH}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not bool(cfg.get("offline_gate_pass", False)):
        raise RuntimeError("multi-scale fusion offline gate did not pass")

    windows = tuple(int(x) for x in cfg.get("windows", []))
    if windows != (5, 13, 33):
        raise RuntimeError(f"unexpected multi-scale windows: {windows}")

    selected = cfg.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("invalid multi-scale selected config")

    # Only targets explicitly enabled by the offline gate can change.
    enabled = set(cfg.get("enabled_targets") or [])
    for name in enabled:
        if name not in TARGET_COLUMNS:
            raise RuntimeError(f"unknown enabled target: {name}")
        item = selected.get(name) or {}
        if not bool(item.get("enabled", False)):
            raise RuntimeError(f"{name}: enabled_targets/config mismatch")
        gains = item.get("gains")
        if not isinstance(gains, list) or len(gains) != 3:
            raise RuntimeError(f"{name}: invalid gains={gains}")

    _CONFIG = cfg
    return _CONFIG


def _bands(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    hp5 = endpoint_zero_highpass(x, 5)
    hp13 = endpoint_zero_highpass(x, 13)
    hp33 = endpoint_zero_highpass(x, 33)
    short = hp5
    medium = hp13 - hp5
    long_local = hp33 - hp13
    return short, medium, long_local


def apply_multiscale_fusion(pred_v8: np.ndarray) -> tuple[np.ndarray, float, int]:
    started = time.perf_counter()
    cfg = load_msf_config()

    pred = np.asarray(pred_v8, dtype=np.float64)
    expected = (pred.shape[0], len(TARGET_COLUMNS))
    if pred.ndim != 2 or pred.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(
            f"V8 prediction shape invalid: {pred.shape}; expected (*,{len(TARGET_COLUMNS)})"
        )

    out = pred.copy()
    selected = cfg["selected"]
    enabled = list(cfg.get("enabled_targets") or [])

    for name in enabled:
        j = TARGET_COLUMNS.index(name)
        item = selected[name]
        gains = np.asarray(item["gains"], dtype=np.float64)
        short, medium, long_local = _bands(pred[:, j])
        out[:, j] = (
            pred[:, j]
            + float(gains[0]) * short
            + float(gains[1]) * medium
            + float(gains[2]) * long_local
        )

    out = np.nan_to_num(out, nan=pred, posinf=pred, neginf=pred)
    return out, time.perf_counter() - started, len(enabled)
