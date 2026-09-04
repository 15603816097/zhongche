from __future__ import annotations

import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, TARGET_COLUMNS
from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster, PatchTSTConfig


PATCHTST_CHECKPOINT = MODEL_DIR / "deep" / "patchtst_v1_pretrain.pt"
PATCHTST_TEMPERATURE_WEIGHT = 0.15
PATCHTST_LOOKBACK = 512
PATCHTST_CLIP_Z = 12.0
TEMP_NAME = "temperature_c"
TEMP_IDX = TARGET_COLUMNS.index(TEMP_NAME)

_MODEL: MaskedPatchTSTForecaster | None = None
_DEVICE: torch.device | None = None


def _config_from_checkpoint(raw: dict) -> PatchTSTConfig:
    cfg = dict(raw.get("config", {}))
    allowed = {f.name for f in fields(PatchTSTConfig)}
    return PatchTSTConfig(**{k: v for k, v in cfg.items() if k in allowed})


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    requested = str(device or os.getenv("PATCHTST_DEVICE", "auto")).strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_patchtst_temperature_runtime(
    device: str | torch.device | None = None,
) -> tuple[MaskedPatchTSTForecaster, torch.device]:
    """Load the frozen PatchTST candidate once.

    This module is candidate-only. Importing it does not modify V8, the API, callback
    payloads, or ensemble_config.pkl.
    """
    global _MODEL, _DEVICE

    wanted = _resolve_device(device)
    if _MODEL is not None and _DEVICE == wanted:
        return _MODEL, _DEVICE

    if not PATCHTST_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing PatchTST checkpoint: {PATCHTST_CHECKPOINT}")

    checkpoint = torch.load(PATCHTST_CHECKPOINT, map_location=wanted, weights_only=False)
    model = MaskedPatchTSTForecaster(_config_from_checkpoint(checkpoint)).to(wanted)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _MODEL = model
    _DEVICE = wanted
    return model, wanted


def clear_patchtst_temperature_runtime() -> None:
    global _MODEL, _DEVICE
    _MODEL = None
    _DEVICE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _interp_finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.arange(len(x), dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.full_like(x, np.nan)
    if finite.sum() == 1:
        return np.full_like(x, float(x[finite][0]))
    out = x.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return out


def normalize_history_for_patchtst(
    history: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce pretrain_corpus_v1 history-only median/IQR normalization.

    Returns:
      x_z    [512, 6]
      mask   [6]
      center [6]
      scale  [6]
    """
    if isinstance(history, pd.DataFrame):
        missing = [c for c in TARGET_COLUMNS if c not in history.columns]
        if missing:
            raise ValueError(f"missing history columns: {missing}")
        values = history[TARGET_COLUMNS].tail(PATCHTST_LOOKBACK).to_numpy(dtype=np.float64)
    else:
        values = np.asarray(history, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(TARGET_COLUMNS):
            raise ValueError(f"unexpected history shape: {values.shape}")
        values = values[-PATCHTST_LOOKBACK:]

    if values.shape != (PATCHTST_LOOKBACK, len(TARGET_COLUMNS)):
        raise ValueError(
            f"PatchTST requires exactly the latest {PATCHTST_LOOKBACK} rows; got {values.shape}"
        )

    z = np.zeros_like(values, dtype=np.float32)
    mask = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
    center = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
    scale = np.ones(len(TARGET_COLUMNS), dtype=np.float32)

    for j in range(len(TARGET_COLUMNS)):
        h = values[:, j]
        finite = np.isfinite(h)
        if finite.sum() < max(8, int(0.80 * PATCHTST_LOOKBACK)):
            continue

        valid = h[finite]
        q25, med, q75 = np.quantile(valid, [0.25, 0.50, 0.75])
        s = float(q75 - q25)
        if not np.isfinite(s) or s < 1e-8:
            s = float(np.std(valid))
        if not np.isfinite(s) or s < 1e-8:
            s = max(abs(float(med)) * 1e-3, 1.0)

        filled = _interp_finite(h)
        channel_z = np.clip((filled - float(med)) / s, -PATCHTST_CLIP_Z, PATCHTST_CLIP_Z)
        z[:, j] = channel_z.astype(np.float32)
        mask[j] = 1.0
        center[j] = float(med)
        scale[j] = float(s)

    return z, mask, center, scale


def predict_patchtst_temperature(
    history: pd.DataFrame | np.ndarray,
    *,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """Predict only temperature_c in physical units for the next 96 steps."""
    started = time.perf_counter()
    x_z, mask, center, scale = normalize_history_for_patchtst(history)
    if mask[TEMP_IDX] <= 0.5:
        raise RuntimeError("temperature_c does not have enough finite history values")

    model, runtime_device = load_patchtst_temperature_runtime(device)
    with torch.inference_mode():
        tx = torch.from_numpy(x_z[None]).to(runtime_device, dtype=torch.float32)
        tm = torch.from_numpy(mask[None]).to(runtime_device, dtype=torch.float32)
        pred_z = model(tx, tm)[0, :, TEMP_IDX].detach().cpu().numpy().astype(np.float64)

    pred_phys = pred_z * float(scale[TEMP_IDX]) + float(center[TEMP_IDX])
    return pred_phys, time.perf_counter() - started


def apply_patchtst_temperature_candidate(
    history: pd.DataFrame | np.ndarray,
    v8_prediction: np.ndarray,
    *,
    weight: float = PATCHTST_TEMPERATURE_WEIGHT,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """Return V8.1 candidate output: only temperature_c is blended.

    All five non-temperature target columns are copied bit-for-bit from v8_prediction.
    """
    v8 = np.asarray(v8_prediction, dtype=np.float64)
    if v8.ndim != 2 or v8.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(f"unexpected V8 prediction shape: {v8.shape}")
    if v8.shape[0] != 96:
        raise ValueError(f"expected 96 forecast steps, got {v8.shape[0]}")

    w = float(np.clip(weight, 0.0, 1.0))
    patch_temp, seconds = predict_patchtst_temperature(history, device=device)
    if patch_temp.shape != (v8.shape[0],):
        raise RuntimeError(f"unexpected PatchTST temperature shape: {patch_temp.shape}")

    out = v8.copy()
    out[:, TEMP_IDX] = (1.0 - w) * v8[:, TEMP_IDX] + w * patch_temp
    return out, seconds
