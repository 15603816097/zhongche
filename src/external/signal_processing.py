from __future__ import annotations

from typing import Callable

import numpy as np


def _as_2d_float(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected 1D/2D array, got shape={arr.shape}")
    return arr.astype(np.float64, copy=False)


def block_reduce(
    data: np.ndarray,
    block_size: int,
    reducer: Callable[[np.ndarray], float],
    *,
    chunk_blocks: int = 256,
) -> np.ndarray:
    arr = _as_2d_float(data)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    n_blocks = arr.shape[0] // block_size
    if n_blocks <= 0:
        return np.empty(0, dtype=np.float64)

    out = np.empty(n_blocks, dtype=np.float64)
    for b0 in range(0, n_blocks, chunk_blocks):
        b1 = min(n_blocks, b0 + chunk_blocks)
        chunk = arr[b0 * block_size : b1 * block_size]
        chunk = chunk.reshape(b1 - b0, block_size, arr.shape[1])
        for i in range(chunk.shape[0]):
            out[b0 + i] = float(reducer(chunk[i]))
    return out


def block_rms(data: np.ndarray, block_size: int) -> np.ndarray:
    return block_reduce(
        data,
        block_size,
        lambda x: np.sqrt(np.mean(np.square(x, dtype=np.float64))),
    )


def block_mean(data: np.ndarray, block_size: int) -> np.ndarray:
    return block_reduce(data, block_size, lambda x: np.mean(x))


def pressure_pa_to_db_spl(rms_pa: np.ndarray, reference_pa: float = 20e-6) -> np.ndarray:
    rms_pa = np.asarray(rms_pa, dtype=np.float64)
    safe = np.maximum(rms_pa, 1e-12)
    return 20.0 * np.log10(safe / float(reference_pa))


def sampling_rate_from_increment(dt_seconds: float) -> float:
    dt = float(dt_seconds)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"invalid sampling increment: {dt_seconds}")
    return 1.0 / dt


def block_size_for_feature_hz(dt_seconds: float, feature_hz: float) -> int:
    feature_hz = float(feature_hz)
    if not np.isfinite(feature_hz) or feature_hz <= 0:
        raise ValueError("feature_hz must be > 0")
    raw_hz = sampling_rate_from_increment(dt_seconds)
    return max(1, int(round(raw_hz / feature_hz)))


def truncate_to_common_length(*arrays: np.ndarray) -> list[np.ndarray]:
    valid = [np.asarray(x) for x in arrays if x is not None]
    if not valid:
        return []
    n = min(len(x) for x in valid)
    return [x[:n] for x in valid]
