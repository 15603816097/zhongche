import numpy as np

from config import LOOKBACK, TARGET_COLUMNS
from src.feature_engineer import extract_features_from_array


PERTURBATIONS = (
    "noise",
    "missing_random",
    "missing_block",
    "bias",
    "drift",
)


def raw_windows_from_features(X: np.ndarray) -> np.ndarray:
    raw_dim = LOOKBACK * len(TARGET_COLUMNS)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < raw_dim:
        raise ValueError(f"X shape 异常: {X.shape}")
    return X[:, :raw_dim].reshape(-1, LOOKBACK, len(TARGET_COLUMNS)).copy()


def _interp_nan_column(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    good = np.isfinite(x)
    if np.all(good):
        return x
    if not np.any(good):
        return np.zeros_like(x)
    idx = np.arange(len(x), dtype=np.float64)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _repair_missing(raw: np.ndarray) -> np.ndarray:
    out = np.asarray(raw, dtype=np.float64).copy()
    for j in range(out.shape[1]):
        out[:, j] = _interp_nan_column(out[:, j])
    return out


def _scales(raw: np.ndarray):
    raw = np.asarray(raw, dtype=np.float64)
    recent = raw[-48:]
    level = np.maximum(np.median(np.abs(recent), axis=0), 1.0)
    value_std = np.std(recent, axis=0)
    diff_std = np.std(np.diff(recent, axis=0), axis=0)
    local = np.maximum(diff_std, 0.02 * value_std)
    local = np.maximum(local, 2e-4 * level)
    return level, value_std, local


def perturb_window(raw: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(raw, dtype=np.float64).copy()
    _, value_std, local = _scales(out)
    n, d = out.shape

    if kind == "noise":
        sigma = 0.30 * local
        out += rng.normal(0.0, 1.0, size=out.shape) * sigma[None, :]

    elif kind == "missing_random":
        mask = rng.random(out.shape) < 0.045
        mask[-1, :] = False
        out[mask] = np.nan
        out = _repair_missing(out)

    elif kind == "missing_block":
        n_vars = int(rng.integers(1, min(4, d) + 1))
        cols = rng.choice(d, size=n_vars, replace=False)
        block = int(rng.integers(6, 17))
        start = int(rng.integers(max(0, n - 64), max(1, n - block)))
        end = min(n - 1, start + block)
        out[start:end, cols] = np.nan
        out = _repair_missing(out)

    elif kind == "bias":
        n_vars = int(rng.integers(1, min(4, d) + 1))
        cols = rng.choice(d, size=n_vars, replace=False)
        sign = rng.choice([-1.0, 1.0], size=n_vars)
        magnitude = np.maximum(0.10 * value_std[cols], 1.2 * local[cols])
        out[:, cols] += sign[None, :] * magnitude[None, :]

    elif kind == "drift":
        n_vars = int(rng.integers(1, min(4, d) + 1))
        cols = rng.choice(d, size=n_vars, replace=False)
        sign = rng.choice([-1.0, 1.0], size=n_vars)
        end_mag = np.maximum(0.16 * value_std[cols], 1.8 * local[cols])
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
        out[:, cols] += ramp * sign[None, :] * end_mag[None, :]

    else:
        raise ValueError(f"未知 perturbation: {kind}")

    return np.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)


def perturb_features(
    X: np.ndarray,
    kinds=PERTURBATIONS,
    seed: int = 15001,
) -> np.ndarray:
    raw = raw_windows_from_features(X)
    kinds = tuple(kinds)
    if not kinds:
        raise ValueError("kinds 不能为空")

    rng = np.random.default_rng(seed)
    out = np.empty((len(raw), 1242), dtype=np.float32)
    for i in range(len(raw)):
        kind = kinds[i % len(kinds)]
        perturbed = perturb_window(raw[i], kind, rng)
        out[i] = extract_features_from_array(perturbed)
    return out
