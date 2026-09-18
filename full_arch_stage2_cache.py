from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.feature_engineer import extract_inference_features, robust_trend_forecast
from src.inference import (
    _predict_lgb_sparse_scaled,
    _required_lgb_output_indices,
    _stepwise_parameters_from_config,
    load_models,
    predict_future,
)
from src.trajectory_fusion import endpoint_zero_highpass
from src.v8_runtime import (
    apply_v8_runtime,
    predict_pca_trajectory,
    required_lgb_targets_for_v8,
    v8_enabled,
    v8_parameters,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts" / "full_arch" / "expert_cache"
SEQUENCES = [f"sequence{i:04d}" for i in range(1, 6)]

# The decomposition path and production path execute the tree estimators twice.
# Parallel tree prediction can differ at the ~1e-6 floating-point level even when
# both paths are mathematically identical. The production V8 output is therefore
# the authoritative baseline stored in the cache; the reconstructed V8 is only a
# consistency check.
V8_REPRO_ATOL = 1e-5


def _read_sequence(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    seq_dir = DATA_DIR / name
    history_path = seq_dir / "history.csv"
    future_path = seq_dir / "future.csv"
    if not history_path.is_file() or not future_path.is_file():
        raise FileNotFoundError(f"missing official files for {name}: {seq_dir}")

    h = pd.read_csv(history_path)
    f = pd.read_csv(future_path)
    missing_h = [c for c in TARGET_COLUMNS if c not in h.columns]
    missing_f = [c for c in TARGET_COLUMNS if c not in f.columns]
    if missing_h or missing_f:
        raise RuntimeError(
            f"{name} missing target columns history={missing_h} future={missing_f}"
        )
    if len(f) != HORIZON:
        raise RuntimeError(f"{name} future length={len(f)} expected={HORIZON}")
    truth = f[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return h, truth


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def predict_expert_pack(history_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], float]:
    history_clean = clean_sequence(history_df)
    features = extract_inference_features(history_clean)
    model_lgb, model_xgb, scalers_lgb, scalers_xgb, cfg = load_models()

    if not v8_enabled(cfg):
        raise RuntimeError(
            f"frozen baseline is not V8: version={cfg.get('version')} "
            f"trajectory_model={cfg.get('trajectory_model')}"
        )

    lgb_step, base_step, gain_step = _stepwise_parameters_from_config(cfg)
    n_targets = len(TARGET_COLUMNS)
    output_dim = HORIZON * n_targets

    X_lgb = scalers_lgb["scaler_X"].transform(features.reshape(1, -1))
    required = set(_required_lgb_output_indices(lgb_step))
    for target_idx in required_lgb_targets_for_v8(cfg):
        for step in range(HORIZON):
            required.add(step * n_targets + int(target_idx))
    required = sorted(required)

    delta_lgb_scaled = _predict_lgb_sparse_scaled(
        model_lgb, X_lgb, output_dim, required
    )
    delta_lgb = scalers_lgb["scaler_y"].inverse_transform(delta_lgb_scaled)[0]
    delta_lgb = delta_lgb.reshape(HORIZON, n_targets)

    X_xgb = scalers_xgb["scaler_X"].transform(features.reshape(1, -1))
    delta_xgb_scaled = np.asarray(model_xgb.predict(X_xgb), dtype=np.float64)
    if delta_xgb_scaled.ndim == 1:
        delta_xgb_scaled = delta_xgb_scaled.reshape(1, -1)
    delta_xgb = scalers_xgb["scaler_y"].inverse_transform(delta_xgb_scaled)[0]
    delta_xgb = delta_xgb.reshape(HORIZON, n_targets)

    last = history_clean.iloc[-1][TARGET_COLUMNS].to_numpy(dtype=np.float64)
    pred_lgb = delta_lgb + last.reshape(1, -1)
    pred_xgb = delta_xgb + last.reshape(1, -1)

    pred_ml = lgb_step * pred_lgb + (1.0 - lgb_step) * pred_xgb
    pred_trend = robust_trend_forecast(history_clean, HORIZON)
    pred_v3 = (1.0 - base_step) * pred_ml + base_step * pred_trend
    pred_v3 = last.reshape(1, -1) + gain_step * (
        pred_v3 - last.reshape(1, -1)
    )

    pred_pca, _ = predict_pca_trajectory(features, last)
    pred_v8_rebuilt, _, _ = apply_v8_runtime(
        features=features,
        last_values=last,
        pred_v3=pred_v3,
        pred_lgb=pred_lgb,
        pred_xgb=pred_xgb,
        config=cfg,
    )

    # Authoritative production reference: exactly the same function used by V8 API.
    pred_v8_reference = np.asarray(predict_future(history_df), dtype=np.float64)
    max_diff = float(np.max(np.abs(pred_v8_rebuilt - pred_v8_reference)))
    if max_diff > V8_REPRO_ATOL:
        raise RuntimeError(
            "V8 decomposition mismatch is too large: "
            f"max_abs_diff={max_diff:.12g} > {V8_REPRO_ATOL:.1e}"
        )

    weights, gains, sources, windows = v8_parameters(cfg)
    low_rank = (
        (1.0 - weights.reshape(1, -1)) * pred_v3
        + weights.reshape(1, -1) * pred_pca
    )

    hp_v3 = np.zeros_like(pred_v3)
    hp_lgb = np.zeros_like(pred_v3)
    hp_xgb = np.zeros_like(pred_v3)
    hp_selected = np.zeros_like(pred_v3)
    hp_correction = np.zeros_like(pred_v3)
    source_map = {"v3": pred_v3, "lgb": pred_lgb, "xgb": pred_xgb}

    for j in range(n_targets):
        window = int(windows[j])
        hp_v3[:, j] = endpoint_zero_highpass(pred_v3[:, j], window)
        hp_lgb[:, j] = endpoint_zero_highpass(pred_lgb[:, j], window)
        hp_xgb[:, j] = endpoint_zero_highpass(pred_xgb[:, j], window)
        hp_selected[:, j] = endpoint_zero_highpass(
            source_map[sources[j]][:, j], window
        )
        hp_correction[:, j] = float(gains[j]) * hp_selected[:, j]

    out = {
        "last": last,
        "features": np.asarray(features, dtype=np.float64),
        "lgb": np.asarray(pred_lgb, dtype=np.float64),
        "xgb": np.asarray(pred_xgb, dtype=np.float64),
        "ml": np.asarray(pred_ml, dtype=np.float64),
        "trend": np.asarray(pred_trend, dtype=np.float64),
        "v3": np.asarray(pred_v3, dtype=np.float64),
        "pca": np.asarray(pred_pca, dtype=np.float64),
        "v8_low_rank": np.asarray(low_rank, dtype=np.float64),
        "hp_v3": hp_v3,
        "hp_lgb": hp_lgb,
        "hp_xgb": hp_xgb,
        "hp_selected": hp_selected,
        "hp_correction": hp_correction,
        "v8_rebuilt": np.asarray(pred_v8_rebuilt, dtype=np.float64),
        # IMPORTANT: all later stages use the production reference, never the rebuilt copy.
        "v8": pred_v8_reference,
    }
    for key, value in out.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite expert output: {key}")
    return out, max_diff


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (MODEL_DIR / "ensemble_config.pkl").open("rb") as f:
        cfg = pickle.load(f)
    if int(cfg.get("version", -1)) != 8 or not v8_enabled(cfg):
        raise RuntimeError(
            f"expected exact V8 baseline; got version={cfg.get('version')} "
            f"trajectory_model={cfg.get('trajectory_model')}"
        )

    weights, gains, sources, windows = v8_parameters(cfg)
    print("=" * 110)
    print("FULL ARCHITECTURE - STAGE 2 MULTI-EXPERT CACHE")
    print("=" * 110)
    print("baseline version     :", cfg.get("version"))
    print("trajectory model     :", cfg.get("trajectory_model"))
    print("pca weights          :", weights.tolist())
    print("highpass gains       :", gains.tolist())
    print("highpass sources     :", sources)
    print("highpass windows     :", windows.tolist())
    print("reproduction atol    :", V8_REPRO_ATOL)
    print("output               :", OUT_DIR)

    all_truth = []
    stacks: dict[str, list[np.ndarray]] = {}
    manifest = {
        "baseline": {
            "version": int(cfg.get("version")),
            "trajectory_model": cfg.get("trajectory_model"),
            "pca_blend_weights": weights.tolist(),
            "v8_highpass_gains": gains.tolist(),
            "v8_highpass_sources": list(sources),
            "v8_highpass_windows": windows.tolist(),
            "authoritative_v8": "src.inference.predict_future",
            "decomposition_check_atol": V8_REPRO_ATOL,
        },
        "target_columns": list(TARGET_COLUMNS),
        "horizon": int(HORIZON),
        "sequences": [],
    }

    repro_diffs = []
    for idx, name in enumerate(SEQUENCES, start=1):
        started = time.perf_counter()
        history, truth = _read_sequence(name)
        pack, repro_diff = predict_expert_pack(history)
        elapsed = time.perf_counter() - started
        repro_diffs.append(repro_diff)

        path = OUT_DIR / f"{name}.npz"
        np.savez_compressed(path, truth=truth, **pack)

        row = {
            "name": name,
            "history_rows": int(len(history)),
            "seconds": elapsed,
            "file": str(path.relative_to(ROOT)),
            "v8_reproduction_max_abs_diff": repro_diff,
            "v8_rmse_by_target": [
                _rmse(truth[:, j], pack["v8"][:, j])
                for j in range(len(TARGET_COLUMNS))
            ],
        }
        manifest["sequences"].append(row)
        print(
            f"[{idx}/{len(SEQUENCES)}] {name}: "
            f"history={len(history)} elapsed={elapsed:.3f}s "
            f"repro_diff={repro_diff:.3e} cache={path.name}"
        )

        all_truth.append(truth)
        for key, arr in pack.items():
            if key in {"last", "features"}:
                continue
            stacks.setdefault(key, []).append(arr)

    truth_all = np.stack(all_truth, axis=0)
    summary = {}
    for key, seq_arrays in stacks.items():
        pred_all = np.stack(seq_arrays, axis=0)
        summary[key] = {
            "flat_rmse": _rmse(truth_all, pred_all),
            "rmse_by_target": [
                _rmse(truth_all[:, :, j], pred_all[:, :, j])
                for j in range(len(TARGET_COLUMNS))
            ],
        }

    manifest["summary"] = summary
    manifest["max_v8_reproduction_abs_diff"] = float(max(repro_diffs))
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 110)
    print("EXPERT RMSE SUMMARY (diagnostic only; target scales differ)")
    print("=" * 110)
    for key in ["lgb", "xgb", "trend", "v3", "pca", "v8_low_rank", "v8"]:
        s = summary[key]
        print(
            f"{key:12s} flat={s['flat_rmse']:.6f} "
            f"per_target="
            + ", ".join(
                f"{TARGET_COLUMNS[j]}:{s['rmse_by_target'][j]:.6f}"
                for j in range(len(TARGET_COLUMNS))
            )
        )

    print("\nV8 cache source      : production predict_future() output")
    print(
        "V8 decomposition     : PASS "
        f"(max_abs_diff={max(repro_diffs):.3e} <= {V8_REPRO_ATOL:.1e})"
    )
    print("manifest             :", manifest_path)
    print("STAGE 2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
