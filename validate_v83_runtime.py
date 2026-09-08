from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.feature_engineer import robust_trend_forecast
from src.inference import predict_future as predict_future_v8
from src.inference_v83 import apply_v83_postprocessor, preload_v83_models
from src.trend_pattern import adaptive_pattern_forecast
from v83_final_sprint_diagnostic import apply_params


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
CANDIDATE_PATH = ROOT / "models" / "v83_final_safe_candidate.json"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def main() -> int:
    required = [
        CORPUS_PATH,
        CANDIDATE_PATH,
        MODEL_DIR / "ensemble_config.pkl",
        MODEL_DIR / "model_lgb.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "model_xgb.pkl",
        MODEL_DIR / "scaler_xgb.pkl",
        MODEL_DIR / "model_pca_xgb.pkl",
        MODEL_DIR / "preprocess_pca_xgb.pkl",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))

    cfg = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    if not bool(cfg.get("offline_gate_pass", False)):
        raise RuntimeError("safe candidate offline gate is not PASS")

    with open(MODEL_DIR / "ensemble_config.pkl", "rb") as f:
        ensemble = pickle.load(f)
    if int(ensemble.get("version", -1)) != 8:
        raise RuntimeError(f"expected V8 ensemble, got {ensemble.get('version')}")

    print("=" * 108)
    print("V8.3 SAFE STRICT RUNTIME PARITY VALIDATION")
    print("=" * 108)
    print(f"enabled targets      : {cfg['enabled_targets']}")
    print(f"offline rmse ratio   : {float(cfg['global_rmse_ratio']):.6f}")
    print(f"offline proxy gain   : {float(cfg['global_proxy_gain_pct']):+.2f}%")

    preload_v83_models()

    data = np.load(CORPUS_PATH, allow_pickle=False)
    X = data["X"].astype(np.float64, copy=False)
    Yz = data["Y"].astype(np.float64, copy=False)
    center = data["center"].astype(np.float64, copy=False)
    scale = data["scale"].astype(np.float64, copy=False)
    group_id = data["group_id"].astype(str)
    targets = data["targets"].astype(str).tolist()
    if targets != list(TARGET_COLUMNS):
        raise RuntimeError(f"target mismatch: {targets}")

    center3 = center[:, None, :]
    scale3 = scale[:, None, :]
    histories = X * scale3 + center3
    truth = Yz * scale3 + center3

    base_all = []
    runtime_all = []
    expected_all = []
    max_parity = 0.0
    max_acoustic_change = 0.0
    acoustic_idx = TARGET_COLUMNS.index("acoustic_db")

    for i, gid in enumerate(group_id):
        hdf = pd.DataFrame(histories[i], columns=TARGET_COLUMNS)
        base = np.asarray(predict_future_v8(hdf), dtype=np.float64)
        runtime = np.asarray(apply_v83_postprocessor(hdf, base), dtype=np.float64)
        pattern = np.asarray(adaptive_pattern_forecast(hdf, HORIZON), dtype=np.float64)
        trend = np.asarray(robust_trend_forecast(hdf, HORIZON), dtype=np.float64)
        anchor = histories[i, -1]

        expected = base.copy()
        for name in cfg["enabled_targets"]:
            j = TARGET_COLUMNS.index(name)
            item = cfg["targets"][name]
            c = item["consensus"]
            params = (
                float(c["pattern_weight"]),
                float(c["trend_weight"]),
                float(c["highpass_gain"]),
                float(c["amplitude_gain"]),
                int(c["highpass_window"]),
            )
            full = apply_params(
                base[:, j], pattern[:, j], trend[:, j], float(anchor[j]), params
            )
            alpha = float(item.get("runtime_scale", 1.0))
            expected[:, j] = base[:, j] + alpha * (full - base[:, j])

        parity = float(np.max(np.abs(runtime - expected)))
        acoustic_change = float(np.max(np.abs(runtime[:, acoustic_idx] - base[:, acoustic_idx])))
        max_parity = max(max_parity, parity)
        max_acoustic_change = max(max_acoustic_change, acoustic_change)
        print(
            f"{gid:14s}: parity={parity:.3e} acoustic_change={acoustic_change:.3e} "
            f"base_rmse={rmse(base, truth[i]):.6f} v83_rmse={rmse(runtime, truth[i]):.6f}"
        )
        base_all.append(base)
        runtime_all.append(runtime)
        expected_all.append(expected)

    base_all = np.stack(base_all)
    runtime_all = np.stack(runtime_all)
    expected_all = np.stack(expected_all)

    base_flat = rmse(base_all, truth)
    runtime_flat = rmse(runtime_all, truth)
    ratio = runtime_flat / max(base_flat, 1e-12)
    recorded_ratio = float(cfg["global_rmse_ratio"])
    ratio_delta = abs(ratio - recorded_ratio)

    print("\n" + "-" * 108)
    print(f"max runtime parity   : {max_parity:.3e}")
    print(f"max acoustic change  : {max_acoustic_change:.3e}")
    print(f"flat RMSE V8->V8.3   : {base_flat:.6f} -> {runtime_flat:.6f} ratio={ratio:.6f}")
    print(f"recorded ratio       : {recorded_ratio:.6f} delta={ratio_delta:.3e}")

    gate = bool(
        max_parity <= 1e-8
        and max_acoustic_change <= 1e-12
        and ratio <= 1.005
        and ratio_delta <= 1e-8
        and np.isfinite(runtime_all).all()
    )
    print(f"V8.3 RUNTIME GATE    : {'PASS' if gate else 'REJECT'}")
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
