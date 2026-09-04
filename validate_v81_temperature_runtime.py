from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, TARGET_COLUMNS
from evaluate_patchtst_vs_v8_official import CORPUS_PATH, PATCH_CHECKPOINT, config_from_checkpoint, gain, rmse, safe_corr
from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster
from src.deep.patchtst_temperature_runtime import (
    PATCHTST_TEMPERATURE_WEIGHT,
    TEMP_IDX,
    apply_patchtst_temperature_candidate,
    clear_patchtst_temperature_runtime,
    load_patchtst_temperature_runtime,
    normalize_history_for_patchtst,
    predict_patchtst_temperature,
)
from src.inference import predict_future
from src.v8_runtime import v8_enabled


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "models" / "deep" / "v81_temperature_runtime_validation.json"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark_patch_runtime(history_df: pd.DataFrame, device: str, repeats: int) -> dict:
    clear_patchtst_temperature_runtime()
    started = time.perf_counter()
    _, cold_seconds = predict_patchtst_temperature(history_df, device=device)
    cold_wall = time.perf_counter() - started

    warm = []
    for _ in range(repeats):
        _, seconds = predict_patchtst_temperature(history_df, device=device)
        warm.append(float(seconds))

    return {
        "device": device,
        "cold_internal_seconds": float(cold_seconds),
        "cold_wall_seconds": float(cold_wall),
        "warm_repeats": int(repeats),
        "warm_mean_seconds": float(statistics.mean(warm)),
        "warm_median_seconds": float(statistics.median(warm)),
        "warm_p95_seconds": percentile(warm, 95.0),
        "warm_max_seconds": float(max(warm)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--skip-cpu", action="store_true")
    args = parser.parse_args()

    required = [
        CORPUS_PATH,
        PATCH_CHECKPOINT,
        MODEL_DIR / "model_lgb.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "model_xgb.pkl",
        MODEL_DIR / "scaler_xgb.pkl",
        MODEL_DIR / "ensemble_config.pkl",
        MODEL_DIR / "model_pca_xgb.pkl",
        MODEL_DIR / "preprocess_pca_xgb.pkl",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print("  ", p)
        return 2

    with open(MODEL_DIR / "ensemble_config.pkl", "rb") as f:
        ensemble_config = pickle.load(f)
    if not v8_enabled(ensemble_config):
        raise RuntimeError("ensemble_config.pkl is not the current V8-compatible config")

    data = np.load(CORPUS_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32, copy=False)
    Y = data["Y"].astype(np.float64, copy=False)
    mask = data["mask"].astype(np.float32, copy=False)
    center = data["center"].astype(np.float64, copy=False)
    scale = data["scale"].astype(np.float64, copy=False)
    group_id = data["group_id"].astype(str)
    targets = data["targets"].astype(str).tolist()
    if targets != list(TARGET_COLUMNS):
        raise RuntimeError(f"target mismatch: {targets}")

    center3 = center[:, None, :]
    scale3 = scale[:, None, :]
    history_phys = X.astype(np.float64) * scale3 + center3
    true_phys = Y * scale3 + center3

    # Direct model path from the already validated corpus representation.
    direct_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PATCH_CHECKPOINT, map_location=direct_device, weights_only=False)
    direct_model = MaskedPatchTSTForecaster(config_from_checkpoint(checkpoint)).to(direct_device)
    direct_model.load_state_dict(checkpoint["model_state"])
    direct_model.eval()
    with torch.inference_mode():
        direct_z = direct_model(
            torch.from_numpy(X).to(direct_device, dtype=torch.float32),
            torch.from_numpy(mask).to(direct_device, dtype=torch.float32),
        ).cpu().numpy().astype(np.float64)
    direct_phys = direct_z * scale3 + center3

    print("=" * 108)
    print("V8.1 TEMPERATURE-ONLY RUNTIME PARITY + LATENCY VALIDATION")
    print("=" * 108)
    print(f"V8 config version  : {ensemble_config.get('version')}")
    print(f"V8 trajectory      : {ensemble_config.get('trajectory_model')}")
    print(f"PatchTST weight    : {PATCHTST_TEMPERATURE_WEIGHT:.2f}")
    print(f"direct device      : {direct_device}")
    print("IMPORTANT          : candidate-only validation; online inference/API/callback/config untouched")

    v8_phys = np.empty_like(true_phys)
    candidate_phys = np.empty_like(true_phys)
    runtime_patch_phys = np.empty((len(X), Y.shape[1]), dtype=np.float64)
    parity_errors = []
    non_temp_errors = []

    # Use candidate runtime normalization on physical histories and compare against
    # direct predictions from official_finetune_v1.npz. This catches any preprocessing drift.
    clear_patchtst_temperature_runtime()
    for i, gid in enumerate(group_id):
        history_df = pd.DataFrame(history_phys[i], columns=TARGET_COLUMNS)

        x_runtime, m_runtime, c_runtime, s_runtime = normalize_history_for_patchtst(history_df)
        x_err = float(np.max(np.abs(x_runtime.astype(np.float64) - X[i].astype(np.float64))))
        m_err = float(np.max(np.abs(m_runtime.astype(np.float64) - mask[i].astype(np.float64))))
        c_err = float(np.max(np.abs(c_runtime.astype(np.float64) - center[i].astype(np.float64))))
        s_err = float(np.max(np.abs(s_runtime.astype(np.float64) - scale[i].astype(np.float64))))

        patch_temp, patch_seconds = predict_patchtst_temperature(history_df, device=str(direct_device))
        direct_temp = direct_phys[i, :, TEMP_IDX]
        pred_err = float(np.max(np.abs(patch_temp - direct_temp)))
        parity_errors.append(pred_err)
        runtime_patch_phys[i] = patch_temp

        print(
            f"parity {gid:14s}: X={x_err:.3e} mask={m_err:.3e} center={c_err:.3e} "
            f"scale={s_err:.3e} pred_temp={pred_err:.3e} patch={patch_seconds*1000:.2f} ms"
        )

        v8 = np.asarray(predict_future(history_df), dtype=np.float64)
        cand, _ = apply_patchtst_temperature_candidate(
            history_df,
            v8,
            weight=PATCHTST_TEMPERATURE_WEIGHT,
            device=str(direct_device),
        )
        v8_phys[i] = v8
        candidate_phys[i] = cand

        other = [j for j in range(len(TARGET_COLUMNS)) if j != TEMP_IDX]
        non_temp_errors.append(float(np.max(np.abs(cand[:, other] - v8[:, other]))))

    v8_temp = v8_phys[:, :, TEMP_IDX]
    cand_temp = candidate_phys[:, :, TEMP_IDX]
    true_temp = true_phys[:, :, TEMP_IDX]
    v8_rmse = rmse(v8_temp, true_temp)
    cand_rmse = rmse(cand_temp, true_temp)
    physical_gain = gain(v8_rmse, cand_rmse)

    v8_z = (v8_phys - center3) / scale3
    cand_z = (candidate_phys - center3) / scale3
    z_gain = gain(
        rmse(v8_z[:, :, TEMP_IDX], Y[:, :, TEMP_IDX]),
        rmse(cand_z[:, :, TEMP_IDX], Y[:, :, TEMP_IDX]),
    )

    dv8 = np.diff(v8_temp, axis=1).reshape(-1)
    dcand = np.diff(cand_temp, axis=1).reshape(-1)
    dy = np.diff(true_temp, axis=1).reshape(-1)
    corr_v8 = safe_corr(dv8, dy)
    corr_cand = safe_corr(dcand, dy)
    corr_delta = corr_cand - corr_v8

    fold_gains = []
    for i, gid in enumerate(group_id):
        b = rmse(v8_temp[i], true_temp[i])
        c = rmse(cand_temp[i], true_temp[i])
        g = gain(b, c)
        fold_gains.append(g)
        print(f"gain   {gid:14s}: V8={b:.6f} V8.1={c:.6f} gain={g:+.2f}%")

    benchmark_history = pd.DataFrame(history_phys[0], columns=TARGET_COLUMNS)
    latency = {}
    if torch.cuda.is_available():
        latency["cuda"] = benchmark_patch_runtime(benchmark_history, "cuda", args.repeats)
    if not args.skip_cpu:
        latency["cpu"] = benchmark_patch_runtime(benchmark_history, "cpu", args.repeats)

    max_parity_error = float(max(parity_errors))
    max_non_temp_error = float(max(non_temp_errors))
    min_fold_gain = float(min(fold_gains))
    positive_folds = int(sum(g > 0.0 for g in fold_gains))

    # Runtime gate is intentionally independent of the current V8 wall time: PatchTST
    # overhead itself must be small enough to remain practical on both GPU and CPU.
    gpu_p95 = latency.get("cuda", {}).get("warm_p95_seconds", float("nan"))
    cpu_p95 = latency.get("cpu", {}).get("warm_p95_seconds", float("nan"))
    latency_ok = bool(
        (not np.isfinite(gpu_p95) or gpu_p95 <= 0.100)
        and (not np.isfinite(cpu_p95) or cpu_p95 <= 0.500)
    )

    gate = bool(
        max_parity_error <= 1e-4
        and max_non_temp_error == 0.0
        and positive_folds == len(group_id)
        and min_fold_gain >= 10.0
        and z_gain >= 20.0
        and physical_gain >= 20.0
        and (not np.isfinite(corr_delta) or corr_delta >= -0.005)
        and latency_ok
    )

    print("\n" + "-" * 108)
    print(f"max preprocessing/pred parity error : {max_parity_error:.3e}")
    print(f"max non-temperature output change   : {max_non_temp_error:.3e}")
    print(f"temperature positive folds          : {positive_folds}/{len(group_id)}")
    print(f"temperature min fold gain           : {min_fold_gain:+.2f}%")
    print(f"temperature pooled gain_z           : {z_gain:+.2f}%")
    print(f"temperature physical gain           : {physical_gain:+.2f}%")
    print(f"temperature diffCorr                : V8={corr_v8:+.4f} V8.1={corr_cand:+.4f} delta={corr_delta:+.4f}")
    for key, row in latency.items():
        print(
            f"PatchTST {key:4s} latency              : cold={row['cold_wall_seconds']*1000:.2f} ms "
            f"warm median={row['warm_median_seconds']*1000:.2f} ms p95={row['warm_p95_seconds']*1000:.2f} ms"
        )
    print(f"runtime latency gate                : {'PASS' if latency_ok else 'REJECT'}")
    print(f"V8.1 RUNTIME GATE                   : {'PASS' if gate else 'REJECT'}")

    result = {
        "model": "v81_v8_plus_patchtst_temperature_only_runtime_candidate",
        "patchtst_weight": PATCHTST_TEMPERATURE_WEIGHT,
        "max_temperature_runtime_parity_error": max_parity_error,
        "max_non_temperature_change": max_non_temp_error,
        "temperature_positive_folds": positive_folds,
        "temperature_min_fold_gain_pct": min_fold_gain,
        "temperature_pooled_gain_z_pct": z_gain,
        "temperature_physical_gain_pct": physical_gain,
        "v8_temperature_diff_corr": corr_v8,
        "v81_temperature_diff_corr": corr_cand,
        "diff_corr_delta": corr_delta,
        "latency": latency,
        "latency_gate_pass": latency_ok,
        "runtime_gate_pass": gate,
        "gate_rule": (
            "runtime parity <=1e-4; all non-temperature outputs exactly V8; all 5 temperature folds positive "
            "with min gain >=10%; pooled normalized/physical gain >=20%; diffCorr delta >=-0.005; "
            "PatchTST warm p95 <=100ms CUDA and <=500ms CPU when measured"
        ),
        "important_note": "No online inference, API, callback, or ensemble_config was modified.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics                            : {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
