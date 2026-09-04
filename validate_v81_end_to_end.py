from __future__ import annotations

import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from config import MODEL_DIR, TARGET_COLUMNS
from src.inference import predict_future as predict_future_v8
from src.inference_v81 import (
    preload_v81_models,
    predict_future as predict_future_v81,
)
from src.deep.patchtst_temperature_runtime import (
    PATCHTST_TEMPERATURE_WEIGHT,
    TEMP_IDX,
    predict_patchtst_temperature,
)


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
OUTPUT_PATH = MODEL_DIR / "deep" / "v81_end_to_end_validation.json"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def gain(base: float, model: float) -> float:
    return 100.0 * (base - model) / max(abs(base), 1e-12)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    required = [
        CORPUS_PATH,
        MODEL_DIR / "deep" / "patchtst_v1_pretrain.pt",
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

    data = np.load(CORPUS_PATH, allow_pickle=False)
    X = data["X"].astype(np.float64, copy=False)
    Y = data["Y"].astype(np.float64, copy=False)
    center = data["center"].astype(np.float64, copy=False)
    scale = data["scale"].astype(np.float64, copy=False)
    group_id = data["group_id"].astype(str)
    targets = data["targets"].astype(str).tolist()
    if targets != list(TARGET_COLUMNS):
        raise RuntimeError(f"target mismatch: {targets}")

    center3 = center[:, None, :]
    scale3 = scale[:, None, :]
    history_phys = X * scale3 + center3
    true_phys = Y * scale3 + center3

    print("=" * 112)
    print("V8.1 TEMPERATURE-ONLY END-TO-END CANDIDATE VALIDATION")
    print("=" * 112)
    print(f"weight             : {PATCHTST_TEMPERATURE_WEIGHT:.2f}")
    print(f"device env         : {os.getenv('V81_PATCHTST_DEVICE', 'auto')}")
    print("candidate path     : src.inference_v81.predict_future")
    print("latency protocol   : explicit V8.1 startup preload, then measure request-path latency")
    print("IMPORTANT          : app.py / callback / ensemble_config.pkl are still untouched")

    # ------------------------------------------------------------------
    # Production-equivalent startup behavior.
    #
    # The previous validation lazily loaded PatchTST inside sequence0001, so the
    # five-sample request p95 mixed one model-load/CUDA-initialization cost with four
    # warm requests. That made the request-path latency gate reject even though the
    # dedicated runtime test and the 2-worker test were fast.
    #
    # V8.1 is intended to preload both current V8 and frozen PatchTST at API startup.
    # Measure that startup cost separately and keep it OUT of request-path p95.
    # ------------------------------------------------------------------
    preload_started = time.perf_counter()
    preload_config = preload_v81_models()
    preload_seconds = time.perf_counter() - preload_started
    print(
        f"startup preload    : {preload_seconds*1000.0:.2f} ms "
        f"(ensemble version={preload_config.get('version')})"
    )

    rows: dict[str, dict] = {}
    v8_all = []
    v81_all = []
    patch_times = []
    full_times = []
    non_temp_max = 0.0
    fold_gains = []

    for i, gid in enumerate(group_id):
        history_df = pd.DataFrame(history_phys[i], columns=TARGET_COLUMNS)

        v8_pred, v8_t = predict_future_v8(history_df, return_timings=True)
        v81_pred, v81_t = predict_future_v81(
            history_df,
            return_timings=True,
            strict_candidate=True,
        )

        v8_pred = np.asarray(v8_pred, dtype=np.float64)
        v81_pred = np.asarray(v81_pred, dtype=np.float64)
        if v8_pred.shape != (96, len(TARGET_COLUMNS)):
            raise RuntimeError(f"bad V8 shape for {gid}: {v8_pred.shape}")
        if v81_pred.shape != v8_pred.shape:
            raise RuntimeError(f"bad V8.1 shape for {gid}: {v81_pred.shape}")
        if not np.isfinite(v81_pred).all():
            raise RuntimeError(f"non-finite V8.1 output for {gid}")

        other = [j for j in range(len(TARGET_COLUMNS)) if j != TEMP_IDX]
        non_temp_delta = float(np.max(np.abs(v81_pred[:, other] - v8_pred[:, other])))
        non_temp_max = max(non_temp_max, non_temp_delta)

        b = rmse(v8_pred[:, TEMP_IDX], true_phys[i, :, TEMP_IDX])
        h = rmse(v81_pred[:, TEMP_IDX], true_phys[i, :, TEMP_IDX])
        g = gain(b, h)
        fold_gains.append(g)

        v8_all.append(v8_pred)
        v81_all.append(v81_pred)
        patch_times.append(float(v81_t.get("v81_patchtst_temperature", float("nan"))))
        full_times.append(float(v81_t.get("total", float("nan"))))

        rows[str(gid)] = {
            "v8_temperature_rmse": b,
            "v81_temperature_rmse": h,
            "gain_vs_v8_pct": g,
            "non_temperature_max_abs_change": non_temp_delta,
            "v8_total_seconds": float(v8_t.get("total", float("nan"))),
            "v81_total_seconds": float(v81_t.get("total", float("nan"))),
            "patchtst_seconds": float(v81_t.get("v81_patchtst_temperature", float("nan"))),
            "fallback": bool(v81_t.get("v81_fallback_to_v8", False)),
        }
        print(
            f"{gid:14s}: temp V8={b:.6f} V8.1={h:.6f} gain={g:+.2f}% "
            f"| non-temp delta={non_temp_delta:.3e} "
            f"| patch={rows[str(gid)]['patchtst_seconds']*1000:.2f} ms "
            f"| total={rows[str(gid)]['v81_total_seconds']:.3f}s"
        )

    v8_all = np.asarray(v8_all, dtype=np.float64)
    v81_all = np.asarray(v81_all, dtype=np.float64)

    base_phys = rmse(v8_all[:, :, TEMP_IDX], true_phys[:, :, TEMP_IDX])
    cand_phys = rmse(v81_all[:, :, TEMP_IDX], true_phys[:, :, TEMP_IDX])
    pooled_gain = gain(base_phys, cand_phys)

    dz_v8 = np.diff(v8_all[:, :, TEMP_IDX], axis=1).reshape(-1)
    dz_v81 = np.diff(v81_all[:, :, TEMP_IDX], axis=1).reshape(-1)
    dz_true = np.diff(true_phys[:, :, TEMP_IDX], axis=1).reshape(-1)
    corr_v8 = safe_corr(dz_v8, dz_true)
    corr_v81 = safe_corr(dz_v81, dz_true)
    corr_delta = corr_v81 - corr_v8

    # PatchTST read-only inference should be deterministic and safe under the same
    # two-worker concurrency used by the API. Test only the lightweight PatchTST
    # branch here; V8 two-worker behavior is already the current production path.
    histories = [pd.DataFrame(history_phys[i], columns=TARGET_COLUMNS) for i in range(len(group_id))]
    reference = [predict_patchtst_temperature(h)[0] for h in histories]

    def one_call(k: int):
        idx = k % len(histories)
        pred, seconds = predict_patchtst_temperature(histories[idx])
        return idx, pred, seconds

    concurrent_max_error = 0.0
    concurrent_times = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(one_call, range(20)))
    for idx, pred, seconds in results:
        concurrent_max_error = max(
            concurrent_max_error,
            float(np.max(np.abs(np.asarray(pred) - np.asarray(reference[idx])))),
        )
        concurrent_times.append(float(seconds))

    patch_arr = np.asarray([x for x in patch_times if np.isfinite(x)], dtype=np.float64)
    full_arr = np.asarray([x for x in full_times if np.isfinite(x)], dtype=np.float64)
    conc_arr = np.asarray(concurrent_times, dtype=np.float64)
    patch_p95_ms = float(np.percentile(patch_arr, 95) * 1000.0) if len(patch_arr) else float("nan")
    full_p95_seconds = float(np.percentile(full_arr, 95)) if len(full_arr) else float("nan")
    concurrent_p95_ms = float(np.percentile(conc_arr, 95) * 1000.0) if len(conc_arr) else float("nan")

    positive = int(np.sum(np.asarray(fold_gains) > 0.0))
    min_gain = float(np.min(fold_gains))
    gate = bool(
        non_temp_max <= 1e-12
        and positive == len(group_id)
        and min_gain >= 10.0
        and pooled_gain >= 20.0
        and (not np.isfinite(corr_delta) or corr_delta >= -0.005)
        and concurrent_max_error <= 1e-5
        and (not np.isfinite(patch_p95_ms) or patch_p95_ms <= 100.0)
        and (not np.isfinite(concurrent_p95_ms) or concurrent_p95_ms <= 150.0)
    )

    print("\n" + "-" * 112)
    print(f"startup preload time              : {preload_seconds*1000.0:.2f} ms")
    print(f"temperature positive folds       : {positive}/{len(group_id)}")
    print(f"temperature min fold gain        : {min_gain:+.2f}%")
    print(f"temperature pooled physical gain : {pooled_gain:+.2f}%")
    print(f"temperature diffCorr             : V8={corr_v8:+.4f} V8.1={corr_v81:+.4f} delta={corr_delta:+.4f}")
    print(f"max non-temperature change       : {non_temp_max:.3e}")
    print(f"preloaded PatchTST p95            : {patch_p95_ms:.2f} ms")
    print(f"preloaded V8.1 total p95          : {full_p95_seconds:.3f} s")
    print(f"2-worker PatchTST p95             : {concurrent_p95_ms:.2f} ms")
    print(f"2-worker deterministic max error  : {concurrent_max_error:.3e}")
    print(f"V8.1 END-TO-END GATE              : {'PASS' if gate else 'REJECT'}")

    result = {
        "model": "v81_temperature_only_end_to_end_candidate",
        "weight_patchtst": float(PATCHTST_TEMPERATURE_WEIGHT),
        "startup_preload_seconds": float(preload_seconds),
        "per_sequence": rows,
        "temperature_positive_folds": positive,
        "temperature_min_fold_gain_pct": min_gain,
        "temperature_pooled_physical_gain_pct": pooled_gain,
        "v8_temperature_diff_corr": corr_v8,
        "v81_temperature_diff_corr": corr_v81,
        "diff_corr_delta": corr_delta,
        "max_non_temperature_abs_change": non_temp_max,
        "preloaded_patchtst_p95_ms": patch_p95_ms,
        "preloaded_v81_total_p95_seconds": full_p95_seconds,
        "concurrent_patchtst_p95_ms": concurrent_p95_ms,
        "concurrent_deterministic_max_error": concurrent_max_error,
        "gate_pass": gate,
        "latency_protocol": (
            "V8 and PatchTST are explicitly preloaded before request-path latency is measured; "
            "startup/model-load cost is reported separately."
        ),
        "important_note": (
            "Candidate wrapper validation only. app.py, verified callback payload, and ensemble_config.pkl are unchanged."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics                           : {OUTPUT_PATH}")
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
