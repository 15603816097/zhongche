from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import app_v81
from config import MODEL_DIR, TARGET_COLUMNS
from src.inference import predict_future as predict_future_v8
import src.inference_v81 as inference_v81


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
OUTPUT_PATH = MODEL_DIR / "deep" / "v81_api_candidate_validation.json"
TEMP_IDX = TARGET_COLUMNS.index("temperature_c")


def _payload_from_history(history: np.ndarray, request_id: str) -> dict:
    return {
        "requestId": request_id,
        "forecast_horizon": 96,
        "history_length": 512,
        "target_columns": list(TARGET_COLUMNS),
        "history": [
            {
                "step": int(i),
                "values": {
                    name: float(history[i, j])
                    for j, name in enumerate(TARGET_COLUMNS)
                },
            }
            for i in range(len(history))
        ],
    }


def _prediction_array(result: dict) -> np.ndarray:
    predictions = result.get("predictions", [])
    if len(predictions) != 96:
        raise RuntimeError(f"expected 96 predictions, got {len(predictions)}")
    out = np.empty((96, len(TARGET_COLUMNS)), dtype=np.float64)
    for i, row in enumerate(predictions):
        if int(row.get("step", -1)) != i:
            raise RuntimeError(f"unexpected step at row {i}: {row.get('step')}")
        values = row.get("values")
        if not isinstance(values, dict) or set(values) != set(TARGET_COLUMNS):
            raise RuntimeError(f"bad values schema at step {i}")
        for j, name in enumerate(TARGET_COLUMNS):
            out[i, j] = float(values[name])
    return out


def _check_callback_schema(result: dict) -> None:
    body = app_v81.base.build_callback_payload(
        request_id="V81_CALLBACK_SCHEMA_TEST",
        callback_token="token-test",
        code=0,
        message="success",
        predictions=result["predictions"],
    )
    if body.get("callback_token") != "token-test":
        raise RuntimeError("callback_token changed")
    results = body.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("callback results schema changed")
    one = results[0]
    if one.get("request_id") != "V81_CALLBACK_SCHEMA_TEST":
        raise RuntimeError("callback request_id schema changed")
    data = one.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("callback data missing")
    if data.get("code") != 0 or data.get("message") != "success":
        raise RuntimeError("callback data code/message changed")
    _prediction_array({"predictions": data.get("predictions", [])})


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
    center = data["center"].astype(np.float64, copy=False)
    scale = data["scale"].astype(np.float64, copy=False)
    group_id = data["group_id"].astype(str)
    history_phys = X * scale[:, None, :] + center[:, None, :]

    print("=" * 112)
    print("V8.1 API CANDIDATE INTEGRATION VALIDATION")
    print("=" * 112)
    print("API module          : app_v81:app")
    print("base app.py         : imported unchanged")
    print("callback path       : exact app.py implementation")
    print("temperature blend   : V8 + PatchTST through src.inference_v81")
    print("rollback            : app:app OR V81_TEMPERATURE_ENABLED=0")

    startup_started = time.perf_counter()
    app_v81.base.preload_latest_models()
    startup_seconds = time.perf_counter() - startup_started
    print(f"startup preload     : {startup_seconds:.3f}s")

    history_df = pd.DataFrame(history_phys[0], columns=TARGET_COLUMNS)
    payload = _payload_from_history(history_phys[0], "V81_API_SMOKE_001")

    v8_pred = np.asarray(predict_future_v8(history_df), dtype=np.float64)
    result = app_v81.base.run_one(payload)
    app_pred = _prediction_array(result)
    direct_v81 = np.asarray(
        inference_v81.predict_future(history_df, strict_candidate=True),
        dtype=np.float64,
    )

    if not np.isfinite(app_pred).all():
        raise RuntimeError("API candidate produced non-finite output")

    direct_error = float(np.max(np.abs(app_pred - direct_v81)))
    other = [j for j in range(len(TARGET_COLUMNS)) if j != TEMP_IDX]
    non_temp_change = float(np.max(np.abs(app_pred[:, other] - v8_pred[:, other])))
    temp_change = float(np.max(np.abs(app_pred[:, TEMP_IDX] - v8_pred[:, TEMP_IDX])))

    _check_callback_schema(result)

    original_apply = inference_v81.apply_patchtst_temperature_candidate
    try:
        def _forced_failure(*args, **kwargs):
            raise RuntimeError("forced PatchTST candidate failure")

        inference_v81.apply_patchtst_temperature_candidate = _forced_failure
        fallback_pred, fallback_t = inference_v81.predict_future(
            history_df,
            return_timings=True,
            strict_candidate=False,
        )
    finally:
        inference_v81.apply_patchtst_temperature_candidate = original_apply

    fallback_pred = np.asarray(fallback_pred, dtype=np.float64)
    fallback_error = float(np.max(np.abs(fallback_pred - v8_pred)))
    fallback_flag = bool(fallback_t.get("v81_fallback_to_v8", False))

    original_enabled = inference_v81.V81_TEMPERATURE_ENABLED
    try:
        inference_v81.V81_TEMPERATURE_ENABLED = False
        disabled_pred = np.asarray(inference_v81.predict_future(history_df), dtype=np.float64)
    finally:
        inference_v81.V81_TEMPERATURE_ENABLED = original_enabled
    disabled_error = float(np.max(np.abs(disabled_pred - v8_pred)))

    def one_call(k: int):
        p = _payload_from_history(history_phys[k % len(group_id)], f"V81_CONC_{k:02d}")
        started = time.perf_counter()
        r = app_v81.base.run_one(p)
        arr = _prediction_array(r)
        return arr, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=2) as ex:
        concurrent = list(ex.map(one_call, range(4)))
    concurrent_times = np.asarray([x[1] for x in concurrent], dtype=np.float64)
    concurrent_p95 = float(np.percentile(concurrent_times, 95))
    concurrent_finite = all(np.isfinite(x[0]).all() for x in concurrent)

    gate = bool(
        direct_error <= 1e-9
        and non_temp_change <= 1e-12
        and temp_change > 1e-6
        and fallback_flag
        and fallback_error <= 1e-12
        and disabled_error <= 1e-12
        and concurrent_finite
    )

    print("\n" + "-" * 112)
    print(f"app vs direct V8.1 max error      : {direct_error:.3e}")
    print(f"max non-temperature change        : {non_temp_change:.3e}")
    print(f"temperature max change vs V8      : {temp_change:.6f}")
    print("callback schema regression        : PASS")
    print(f"forced-failure fallback flag      : {fallback_flag}")
    print(f"forced-failure fallback max error : {fallback_error:.3e}")
    print(f"V81 disabled -> V8 max error      : {disabled_error:.3e}")
    print(f"2-worker app smoke finite         : {concurrent_finite}")
    print(f"2-worker app smoke p95            : {concurrent_p95:.3f}s")
    print(f"V8.1 API CANDIDATE GATE           : {'PASS' if gate else 'REJECT'}")

    result_json = {
        "candidate": "app_v81",
        "startup_preload_seconds": startup_seconds,
        "app_vs_direct_v81_max_error": direct_error,
        "max_non_temperature_change": non_temp_change,
        "temperature_max_change_vs_v8": temp_change,
        "callback_schema_pass": True,
        "forced_failure_fallback_flag": fallback_flag,
        "forced_failure_fallback_max_error": fallback_error,
        "disabled_path_max_error_vs_v8": disabled_error,
        "two_worker_finite": concurrent_finite,
        "two_worker_p95_seconds": concurrent_p95,
        "gate_pass": gate,
        "note": "app.py and its verified callback implementation are not modified; app_v81 is a candidate wrapper.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics                             : {OUTPUT_PATH}")
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
