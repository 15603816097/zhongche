from __future__ import annotations

import gc
import os
import sys

import torch

import app_v81
import src.inference as inference_v8
from src.deep.patchtst_temperature_runtime import clear_patchtst_temperature_runtime
import validate_v81_api_candidate as validator


def _shutdown_executor(name: str, executor) -> None:
    if executor is None:
        return
    try:
        executor.shutdown(wait=True, cancel_futures=True)
        print(f"[CLEANUP] {name}: shutdown OK")
    except TypeError:
        executor.shutdown(wait=True)
        print(f"[CLEANUP] {name}: shutdown OK")
    except Exception as exc:
        print(f"[CLEANUP] {name}: {type(exc).__name__}: {exc}")


def _cleanup_runtime() -> None:
    """Quiesce validation-only thread/CUDA resources before process exit.

    The short-lived validator combines LightGBM thread pools, XGBoost CUDA and
    PyTorch CUDA in one interpreter. All functional gates can pass, yet some native
    library finalizers may still abort during normal CPython teardown. We therefore
    clean up explicitly and then let the entry point terminate with os._exit(), which
    skips CPython/native destructor finalization after all results have been flushed.

    This workaround is only for the synthetic validator. The real FastAPI service is
    long-lived and is not changed by this file.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:
        print(f"[CLEANUP] cuda synchronize before shutdown: {exc}")

    _shutdown_executor(
        "app.PREDICT_EXECUTOR",
        getattr(app_v81.base, "PREDICT_EXECUTOR", None),
    )
    _shutdown_executor(
        "app.CALLBACK_EXECUTOR",
        getattr(app_v81.base, "CALLBACK_EXECUTOR", None),
    )
    _shutdown_executor(
        "inference.LGB_INFER_EXECUTOR",
        getattr(inference_v8, "LGB_INFER_EXECUTOR", None),
    )

    try:
        clear_patchtst_temperature_runtime()
        print("[CLEANUP] PatchTST runtime cleared")
    except Exception as exc:
        print(f"[CLEANUP] PatchTST clear: {type(exc).__name__}: {exc}")

    for name in (
        "_model_lgb",
        "_model_xgb",
        "_scalers_lgb",
        "_scalers_xgb",
        "_ensemble_config",
        "_model_trend_xgb",
        "_scalers_trend_xgb",
    ):
        if hasattr(inference_v8, name):
            setattr(inference_v8, name, None)

    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("[CLEANUP] CUDA/runtime cleanup OK")
    except Exception as exc:
        print(f"[CLEANUP] cuda cleanup: {type(exc).__name__}: {exc}")


def run_validation() -> int:
    code = 3
    try:
        code = int(validator.main())
    finally:
        _cleanup_runtime()
        sys.stdout.flush()
        sys.stderr.flush()
    return code


if __name__ == "__main__":
    exit_code = run_validation()
    print(f"[CLEANUP] validation process exiting via os._exit({exit_code})")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
