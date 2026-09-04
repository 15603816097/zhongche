from __future__ import annotations

import gc
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
    """Tear down validation-only thread/CUDA resources before interpreter exit.

    The candidate validation exercises LightGBM thread pools, XGBoost CUDA and
    PatchTST CUDA in one short-lived Python process. The actual FastAPI service is
    long-lived, but during this synthetic validation the interpreter previously
    received SIGABRT after all gates had already passed while native runtimes were
    being finalized. Explicitly quiesce executors and CUDA first so the shell sees
    the real validation exit code rather than a teardown-time abort.
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

    # Release Python references while CUDA/runtime libraries are still fully alive.
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


def main() -> int:
    code = 3
    try:
        code = int(validator.main())
        return code
    finally:
        _cleanup_runtime()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
