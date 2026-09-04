from __future__ import annotations

import os

import app as base
from src.inference_v81 import (
    V81_PATCHTST_DEVICE,
    V81_TEMPERATURE_ENABLED,
    V81_TEMPERATURE_WEIGHT,
    preload_v81_models,
    predict_future as predict_future_v81,
)


# -----------------------------------------------------------------------------
# V8.1 candidate API wrapper
# -----------------------------------------------------------------------------
# Keep the already accepted app.py/callback implementation completely untouched.
# We only replace two module globals BEFORE FastAPI startup executes:
#   1) model preload -> V8 + fully warmed PatchTST temperature branch
#   2) predictor     -> V8.1 wrapper
# All request parsing, async scheduling, callback serialization/retry/ordering and
# HTTP response code paths continue to be the exact app.py implementation.
#
# Rollback is immediate:
#   - run the original `app:app`, OR
#   - start this wrapper with V81_TEMPERATURE_ENABLED=0 (then prediction is exact V8).
# -----------------------------------------------------------------------------

APP_VERSION = "2.10.0-v81-candidate"


def _load_models_v81_adapter():
    """Match app.py's historical five-value load_models() return contract."""
    ensemble_config = preload_v81_models()
    return None, None, None, None, ensemble_config


# app.py functions resolve these names through the app module globals at call time.
# Patching here therefore keeps the verified API/callback code byte-for-byte intact.
base.load_models = _load_models_v81_adapter
base.predict_future = predict_future_v81
base.APP_VERSION = APP_VERSION
base.app.version = APP_VERSION

app = base.app


@app.get("/candidate")
def candidate_status():
    return {
        "candidate": "V8.1 temperature-only PatchTST blend",
        "version": APP_VERSION,
        "v81_temperature_enabled": bool(V81_TEMPERATURE_ENABLED),
        "v81_temperature_weight": float(V81_TEMPERATURE_WEIGHT),
        "v81_patchtst_device": str(V81_PATCHTST_DEVICE),
        "formula": "temperature_c = 0.85*V8 + 0.15*PatchTST (default)",
        "non_temperature_targets": "exact V8",
        "fallback": "PatchTST failure -> exact V8 unless strict mode is enabled",
        "callback": "unchanged app.py verified schema",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app_v81:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8800")),
        workers=1,
        log_level="info",
    )
