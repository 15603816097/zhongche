from __future__ import annotations

import os

import app as base
from src.inference_v83 import preload_v83_models, predict_future as predict_future_v83


APP_VERSION = "2.11.0-v83-final"


def _load_models_v83_adapter():
    """Preserve app.py's historical five-value startup contract."""
    ensemble_config = preload_v83_models()
    return None, None, None, None, ensemble_config


# Keep the already accepted app.py request/callback implementation byte-for-byte intact.
# Only the startup preload function and predictor global are swapped before startup.
base.load_models = _load_models_v83_adapter
base.predict_future = predict_future_v83
base.APP_VERSION = APP_VERSION
base.app.version = APP_VERSION

app = base.app


@app.get("/candidate")
def candidate_status():
    from src.inference_v83 import _load_candidate

    cfg = _load_candidate()
    return {
        "candidate": "V8.3 final-sprint conservative trend postprocessor",
        "version": APP_VERSION,
        "base_version": int(cfg.get("base_version", 0)),
        "enabled_targets": list(cfg.get("enabled_targets", [])),
        "offline_gate_pass": bool(cfg.get("offline_gate_pass", False)),
        "offline_rmse_ratio": float(cfg.get("global_rmse_ratio", float("nan"))),
        "offline_proxy_gain_pct": float(cfg.get("global_proxy_gain_pct", float("nan"))),
        "acoustic_db": "exact V8 unless enabled by candidate config",
        "callback": "unchanged app.py verified schema",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app_v83:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8800")),
        workers=1,
        log_level="info",
    )
