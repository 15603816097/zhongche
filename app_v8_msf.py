from __future__ import annotations

# Import the already validated official API implementation, then replace only
# the prediction function. Request/callback protocol remains byte-for-byte the
# same as the V8 service.
import app as base_app

from src.inference import predict_future as predict_v8
from src.v8_msf_runtime import apply_multiscale_fusion, load_msf_config


MSF_VERSION = "2.9.0-v8-msf"


def predict_future_msf(history_df, return_timings: bool = False):
    pred_v8, timings = predict_v8(history_df, return_timings=True)
    pred, msf_seconds, msf_targets = apply_multiscale_fusion(pred_v8)

    timings = dict(timings)
    timings["msf"] = float(msf_seconds)
    timings["msf_targets"] = int(msf_targets)
    timings["total"] = float(timings.get("total", 0.0)) + float(msf_seconds)

    if return_timings:
        return pred, timings
    return pred


# Fail fast on startup/import if the candidate config is missing or rejected.
MSF_CONFIG = load_msf_config()

# app.py's run_one resolves predict_future from its module global namespace.
base_app.predict_future = predict_future_msf
base_app.APP_VERSION = MSF_VERSION
base_app.app.version = MSF_VERSION

# Add MSF metadata to health without touching the validated request protocol.
_original_health = base_app.health


def _health_msf():
    body = dict(_original_health())
    body["version"] = MSF_VERSION
    body["multiscale_fusion"] = True
    body["multiscale_enabled_targets"] = list(
        MSF_CONFIG.get("enabled_targets") or []
    )
    body["multiscale_windows"] = list(MSF_CONFIG.get("windows") or [])
    return body


# Replace the existing /health endpoint function object in FastAPI routes.
for route in base_app.app.routes:
    if getattr(route, "path", None) == "/health":
        route.endpoint = _health_msf
        break

app = base_app.app
