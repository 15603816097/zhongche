from __future__ import annotations

from typing import Any, Dict

from fastapi import Request

import app as base_app
from src.full_arch_runtime import (
    preload_full_arch_runtime,
    predict_future as predict_future_full_arch,
)

base_app.APP_NAME = "Rail Transit Time-Series Forecast API - Full Architecture"
base_app.APP_VERSION = "3.0.1-full-arch"
base_app.predict_future = predict_future_full_arch

app = base_app.app


@app.on_event("startup")
def preload_full_arch() -> None:
    preload_full_arch_runtime()
    print(
        "[FULL-ARCH READY] "
        "base=V8 "
        "gate=target_x_horizon_confidence_ood "
        "deep_direct_weight=0 "
        "post_root_alias=true"
    )


@app.post("/")
def predict_root_alias(payload: Dict[str, Any], request: Request):
    """
    Compatibility alias for evaluators that POST to the submitted base URL.

    The official business logic remains base_app.predict(), so POST / and
    POST /predict share exactly the same validation, async callback behavior,
    prediction path, and response format.
    """
    return base_app.predict(payload, request)
