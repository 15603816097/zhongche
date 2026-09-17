from __future__ import annotations

import app as base_app
from src.v9_safe_runtime import (
    PRESSURE_ALPHA,
    TEMP_ALPHA,
    predict_future as predict_future_v9_safe,
    preload_v9_safe_runtime,
)


# Reuse the already field-tested V8 API/callback implementation verbatim.
# Only replace the prediction function inside this isolated process.
base_app.APP_NAME = "Rail Transit Time-Series Forecast API - V9 Safe"
base_app.APP_VERSION = "2.9.0-v9-safe"
base_app.predict_future = predict_future_v9_safe

app = base_app.app


@app.on_event("startup")
def preload_v9_safe_bank() -> None:
    preload_v9_safe_runtime()
    print(
        "[V9-SAFE READY] "
        f"temperature_alpha={TEMP_ALPHA:.3f} "
        f"pressure_alpha={PRESSURE_ALPHA:.3f} "
        "untouched=vibration_rms,current_a,speed_rpm,acoustic_db"
    )
