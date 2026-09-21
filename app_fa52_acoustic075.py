from __future__ import annotations

import os

# Must be set before importing app_full_arch / src.full_arch_runtime.
os.environ.setdefault(
    "FULL_ARCH_GATE_CONFIG",
    "full_arch_gate_acoustic075.json",
)

import app_full_arch as full_app  # noqa: E402
import app as base_app  # noqa: E402
from src.full_arch_runtime import GATE_CONFIG_PATH  # noqa: E402

CANDIDATE_VERSION = "3.0.1-fa52-acoustic075"

base_app.APP_NAME = "Rail Transit Time-Series Forecast API - FA52 Acoustic 0.75"
base_app.APP_VERSION = CANDIDATE_VERSION
base_app.app.version = CANDIDATE_VERSION

app = full_app.app


@app.get("/candidate")
def candidate_info():
    return {
        "version": CANDIDATE_VERSION,
        "baseline": "3.0.0-full-arch",
        "gate_config": str(GATE_CONFIG_PATH),
        "shrink_scales": {
            "speed_rpm": 1.0,
            "acoustic_db": 0.75,
            "pressure_kpa": 1.0,
        },
        "production_8800_untouched": True,
    }
