from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, TARGET_COLUMNS
from src.full_arch_runtime import predict_future as predict_full
from src.inference import predict_future as predict_v8

ROOT = Path(__file__).resolve().parent
STAGE5 = ROOT / "artifacts" / "full_arch" / "dynamic_gate" / "stage5_predictions.npz"
CONFIG = ROOT / "models" / "full_arch" / "dynamic_gate_config.json"

UNTOUCHED = [
    TARGET_COLUMNS.index("vibration_rms"),
    TARGET_COLUMNS.index("temperature_c"),
    TARGET_COLUMNS.index("current_a"),
]


def main() -> int:
    if not STAGE5.is_file():
        raise FileNotFoundError(STAGE5)
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    with np.load(STAGE5) as z:
        expected = np.asarray(z["final_known_prediction"], dtype=np.float64)
        truth = np.asarray(z["truth"], dtype=np.float64)

    preds = []
    v8s = []
    times = []

    print("=" * 112)
    print("FULL ARCHITECTURE - STAGE 6 DIRECT RUNTIME REPRODUCTION")
    print("=" * 112)
    print("enabled targets:", cfg.get("enabled_targets"))

    for i in range(5):
        name = f"sequence{i+1:04d}"
        hdf = pd.read_csv(DATA_DIR / name / "history.csv")

        t0 = time.perf_counter()
        p = np.asarray(predict_full(hdf), dtype=np.float64)
        elapsed = time.perf_counter() - t0

        p8 = np.asarray(predict_v8(hdf), dtype=np.float64)
        ref = expected[i]

        repro = float(np.max(np.abs(p - ref)))
        untouched = float(np.max(np.abs(p[:, UNTOUCHED] - p8[:, UNTOUCHED])))
        print(
            f"{name}: time={elapsed:.3f}s "
            f"stage5_max_diff={repro:.3e} "
            f"untouched_vs_v8={untouched:.3e}"
        )
        if repro > 2e-5:
            raise RuntimeError(f"{name}: Stage5 reproduction mismatch {repro}")
        if untouched > 2e-5:
            raise RuntimeError(f"{name}: untouched target mismatch {untouched}")

        preds.append(p)
        v8s.append(p8)
        times.append(elapsed)

    pred = np.stack(preds)
    v8 = np.stack(v8s)
    rmse8 = float(np.sqrt(np.mean((truth - v8) ** 2)))
    rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
    print("\nflat RMSE V8       :", f"{rmse8:.6f}")
    print("flat RMSE runtime   :", f"{rmse:.6f}")
    print("flat RMSE ratio     :", f"{rmse / max(rmse8, 1e-12):.6f}")
    print("mean runtime        :", f"{np.mean(times):.3f}s")

    if rmse >= rmse8:
        raise RuntimeError("runtime candidate does not improve flat RMSE")
    print("STAGE 6 DIRECT RUNTIME: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
