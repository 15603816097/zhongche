from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR
from src.full_arch_runtime import predict_future

ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "validation" / "sequence0001_expected.npy"


def main() -> int:
    if not EXPECTED.is_file():
        raise FileNotFoundError(EXPECTED)

    history = pd.read_csv(DATA_DIR / "sequence0001" / "history.csv")
    expected = np.load(EXPECTED)
    pred = np.asarray(predict_future(history), dtype=np.float64)

    if pred.shape != expected.shape:
        raise RuntimeError(f"shape mismatch: pred={pred.shape}, expected={expected.shape}")

    diff = float(np.max(np.abs(pred - expected)))
    print("=" * 88)
    print("FULL ARCHITECTURE V1 - INSTALL VERIFICATION")
    print("=" * 88)
    print("prediction shape :", pred.shape)
    print("max abs diff     :", f"{diff:.3e}")
    print("enabled targets  :", ["speed_rpm", "acoustic_db", "pressure_kpa"])

    if diff > 2e-5:
        raise RuntimeError(f"runtime reproduction mismatch: {diff}")

    print("VERIFY INSTALL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
