from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.full_arch_runtime import (
    preload_full_arch_runtime,
    predict_future as predict_full_arch,
)

OUT_DIR = Path("artifacts/fa52_speed")
OUT_PATH = OUT_DIR / "baseline_predictions.npz"
EXPECTED_FLAT_RMSE = 7.801844
TOL = 5e-3


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    preds = []
    truth = []

    preload_full_arch_runtime()

    for seq_dir in sorted(Path(DATA_DIR).glob("sequence*")):
        hp = seq_dir / "history.csv"
        fp = seq_dir / "future.csv"
        if not hp.is_file() or not fp.is_file():
            continue

        hdf = pd.read_csv(hp)
        fdf = pd.read_csv(fp)
        pred = np.asarray(
            predict_full_arch(hdf[TARGET_COLUMNS], return_timings=False),
            dtype=np.float64,
        )
        y = fdf[TARGET_COLUMNS].iloc[:HORIZON].to_numpy(dtype=np.float64)

        if pred.shape != (HORIZON, len(TARGET_COLUMNS)):
            raise RuntimeError(f"{seq_dir.name}: bad pred shape={pred.shape}")
        if not np.isfinite(pred).all():
            raise RuntimeError(f"{seq_dir.name}: pred contains NaN/Inf")

        names.append(seq_dir.name)
        preds.append(pred)
        truth.append(y)
        print(f"{seq_dir.name}: captured", flush=True)

    if len(names) < 5:
        raise RuntimeError(f"expected >=5 sequences, got {len(names)}")

    preds = np.stack(preds)
    truth = np.stack(truth)
    rmse = float(np.sqrt(np.mean((truth - preds) ** 2)))

    print(f"FA52 frozen flat RMSE: {rmse:.6f}")
    print(f"expected              : {EXPECTED_FLAT_RMSE:.6f}")
    if abs(rmse - EXPECTED_FLAT_RMSE) > TOL:
        raise RuntimeError(
            f"baseline integrity failed: {rmse:.6f} vs {EXPECTED_FLAT_RMSE:.6f}"
        )

    np.savez_compressed(
        OUT_PATH,
        names=np.asarray(names),
        predictions=preds,
    )
    print(f"baseline artifact      : {OUT_PATH}")
    print("FA52 BASELINE CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
