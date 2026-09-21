from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, TARGET_COLUMNS
from src.full_arch_runtime import predict_future as predict_full_arch
from src.inference import predict_future as predict_v8

OUT_DIR = Path("artifacts/fa52_acoustic075")
OUT_PATH = OUT_DIR / "expected_predictions.npz"

EXPECTED_V8_RMSE = 8.072473
EXPECTED_FA_RMSE = 7.801844
EXPECTED_CAND_RMSE = 7.802005
TOL = 5e-3

SCALES = {
    "speed_rpm": 1.0,
    "acoustic_db": 0.75,
    "pressure_kpa": 1.0,
}


def flat_rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    names = []
    v8_rows = []
    fa_rows = []
    cand_rows = []
    truth_rows = []

    for seq_dir in sorted(Path(DATA_DIR).glob("sequence*")):
        hp = seq_dir / "history.csv"
        fp = seq_dir / "future.csv"
        if not hp.is_file() or not fp.is_file():
            continue

        hdf = pd.read_csv(hp)[TARGET_COLUMNS]
        fdf = pd.read_csv(fp)[TARGET_COLUMNS].iloc[:HORIZON]

        print(f"capture {seq_dir.name}", flush=True)
        pv8 = np.asarray(
            predict_v8(hdf, return_timings=False),
            dtype=np.float64,
        )
        pfa = np.asarray(
            predict_full_arch(hdf, return_timings=False),
            dtype=np.float64,
        )
        y = fdf.to_numpy(dtype=np.float64)

        pcand = pv8.copy()
        for target, scale in SCALES.items():
            j = TARGET_COLUMNS.index(target)
            pcand[:, j] = (
                pv8[:, j]
                + float(scale) * (pfa[:, j] - pv8[:, j])
            )

        names.append(seq_dir.name)
        v8_rows.append(pv8)
        fa_rows.append(pfa)
        cand_rows.append(pcand)
        truth_rows.append(y)

    if len(names) < 5:
        raise RuntimeError(f"expected >=5 sequences, got {len(names)}")

    v8 = np.stack(v8_rows)
    fa = np.stack(fa_rows)
    cand = np.stack(cand_rows)
    truth = np.stack(truth_rows)

    v8_rmse = flat_rmse(truth, v8)
    fa_rmse = flat_rmse(truth, fa)
    cand_rmse = flat_rmse(truth, cand)

    print(f"V8 RMSE       : {v8_rmse:.6f}")
    print(f"FA52 RMSE      : {fa_rmse:.6f}")
    print(f"candidate RMSE : {cand_rmse:.6f}")

    if abs(v8_rmse - EXPECTED_V8_RMSE) > TOL:
        raise RuntimeError("V8 integrity failed")
    if abs(fa_rmse - EXPECTED_FA_RMSE) > TOL:
        raise RuntimeError("FA52 integrity failed")
    if abs(cand_rmse - EXPECTED_CAND_RMSE) > TOL:
        raise RuntimeError("candidate integrity failed")

    np.savez_compressed(
        OUT_PATH,
        names=np.asarray(names),
        expected=cand,
        v8=v8,
        full_arch=fa,
    )
    print(f"expected artifact: {OUT_PATH}")
    print("FA52 ACOUSTIC075 EXPECTED CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
