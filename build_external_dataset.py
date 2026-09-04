from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.external import KaistAdapter, MetroPTAdapter, OfficialDomainCalibrator


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"
PROCESSED = EXTERNAL / "processed"


def find_kaist_root() -> Path:
    base = EXTERNAL / "kaist"
    candidates = [p for p in base.rglob("*") if p.is_dir() and (p / "vibration").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"cannot locate KAIST dataset below {base}")
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def finite_summary(series: pd.Series) -> dict[str, float | int | None]:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "min": None, "median": None, "max": None, "std": None}
    return {
        "count": int(len(x)),
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
    }


def build_kaist(feature_hz: float, limit: int | None) -> list[dict]:
    root = find_kaist_root()
    adapter = KaistAdapter(root)
    conditions = adapter.list_conditions(require_acoustic=False)

    # Make the smoke run maximally informative: acoustic-complete runs first, then the rest.
    conditions = sorted(conditions, key=lambda c: (c.acoustic_path is None, c.stem))
    if limit is not None:
        conditions = conditions[:limit]

    output_dir = PROCESSED / "kaist_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    print(f"KAIST root       : {root}")
    print(f"KAIST conditions : {len(adapter.list_conditions())}")
    print(f"processing       : {len(conditions)}")
    print(f"feature_hz       : {feature_hz}")

    for i, condition in enumerate(conditions, 1):
        print(f"[{i:02d}/{len(conditions):02d}] {condition.stem}")
        df = adapter.load_condition(condition, feature_hz=feature_hz)
        out = output_dir / f"{condition.stem}.csv"
        df.to_csv(out, index=False)

        row = {
            "condition": condition.stem,
            "rows": int(len(df)),
            "duration_s": float(df["time_s"].iloc[-1]) if len(df) else 0.0,
            "has_acoustic": bool(df["acoustic_db"].notna().any()),
            "output": str(out.relative_to(ROOT)),
            "stats": {
                col: finite_summary(df[col])
                for col in ["vibration_rms", "temperature_c", "current_a", "acoustic_db"]
            },
        }
        manifest.append(row)

    return manifest


def build_metropt(nrows: int | None) -> dict:
    csv_path = EXTERNAL / "metropt" / "MetroPT3(AirCompressor).csv"
    adapter = MetroPTAdapter(csv_path)
    print(f"MetroPT CSV      : {csv_path}")
    print(f"MetroPT rows cap : {nrows if nrows is not None else 'FULL'}")
    df = adapter.load_core(nrows=nrows)

    columns = [
        "timestamp",
        "temperature_c",
        "current_a",
        "pressure_kpa",
        "TP2_kpa",
        "TP3_kpa",
        "H1_kpa",
        "DV_pressure_kpa",
        "Reservoirs_kpa",
        "COMP",
        "DV_eletric",
        "Towers",
        "MPG",
        "LPS",
        "Pressure_switch",
        "Oil_level",
        "Caudal_impulses",
        "source",
    ]
    out = PROCESSED / "metropt_core.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(out, index=False)

    segments = adapter.split_contiguous_segments(df, max_gap_seconds=30.0, min_rows=64)
    return {
        "rows": int(len(df)),
        "sampling_seconds_median": adapter.infer_sampling_seconds(df),
        "contiguous_segments": int(len(segments)),
        "output": str(out.relative_to(ROOT)),
        "stats": {
            col: finite_summary(df[col])
            for col in ["temperature_c", "current_a", "pressure_kpa"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-hz", type=float, default=10.0)
    parser.add_argument("--kaist-limit", type=int, default=None)
    parser.add_argument(
        "--metro-rows",
        type=int,
        default=None,
        help="limit MetroPT rows for smoke test; omit for full dataset",
    )
    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("BUILD EXTERNAL SOURCE-DOMAIN DATA")
    print("=" * 100)
    kaist_manifest = build_kaist(args.feature_hz, args.kaist_limit)

    print("\n" + "=" * 100)
    metro_manifest = build_metropt(args.metro_rows)

    print("\n" + "=" * 100)
    print("OFFICIAL SAMPLE CALIBRATION PRIORS")
    calibrator = OfficialDomainCalibrator(ROOT / "data" / "raw")
    calibration_path = PROCESSED / "official_domain_stats.json"
    calibration = calibrator.save_stats(calibration_path)
    print(f"sequences        : {calibration['num_sequences']}")
    print(f"saved            : {calibration_path}")

    manifest = {
        "kaist": kaist_manifest,
        "metropt": metro_manifest,
        "official_calibration": str(calibration_path.relative_to(ROOT)),
        "important_note": (
            "These are source-domain processed signals. They are NOT yet a fabricated synchronized "
            "six-sensor machine sequence and must not be concatenated across unrelated machines."
        ),
    }
    manifest_path = PROCESSED / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("BUILD COMPLETE")
    print("=" * 100)
    print(f"manifest         : {manifest_path}")
    print("NOTE: no online V8 model/API files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
