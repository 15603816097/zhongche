from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    "vibration_rms",
    "temperature_c",
    "current_a",
    "speed_rpm",
    "acoustic_db",
    "pressure_kpa",
]


class OfficialDomainCalibrator:
    """Estimate scale/dynamics priors from the five official example sequences.

    The example data are used only for calibration priors, never as proof that hidden-test data
    follow the same distribution. This class intentionally keeps the mapping conservative.
    """

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(self.raw_dir)

    def _sequence_frames(self) -> Iterable[tuple[str, pd.DataFrame]]:
        for seq_dir in sorted(self.raw_dir.glob("sequence*")):
            if not seq_dir.is_dir():
                continue
            frames = []
            for name in ("history.csv", "future.csv"):
                path = seq_dir / name
                if path.is_file():
                    df = pd.read_csv(path)
                    available = [c for c in TARGET_COLUMNS if c in df.columns]
                    if len(available) == len(TARGET_COLUMNS):
                        frames.append(df[TARGET_COLUMNS].copy())
            if frames:
                yield seq_dir.name, pd.concat(frames, ignore_index=True)

    def compute_stats(self) -> dict:
        all_frames = list(self._sequence_frames())
        if not all_frames:
            raise RuntimeError(f"no usable sequence data found under {self.raw_dir}")

        pooled = pd.concat([df for _, df in all_frames], ignore_index=True)
        result: dict[str, object] = {
            "note": "official example sequences used only as domain-calibration priors",
            "num_sequences": len(all_frames),
            "variables": {},
            "per_sequence": {},
        }

        variables = result["variables"]
        assert isinstance(variables, dict)
        for col in TARGET_COLUMNS:
            x = pd.to_numeric(pooled[col], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            dx = np.diff(x)
            dx = dx[np.isfinite(dx)]
            q05, q25, q50, q75, q95 = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
            variables[col] = {
                "q05": float(q05),
                "q25": float(q25),
                "median": float(q50),
                "q75": float(q75),
                "q95": float(q95),
                "iqr": float(max(q75 - q25, 1e-12)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "diff_std": float(np.std(dx)) if len(dx) else 0.0,
            }

        per_sequence = result["per_sequence"]
        assert isinstance(per_sequence, dict)
        for seq_name, df in all_frames:
            seq_stats: dict[str, dict[str, float]] = {}
            for col in TARGET_COLUMNS:
                x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    continue
                q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75])
                seq_stats[col] = {
                    "median": float(q50),
                    "iqr": float(max(q75 - q25, 1e-12)),
                    "std": float(np.std(x)),
                    "diff_std": float(np.std(np.diff(x))) if len(x) > 1 else 0.0,
                }
            per_sequence[seq_name] = seq_stats

        return result

    @staticmethod
    def robust_affine_map(series: np.ndarray, target_stats: dict) -> np.ndarray:
        """Match median/IQR only; keep source temporal shape intact."""
        x = np.asarray(series, dtype=float)
        finite = np.isfinite(x)
        if not finite.any():
            return x.copy()

        src = x[finite]
        q25, q50, q75 = np.quantile(src, [0.25, 0.50, 0.75])
        src_iqr = max(float(q75 - q25), 1e-12)
        target_iqr = max(float(target_stats["iqr"]), 1e-12)
        mapped = (x - float(q50)) / src_iqr
        mapped = mapped * target_iqr + float(target_stats["median"])
        return mapped

    def save_stats(self, output_path: str | Path) -> dict:
        stats = self.compute_stats()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        return stats
