from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class MetroPTAdapter:
    """Adapter for MetroPT-3 compressor data.

    The source dataset is sampled at roughly 10-second intervals. We keep that native timeline
    here and only convert physical units / select useful channels. Any 1-second interpolation or
    competition-domain calibration is intentionally handled later.
    """

    REQUIRED_COLUMNS = [
        "timestamp",
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Oil_temperature",
        "Motor_current",
        "COMP",
        "DV_eletric",
        "Towers",
        "MPG",
        "LPS",
        "Pressure_switch",
        "Oil_level",
        "Caudal_impulses",
    ]

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(self.csv_path)

    def load_core(self, nrows: int | None = None) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, usecols=self.REQUIRED_COLUMNS, nrows=nrows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # The MetroPT pressure channels are documented in bar. Convert to kPa.
        for col in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"]:
            df[f"{col}_kpa"] = pd.to_numeric(df[col], errors="coerce") * 100.0

        df["temperature_c"] = pd.to_numeric(df["Oil_temperature"], errors="coerce")
        df["current_a"] = pd.to_numeric(df["Motor_current"], errors="coerce")

        # Reservoirs / TP3 are nearly identical in the inspected sample. Reservoirs is chosen
        # as the main pressure state because it directly represents stored system pressure.
        df["pressure_kpa"] = df["Reservoirs_kpa"]
        df["source"] = "metropt3"
        return df

    @staticmethod
    def infer_sampling_seconds(df: pd.DataFrame) -> float:
        if len(df) < 2:
            return float("nan")
        delta = df["timestamp"].diff().dt.total_seconds().dropna()
        delta = delta[np.isfinite(delta) & (delta > 0)]
        if delta.empty:
            return float("nan")
        return float(delta.median())

    @staticmethod
    def split_contiguous_segments(
        df: pd.DataFrame,
        *,
        max_gap_seconds: float = 30.0,
        min_rows: int = 64,
    ) -> list[pd.DataFrame]:
        if df.empty:
            return []
        gap = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
        group_id = (gap > float(max_gap_seconds)).cumsum()
        out: list[pd.DataFrame] = []
        for _, part in df.groupby(group_id):
            part = part.reset_index(drop=True)
            if len(part) >= int(min_rows):
                out.append(part)
        return out
