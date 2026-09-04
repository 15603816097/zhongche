from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
from nptdms import TdmsFile

from .signal_processing import (
    block_mean,
    block_rms,
    block_size_for_feature_hz,
    pressure_pa_to_db_spl,
)


@dataclass(frozen=True)
class KaistCondition:
    stem: str
    vibration_path: Path
    tdms_path: Path
    acoustic_path: Optional[Path]


class KaistAdapter:
    """Read KAIST rotating-machine files and convert raw signals to low-rate physical features.

    This adapter deliberately does NOT force the KAIST data into the competition's six-column
    target space. It preserves real source-domain values first; domain calibration is a separate step.

    Some KAIST TDMS files contain channel definitions with zero samples. Those channels are metadata
    artefacts, not usable sensor streams, so they are ignored rather than truncating every modality
    to length zero.
    """

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.vibration_dir = self.root / "vibration"
        self.acoustic_dir = self.root / "acoustic"
        self.tdms_dir = self.root / "current_temp"
        for p in (self.vibration_dir, self.tdms_dir):
            if not p.is_dir():
                raise FileNotFoundError(p)

    @staticmethod
    def canonical_stem(stem: str) -> str:
        # The downloaded vibration folder contains several "Unbalalnce" typos while
        # current_temp uses "Unbalance". Canonicalising them recovers the true matches.
        return stem.replace("Unbalalnce", "Unbalance")

    @staticmethod
    def _index(directory: Path, suffix: str) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        for path in sorted(directory.glob(f"*{suffix}")):
            out[KaistAdapter.canonical_stem(path.stem)] = path
        return out

    def list_conditions(self, require_acoustic: bool = False) -> list[KaistCondition]:
        vib = self._index(self.vibration_dir, ".mat")
        tdms = self._index(self.tdms_dir, ".tdms")
        ac = self._index(self.acoustic_dir, ".mat") if self.acoustic_dir.is_dir() else {}

        stems = sorted(set(vib) & set(tdms))
        if require_acoustic:
            stems = [s for s in stems if s in ac]

        return [
            KaistCondition(
                stem=s,
                vibration_path=vib[s],
                tdms_path=tdms[s],
                acoustic_path=ac.get(s),
            )
            for s in stems
        ]

    @staticmethod
    def _load_mat_signal(path: Path) -> tuple[np.ndarray, float, str]:
        obj = loadmat(path, squeeze_me=True, struct_as_record=False)
        signal = obj.get("Signal")
        if signal is None:
            raise KeyError(f"Signal not found in {path}")
        values = np.asarray(signal.y_values.values, dtype=np.float64)
        dt = float(signal.x_values.increment)
        unit = str(signal.y_values.quantity.label)
        if values.size == 0:
            raise ValueError(f"empty MAT signal in {path}")
        return values, dt, unit

    @staticmethod
    def _is_usable_channel(arr: np.ndarray) -> bool:
        arr = np.asarray(arr)
        return bool(arr.size > 0 and np.isfinite(arr).any())

    @staticmethod
    def _load_tdms_channels(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        tdms = TdmsFile.read(path)
        temperature: list[np.ndarray] = []
        current: list[np.ndarray] = []
        increments: list[float] = []

        for group in tdms.groups():
            for channel in group.channels():
                props = dict(channel.properties)
                unit = str(props.get("unit_string", "")).strip()
                if unit not in {"°C", "A"}:
                    continue

                arr = np.asarray(channel[:], dtype=np.float64)
                # Several files expose zero-length TDMS channels. Keeping them would make
                # min(channel_lengths)==0 and incorrectly discard an otherwise valid run.
                if not KaistAdapter._is_usable_channel(arr):
                    continue

                if unit == "°C":
                    temperature.append(arr)
                elif unit == "A":
                    current.append(arr)

                inc = props.get("wf_increment")
                if inc is not None:
                    try:
                        inc = float(inc)
                    except (TypeError, ValueError):
                        inc = np.nan
                    if np.isfinite(inc) and inc > 0:
                        increments.append(inc)

        if not temperature:
            raise ValueError(f"no usable °C channels found in {path}")
        if not current:
            raise ValueError(f"no usable A channels found in {path}")
        if not increments:
            raise ValueError(f"no valid wf_increment found in {path}")

        return temperature, current, float(np.median(increments))

    @staticmethod
    def _stack_same_length(channels: list[np.ndarray]) -> np.ndarray:
        usable = [
            np.asarray(x, dtype=np.float64).reshape(-1)
            for x in channels
            if KaistAdapter._is_usable_channel(x)
        ]
        if not usable:
            raise ValueError("no usable non-empty channels")

        n = min(len(x) for x in usable)
        if n <= 0:
            raise ValueError("all usable channels are empty")
        return np.column_stack([x[:n] for x in usable])

    @staticmethod
    def parse_condition(stem: str) -> tuple[float | None, str]:
        if "Nm_" in stem:
            left, fault = stem.split("Nm_", 1)
            try:
                return float(left), fault
            except ValueError:
                pass
        return None, stem

    def load_condition(self, condition: KaistCondition | str, feature_hz: float = 10.0) -> pd.DataFrame:
        if isinstance(condition, str):
            matches = {c.stem: c for c in self.list_conditions()}
            key = self.canonical_stem(condition)
            if key not in matches:
                raise KeyError(f"unknown KAIST condition: {condition}")
            condition = matches[key]

        vib_raw, vib_dt, vib_unit = self._load_mat_signal(condition.vibration_path)
        if vib_unit.lower() != "g":
            raise ValueError(f"expected vibration unit g, got {vib_unit!r}")
        vib_block = block_size_for_feature_hz(vib_dt, feature_hz)
        vibration_rms = block_rms(vib_raw, vib_block)

        temp_channels, current_channels, tdms_dt = self._load_tdms_channels(condition.tdms_path)
        temp_raw = self._stack_same_length(temp_channels)
        current_raw = self._stack_same_length(current_channels)
        tdms_block = block_size_for_feature_hz(tdms_dt, feature_hz)
        temperature_c = block_mean(temp_raw, tdms_block)
        current_a = block_rms(current_raw, tdms_block)

        acoustic_db: Optional[np.ndarray] = None
        if condition.acoustic_path is not None:
            ac_raw, ac_dt, ac_unit = self._load_mat_signal(condition.acoustic_path)
            if ac_unit.lower() != "pa":
                raise ValueError(f"expected acoustic unit Pa, got {ac_unit!r}")
            ac_block = block_size_for_feature_hz(ac_dt, feature_hz)
            ac_rms = block_rms(ac_raw, ac_block)
            acoustic_db = pressure_pa_to_db_spl(ac_rms)

        lengths = [len(vibration_rms), len(temperature_c), len(current_a)]
        if acoustic_db is not None:
            lengths.append(len(acoustic_db))
        n = min(lengths)
        if n <= 0:
            raise ValueError(f"no aggregated samples for {condition.stem}")

        load_nm, fault = self.parse_condition(condition.stem)
        df = pd.DataFrame(
            {
                "time_s": np.arange(n, dtype=np.float64) / float(feature_hz),
                "vibration_rms": vibration_rms[:n],
                "temperature_c": temperature_c[:n],
                "current_a": current_a[:n],
            }
        )
        if acoustic_db is not None:
            df["acoustic_db"] = acoustic_db[:n]
        else:
            df["acoustic_db"] = np.nan

        df["source"] = "kaist"
        df["condition"] = condition.stem
        df["fault"] = fault
        df["load_nm"] = np.nan if load_nm is None else load_nm
        df["feature_hz"] = float(feature_hz)
        df["temperature_channel_count"] = int(len(temp_channels))
        df["current_channel_count"] = int(len(current_channels))
        return df
