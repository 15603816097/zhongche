from __future__ import annotations

import json
from dataclasses import dataclass
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
LOOKBACK = 512
HORIZON = 96
TOTAL = LOOKBACK + HORIZON


@dataclass(frozen=True)
class SourceGroup:
    source: str
    group_id: str
    values: np.ndarray


class DeepPretrainCorpusBuilder:
    """Build a leakage-safe masked 512->96 corpus from heterogeneous real datasets.

    Design principles:
    - Never fabricate cross-machine sensor synchrony.
    - Keep each real run/contiguous segment as one independent group.
    - Split by group before window sampling.
    - Normalize every channel using HISTORY ONLY, then transform future with the same history stats.
    - Missing modalities are zero-filled and accompanied by a modality mask.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        seed: int = 42,
        kaist_windows_per_group: int = 8,
        metro_windows_per_group: int = 12,
        min_group_rows: int = 128,
        clip_z: float = 12.0,
    ):
        self.root = Path(root)
        self.processed = self.root / "external_data" / "processed"
        self.raw = self.root / "data" / "raw"
        self.seed = int(seed)
        self.kaist_windows_per_group = int(kaist_windows_per_group)
        self.metro_windows_per_group = int(metro_windows_per_group)
        self.min_group_rows = int(min_group_rows)
        self.clip_z = float(clip_z)

    @staticmethod
    def _interp_finite(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        idx = np.arange(len(x), dtype=np.float64)
        finite = np.isfinite(x)
        if finite.sum() == 0:
            return np.full_like(x, np.nan)
        if finite.sum() == 1:
            return np.full_like(x, float(x[finite][0]))
        out = x.copy()
        out[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
        return out

    @classmethod
    def _frame_to_matrix(cls, df: pd.DataFrame) -> np.ndarray:
        out = np.full((len(df), len(TARGET_COLUMNS)), np.nan, dtype=np.float64)
        for j, col in enumerate(TARGET_COLUMNS):
            if col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(x)
            if finite.sum() < max(8, int(0.50 * len(x))):
                continue
            out[:, j] = cls._interp_finite(x)
        return out

    def _load_kaist_groups(self) -> list[SourceGroup]:
        run_dir = self.processed / "kaist_runs"
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        groups: list[SourceGroup] = []
        for path in sorted(run_dir.glob("*.csv")):
            df = pd.read_csv(path)
            if len(df) < self.min_group_rows:
                continue
            values = self._frame_to_matrix(df)
            if not np.isfinite(values).any():
                continue
            groups.append(SourceGroup("kaist", f"kaist:{path.stem}", values))
        return groups

    def _load_metro_groups(self) -> list[SourceGroup]:
        path = self.processed / "metropt_core.csv"
        if not path.is_file():
            raise FileNotFoundError(path)

        usecols = ["timestamp", "temperature_c", "current_a", "pressure_kpa"]
        df = pd.read_csv(path, usecols=usecols)
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        dt = ts.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        cut = np.zeros(len(df), dtype=bool)
        if len(cut):
            cut[0] = True
        if len(dt) > 1:
            cut[1:] = (~np.isfinite(dt[1:])) | (dt[1:] <= 0.0) | (dt[1:] > 30.0)
        starts = np.flatnonzero(cut)
        ends = np.r_[starts[1:], len(df)]

        groups: list[SourceGroup] = []
        for idx, (s, e) in enumerate(zip(starts, ends)):
            if e - s < self.min_group_rows:
                continue
            part = df.iloc[s:e]
            values = self._frame_to_matrix(part)
            if not np.isfinite(values).any():
                continue
            groups.append(SourceGroup("metropt", f"metropt:{idx:04d}", values))
        return groups

    def load_groups(self) -> list[SourceGroup]:
        return self._load_kaist_groups() + self._load_metro_groups()

    def _split_map(self, groups: Iterable[SourceGroup]) -> dict[str, str]:
        by_source: dict[str, list[str]] = {}
        for g in groups:
            by_source.setdefault(g.source, []).append(g.group_id)

        rng = np.random.default_rng(self.seed)
        mapping: dict[str, str] = {}
        for source, ids in sorted(by_source.items()):
            ids = sorted(set(ids))
            order = np.asarray(ids, dtype=object)[rng.permutation(len(ids))].tolist()
            n = len(order)
            if n < 3:
                for gid in order:
                    mapping[gid] = "train"
                continue
            n_val = max(1, int(round(0.10 * n)))
            n_test = max(1, int(round(0.10 * n)))
            if n_val + n_test >= n:
                n_val = 1
                n_test = 1
            n_train = n - n_val - n_test
            for gid in order[:n_train]:
                mapping[gid] = "train"
            for gid in order[n_train:n_train + n_val]:
                mapping[gid] = "val"
            for gid in order[n_train + n_val:]:
                mapping[gid] = "test"
        return mapping

    @staticmethod
    def _resample_rows(values: np.ndarray, target_len: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        old_x = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
        new_x = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
        out = np.full((target_len, values.shape[1]), np.nan, dtype=np.float64)
        for j in range(values.shape[1]):
            x = values[:, j]
            finite = np.isfinite(x)
            if finite.sum() == 0:
                continue
            if finite.sum() == 1:
                out[:, j] = float(x[finite][0])
            else:
                out[:, j] = np.interp(new_x, old_x[finite], x[finite])
        return out

    @staticmethod
    def _even_starts(n: int, total: int, max_windows: int) -> list[int]:
        if n < total:
            return []
        max_start = n - total
        if max_start == 0:
            return [0]
        count = max(1, min(int(max_windows), max_start + 1))
        starts = np.linspace(0, max_start, count, dtype=int)
        return sorted(set(int(x) for x in starts))

    def _iter_group_samples(self, group: SourceGroup) -> Iterable[tuple[np.ndarray, int]]:
        values = group.values
        n = len(values)
        max_windows = (
            self.kaist_windows_per_group if group.source == "kaist" else self.metro_windows_per_group
        )
        if n < self.min_group_rows:
            return
        if n < TOTAL:
            yield self._resample_rows(values, TOTAL), -1
            return
        for start in self._even_starts(n, TOTAL, max_windows):
            yield values[start:start + TOTAL], start

    def _normalize_sample(
        self, sample: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if sample.shape != (TOTAL, len(TARGET_COLUMNS)):
            raise ValueError(f"unexpected sample shape {sample.shape}")

        z = np.zeros_like(sample, dtype=np.float32)
        mask = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
        center = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
        scale = np.ones(len(TARGET_COLUMNS), dtype=np.float32)

        for j in range(len(TARGET_COLUMNS)):
            x = sample[:, j]
            h = x[:LOOKBACK]
            finite_h = np.isfinite(h)
            finite_all = np.isfinite(x)
            if finite_h.sum() < max(8, int(0.80 * LOOKBACK)):
                continue
            if finite_all.sum() < max(8, int(0.80 * TOTAL)):
                continue

            h_valid = h[finite_h]
            q25, med, q75 = np.quantile(h_valid, [0.25, 0.50, 0.75])
            s = float(q75 - q25)
            if not np.isfinite(s) or s < 1e-8:
                s = float(np.std(h_valid))
            if not np.isfinite(s) or s < 1e-8:
                s = max(abs(float(med)) * 1e-3, 1.0)

            filled = self._interp_finite(x)
            channel_z = (filled - float(med)) / s
            channel_z = np.clip(channel_z, -self.clip_z, self.clip_z)

            z[:, j] = channel_z.astype(np.float32)
            mask[j] = 1.0
            center[j] = float(med)
            scale[j] = float(s)

        return z[:LOOKBACK], z[LOOKBACK:], mask, center, scale

    def build_pretrain(self) -> dict[str, np.ndarray]:
        groups = self.load_groups()
        split_map = self._split_map(groups)

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        sources: list[str] = []
        group_ids: list[str] = []
        splits: list[str] = []
        starts: list[int] = []

        for group in groups:
            for sample, start in self._iter_group_samples(group):
                x, y, mask, center, scale = self._normalize_sample(sample)
                if mask.sum() <= 0:
                    continue
                xs.append(x)
                ys.append(y)
                masks.append(mask)
                centers.append(center)
                scales.append(scale)
                sources.append(group.source)
                group_ids.append(group.group_id)
                splits.append(split_map[group.group_id])
                starts.append(int(start))

        if not xs:
            raise RuntimeError("no pretraining samples were generated")

        return {
            "X": np.stack(xs).astype(np.float32),
            "Y": np.stack(ys).astype(np.float32),
            "mask": np.stack(masks).astype(np.float32),
            "center": np.stack(centers).astype(np.float32),
            "scale": np.stack(scales).astype(np.float32),
            "source": np.asarray(sources, dtype="U16"),
            "group_id": np.asarray(group_ids, dtype="U64"),
            "split": np.asarray(splits, dtype="U8"),
            "start": np.asarray(starts, dtype=np.int32),
            "targets": np.asarray(TARGET_COLUMNS, dtype="U32"),
        }

    def build_official_finetune(self) -> dict[str, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        group_ids: list[str] = []

        for seq_dir in sorted(self.raw.glob("sequence*")):
            history_path = seq_dir / "history.csv"
            future_path = seq_dir / "future.csv"
            if not history_path.is_file() or not future_path.is_file():
                continue
            h = pd.read_csv(history_path)
            f = pd.read_csv(future_path)
            if len(h) < LOOKBACK or len(f) < HORIZON:
                continue
            missing = [c for c in TARGET_COLUMNS if c not in h.columns or c not in f.columns]
            if missing:
                continue
            full = pd.concat(
                [h[TARGET_COLUMNS].tail(LOOKBACK), f[TARGET_COLUMNS].head(HORIZON)],
                ignore_index=True,
            )
            sample = self._frame_to_matrix(full)
            x, y, mask, center, scale = self._normalize_sample(sample)
            if mask.sum() != len(TARGET_COLUMNS):
                continue
            xs.append(x)
            ys.append(y)
            masks.append(mask)
            centers.append(center)
            scales.append(scale)
            group_ids.append(seq_dir.name)

        if not xs:
            raise RuntimeError("no official finetune samples generated")

        return {
            "X": np.stack(xs).astype(np.float32),
            "Y": np.stack(ys).astype(np.float32),
            "mask": np.stack(masks).astype(np.float32),
            "center": np.stack(centers).astype(np.float32),
            "scale": np.stack(scales).astype(np.float32),
            "group_id": np.asarray(group_ids, dtype="U64"),
            "targets": np.asarray(TARGET_COLUMNS, dtype="U32"),
        }

    @staticmethod
    def summarize(corpus: dict[str, np.ndarray]) -> dict:
        split = corpus["split"]
        source = corpus["source"]
        mask = corpus["mask"]
        group_id = corpus["group_id"]

        summary: dict[str, object] = {
            "samples": int(len(split)),
            "groups": int(len(np.unique(group_id))),
            "split_samples": {},
            "split_groups": {},
            "source_samples": {},
            "modality_samples": {},
        }
        for s in ("train", "val", "test"):
            idx = split == s
            summary["split_samples"][s] = int(idx.sum())
            summary["split_groups"][s] = int(len(np.unique(group_id[idx])))
        for src in np.unique(source):
            summary["source_samples"][str(src)] = int((source == src).sum())
        for j, col in enumerate(TARGET_COLUMNS):
            summary["modality_samples"][col] = int((mask[:, j] > 0.5).sum())
        return summary

    def save(
        self,
        pretrain_path: str | Path,
        official_path: str | Path,
        manifest_path: str | Path,
    ) -> dict:
        pretrain = self.build_pretrain()
        official = self.build_official_finetune()

        pretrain_path = Path(pretrain_path)
        official_path = Path(official_path)
        manifest_path = Path(manifest_path)
        pretrain_path.parent.mkdir(parents=True, exist_ok=True)
        official_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(pretrain_path, **pretrain)
        np.savez_compressed(official_path, **official)

        summary = self.summarize(pretrain)
        summary["official_finetune_samples"] = int(len(official["X"]))
        summary["lookback"] = LOOKBACK
        summary["horizon"] = HORIZON
        summary["targets"] = TARGET_COLUMNS
        summary["normalization"] = (
            "per-sample robust normalization using history median/IQR only; future uses same history transform"
        )
        summary["missing_modalities"] = (
            "zero-filled after normalization and exposed through mask; no unrelated machines are synchronized"
        )
        manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
