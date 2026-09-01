from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    HORIZON,
    LOOKBACK,
    TARGET_COLUMNS,
    VALIDATION_FRACTION,
    VALIDATION_GAP,
    VALIDATION_MIN_SAMPLES,
)
from src.data_cleaner import clean_sequence, clean_target_sequence
from src.feature_engineer import (
    extract_features_from_window,
    robust_trend_forecast,
)


@dataclass
class DatasetBundle:
    X: np.ndarray
    y_delta: np.ndarray
    y_abs: np.ndarray
    last_values: np.ndarray
    baseline_abs: np.ndarray
    sequence_names: np.ndarray
    starts: np.ndarray


def _sequence_dirs():
    return sorted(
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith("sequence")
    )


def build_sequence_samples(
    history: pd.DataFrame,
    future: pd.DataFrame,
    sequence_name: str,
) -> DatasetBundle:
    """
    正确的连续时序监督构造。

    对每个 start：
      输入 = history[start : start + LOOKBACK]
      标签 = 紧接输入后的 HORIZON 步

    输入始终只来自 history；标签来自 history 后续或 future。
    最后一条样本恰好对应官方任务：
      history[-LOOKBACK:] -> future[:HORIZON]
    """
    history_input = clean_sequence(history)
    history_target = clean_target_sequence(history)
    future_target = clean_target_sequence(future)

    target_timeline = pd.concat(
        [
            history_target[TARGET_COLUMNS],
            future_target[TARGET_COLUMNS],
        ],
        ignore_index=True,
    )

    max_start_by_input = len(history_input) - LOOKBACK
    max_start_by_target = len(target_timeline) - LOOKBACK - HORIZON
    max_start = min(max_start_by_input, max_start_by_target)

    if max_start < 0:
        raise ValueError(
            f"{sequence_name} 长度不足: "
            f"history={len(history_input)}, future={len(future_target)}"
        )

    X, y_delta, y_abs = [], [], []
    last_values, baseline_abs = [], []
    sequence_names, starts = [], []

    for start in range(max_start + 1):
        window = history_input.iloc[
            start:start + LOOKBACK
        ][TARGET_COLUMNS]

        target_start = start + LOOKBACK
        target = target_timeline.iloc[
            target_start:target_start + HORIZON
        ][TARGET_COLUMNS].to_numpy(dtype=np.float64)

        if target.shape != (HORIZON, len(TARGET_COLUMNS)):
            continue

        last = window.iloc[-1].to_numpy(dtype=np.float64)
        delta = target - last.reshape(1, -1)

        X.append(extract_features_from_window(window))
        y_delta.append(delta.reshape(-1))
        y_abs.append(target)
        last_values.append(last)
        baseline_abs.append(robust_trend_forecast(window, HORIZON))
        sequence_names.append(sequence_name)
        starts.append(start)

    return DatasetBundle(
        X=np.asarray(X, dtype=np.float32),
        y_delta=np.asarray(y_delta, dtype=np.float32),
        y_abs=np.asarray(y_abs, dtype=np.float32),
        last_values=np.asarray(last_values, dtype=np.float32),
        baseline_abs=np.asarray(baseline_abs, dtype=np.float32),
        sequence_names=np.asarray(sequence_names),
        starts=np.asarray(starts, dtype=np.int32),
    )


def load_all_data() -> DatasetBundle:
    bundles: List[DatasetBundle] = []
    seq_dirs = _sequence_dirs()

    if not seq_dirs:
        raise FileNotFoundError(f"未在 {DATA_DIR} 找到 sequence* 目录")

    print(f"找到 {len(seq_dirs)} 个序列: {[d.name for d in seq_dirs]}")

    for seq_dir in seq_dirs:
        history_path = seq_dir / "history.csv"
        future_path = seq_dir / "future.csv"

        if not history_path.exists() or not future_path.exists():
            print(f"跳过 {seq_dir.name}: 缺少 history.csv 或 future.csv")
            continue

        history = pd.read_csv(history_path)
        future = pd.read_csv(future_path)
        bundle = build_sequence_samples(history, future, seq_dir.name)
        bundles.append(bundle)

        print(
            f"  {seq_dir.name}: "
            f"history={len(history)}, future={len(future)}, "
            f"samples={len(bundle.X)}, "
            f"last_start={int(bundle.starts[-1])}"
        )

    if not bundles:
        raise RuntimeError("没有可用训练数据")

    return DatasetBundle(
        X=np.concatenate([b.X for b in bundles], axis=0),
        y_delta=np.concatenate([b.y_delta for b in bundles], axis=0),
        y_abs=np.concatenate([b.y_abs for b in bundles], axis=0),
        last_values=np.concatenate([b.last_values for b in bundles], axis=0),
        baseline_abs=np.concatenate([b.baseline_abs for b in bundles], axis=0),
        sequence_names=np.concatenate(
            [b.sequence_names for b in bundles], axis=0
        ),
        starts=np.concatenate([b.starts for b in bundles], axis=0),
    )


def temporal_train_val_indices(
    bundle: DatasetBundle,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    每个 sequence 独立做时间切分，并在 train / val 之间加 embargo gap，
    避免训练标签进入验证预测区间造成过度乐观的验证结果。
    """
    train_idx: List[int] = []
    val_idx: List[int] = []

    for seq in np.unique(bundle.sequence_names):
        idx = np.where(bundle.sequence_names == seq)[0]
        idx = idx[np.argsort(bundle.starts[idx])]
        n = len(idx)

        val_count = max(
            VALIDATION_MIN_SAMPLES,
            int(round(n * VALIDATION_FRACTION)),
        )
        val_count = min(val_count, max(1, n // 3))
        val_start_pos = n - val_count

        gap = min(VALIDATION_GAP, max(0, val_start_pos - 1))
        train_end_pos = val_start_pos - gap

        if train_end_pos <= 0:
            train_end_pos = max(1, val_start_pos // 2)

        train_idx.extend(idx[:train_end_pos].tolist())
        val_idx.extend(idx[val_start_pos:].tolist())

        print(
            f"  split {seq}: train={train_end_pos}, "
            f"gap={val_start_pos-train_end_pos}, val={n-val_start_pos}"
        )

    return (
        np.asarray(sorted(train_idx), dtype=np.int64),
        np.asarray(sorted(val_idx), dtype=np.int64),
    )


def sample_weights(bundle: DatasetBundle) -> np.ndarray:
    """
    越靠近 history 末端的样本越接近官方真实推理边界，因此给予更高权重。
    """
    weights = np.ones(len(bundle.X), dtype=np.float64)

    for seq in np.unique(bundle.sequence_names):
        idx = np.where(bundle.sequence_names == seq)[0]
        starts = bundle.starts[idx].astype(np.float64)
        max_start = max(float(starts.max()), 1.0)
        relative = starts / max_start
        weights[idx] = 1.0 + 1.5 * (relative ** 2)

        boundary_idx = idx[starts == starts.max()]
        weights[boundary_idx] = np.maximum(weights[boundary_idx], 4.0)

    return weights
