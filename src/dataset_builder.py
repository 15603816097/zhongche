from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import hashlib

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    HORIZON,
    LOOKBACK,
    MODEL_DIR,
    TARGET_COLUMNS,
    VALIDATION_FRACTION,
    VALIDATION_GAP,
    VALIDATION_MIN_SAMPLES,
)
from src.data_cleaner import clean_sequence, clean_target_sequence
from src.feature_engineer import (
    FEATURE_VERSION,
    extract_features_from_array,
    robust_trend_forecast_array,
)


CACHE_PATH = MODEL_DIR / f"training_dataset_cache_v{FEATURE_VERSION}.npz"


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


def _cache_signature(seq_dirs) -> str:
    """根据数据文件信息和核心时序参数生成缓存签名。"""
    h = hashlib.sha1()
    h.update(str(FEATURE_VERSION).encode())
    h.update(str(LOOKBACK).encode())
    h.update(str(HORIZON).encode())
    h.update("|".join(TARGET_COLUMNS).encode())

    for seq_dir in seq_dirs:
        for filename in ("history.csv", "future.csv"):
            path = seq_dir / filename
            if not path.exists():
                continue
            st = path.stat()
            h.update(str(path.relative_to(DATA_DIR)).encode())
            h.update(str(st.st_size).encode())
            h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()


def _save_cache(bundle: DatasetBundle, signature: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        signature=np.asarray(signature),
        X=bundle.X,
        y_delta=bundle.y_delta,
        y_abs=bundle.y_abs,
        last_values=bundle.last_values,
        baseline_abs=bundle.baseline_abs,
        sequence_names=bundle.sequence_names.astype(str),
        starts=bundle.starts,
    )
    print(f"训练数据缓存已保存: {CACHE_PATH}")


def _load_cache(signature: str):
    if not CACHE_PATH.exists():
        return None
    try:
        with np.load(CACHE_PATH, allow_pickle=False) as data:
            cached_signature = str(data["signature"].item())
            if cached_signature != signature:
                print("训练数据缓存已过期，重新生成。")
                return None

            bundle = DatasetBundle(
                X=data["X"].astype(np.float32, copy=False),
                y_delta=data["y_delta"].astype(np.float32, copy=False),
                y_abs=data["y_abs"].astype(np.float32, copy=False),
                last_values=data["last_values"].astype(np.float32, copy=False),
                baseline_abs=data["baseline_abs"].astype(np.float32, copy=False),
                sequence_names=data["sequence_names"].astype(str),
                starts=data["starts"].astype(np.int32, copy=False),
            )
        print(
            f"直接加载训练数据缓存: {CACHE_PATH} "
            f"samples={len(bundle.X)}, feature_dim={bundle.X.shape[1]}"
        )
        return bundle
    except Exception as exc:
        print(f"训练数据缓存读取失败，将重新生成: {exc}")
        return None


def build_sequence_samples(
    history: pd.DataFrame,
    future: pd.DataFrame,
    sequence_name: str,
) -> DatasetBundle:
    """
    正确的连续时序监督构造。

    最后一条样本严格对应官网任务：
      history[-LOOKBACK:] -> future[:HORIZON]

    本版本训练循环全部使用 NumPy 切片，避免每个窗口反复创建 DataFrame。
    """
    history_input_df = clean_sequence(history)
    history_target_df = clean_target_sequence(history)
    future_target_df = clean_target_sequence(future)

    history_input = history_input_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    target_timeline = np.concatenate(
        [
            history_target_df[TARGET_COLUMNS].to_numpy(dtype=np.float64),
            future_target_df[TARGET_COLUMNS].to_numpy(dtype=np.float64),
        ],
        axis=0,
    )

    max_start_by_input = len(history_input) - LOOKBACK
    max_start_by_target = len(target_timeline) - LOOKBACK - HORIZON
    max_start = min(max_start_by_input, max_start_by_target)

    if max_start < 0:
        raise ValueError(
            f"{sequence_name} 长度不足: "
            f"history={len(history_input)}, future={len(future_target_df)}"
        )

    n_samples = max_start + 1
    X = np.empty((n_samples, 1242), dtype=np.float32)
    y_delta = np.empty(
        (n_samples, HORIZON * len(TARGET_COLUMNS)), dtype=np.float32
    )
    y_abs = np.empty(
        (n_samples, HORIZON, len(TARGET_COLUMNS)), dtype=np.float32
    )
    last_values = np.empty((n_samples, len(TARGET_COLUMNS)), dtype=np.float32)
    baseline_abs = np.empty_like(y_abs)
    starts = np.arange(n_samples, dtype=np.int32)
    sequence_names = np.full(n_samples, sequence_name, dtype=f"<U{max(16, len(sequence_name))}")

    for start in range(n_samples):
        window = history_input[start:start + LOOKBACK]
        target_start = start + LOOKBACK
        target = target_timeline[target_start:target_start + HORIZON]

        if target.shape != (HORIZON, len(TARGET_COLUMNS)):
            raise RuntimeError(
                f"{sequence_name} start={start} 标签 shape 异常: {target.shape}"
            )

        last = window[-1]
        delta = target - last[None, :]

        X[start] = extract_features_from_array(window)
        y_delta[start] = delta.reshape(-1)
        y_abs[start] = target
        last_values[start] = last
        baseline_abs[start] = robust_trend_forecast_array(window, HORIZON)

        if (start + 1) % 100 == 0 or start + 1 == n_samples:
            print(
                f"    {sequence_name} 特征进度: "
                f"{start + 1}/{n_samples}"
            )

    return DatasetBundle(
        X=X,
        y_delta=y_delta,
        y_abs=y_abs,
        last_values=last_values,
        baseline_abs=baseline_abs,
        sequence_names=sequence_names,
        starts=starts,
    )


def load_all_data(use_cache: bool = True) -> DatasetBundle:
    seq_dirs = _sequence_dirs()

    if not seq_dirs:
        raise FileNotFoundError(f"未在 {DATA_DIR} 找到 sequence* 目录")

    signature = _cache_signature(seq_dirs)
    if use_cache:
        cached = _load_cache(signature)
        if cached is not None:
            return cached

    bundles: List[DatasetBundle] = []
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

    bundle = DatasetBundle(
        X=np.concatenate([b.X for b in bundles], axis=0),
        y_delta=np.concatenate([b.y_delta for b in bundles], axis=0),
        y_abs=np.concatenate([b.y_abs for b in bundles], axis=0),
        last_values=np.concatenate([b.last_values for b in bundles], axis=0),
        baseline_abs=np.concatenate([b.baseline_abs for b in bundles], axis=0),
        sequence_names=np.concatenate([b.sequence_names for b in bundles], axis=0),
        starts=np.concatenate([b.starts for b in bundles], axis=0),
    )

    if use_cache:
        _save_cache(bundle, signature)
    return bundle


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
    """越靠近 history 末端的样本越接近官网真实推理边界，权重越高。"""
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
