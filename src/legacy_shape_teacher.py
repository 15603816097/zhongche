import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import DATA_DIR, HORIZON, TARGET_COLUMNS
from src.data_cleaner import clean_sequence, clean_target_sequence
from src.template_shape import endpoint_zero_future_shape


EPS = 1e-9


def train_sequence_classifier(X, sequence_names, train_idx):
    """用历史特征识别 sequence，复刻最初错误标签模型实际在做的“序列身份识别”。"""
    X = np.asarray(X, dtype=np.float32)
    names = np.asarray(sequence_names).astype(str)
    train_idx = np.asarray(train_idx, dtype=np.int64)

    classes = sorted(np.unique(names[train_idx]).tolist())
    class_to_id = {name: i for i, name in enumerate(classes)}
    y = np.asarray([class_to_id[name] for name in names], dtype=np.int32)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=260,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=18,
        subsample=0.90,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=0.60,
        random_state=42,
        n_jobs=8,
        verbose=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    return model, classes, y


def classifier_probabilities(model, X):
    prob = np.asarray(model.predict_proba(np.asarray(X, dtype=np.float32)), dtype=np.float64)
    if prob.ndim != 2:
        raise RuntimeError(f"sequence classifier predict_proba shape 异常: {prob.shape}")
    prob = np.clip(prob, 0.0, 1.0)
    denom = np.maximum(np.sum(prob, axis=1, keepdims=True), EPS)
    return prob / denom


def classification_diagnostics(prob, y_true):
    y_true = np.asarray(y_true, dtype=np.int32)
    pred = np.argmax(prob, axis=1)
    top1 = np.max(prob, axis=1)
    sorted_prob = np.sort(prob, axis=1)
    top2 = sorted_prob[:, -2] if prob.shape[1] >= 2 else np.zeros(len(prob))
    margin = top1 - top2
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "mean_top1": float(np.mean(top1)),
        "mean_margin": float(np.mean(margin)),
        "top1": top1,
        "margin": margin,
        "pred": pred,
    }


def build_fixed_future_shape_bank(classes):
    """
    直接从每个 sequence 已提供的 future.csv 提取 endpoint-zero 形状。

    这比 V9/V10 的“训练段未来形状中位模板”更贴近最初 55.97 版本：
    最初训练代码对同一 sequence 的所有窗口都使用固定 future[:96] 作为标签。
    """
    shapes = []
    for seq in classes:
        seq_dir = DATA_DIR / seq
        hist_path = seq_dir / "history.csv"
        fut_path = seq_dir / "future.csv"
        if not hist_path.exists() or not fut_path.exists():
            raise FileNotFoundError(f"{seq} 缺少 history.csv 或 future.csv")

        history = clean_sequence(pd.read_csv(hist_path))
        future = clean_target_sequence(pd.read_csv(fut_path))
        if len(future) < HORIZON:
            raise ValueError(f"{seq} future 长度不足 {HORIZON}")

        last = history.iloc[-1][TARGET_COLUMNS].to_numpy(dtype=np.float64)
        y = future.iloc[:HORIZON][TARGET_COLUMNS].to_numpy(dtype=np.float64)
        shape, rms = endpoint_zero_future_shape(y[None, :, :], last[None, :])
        shapes.append(shape[0])

    return {
        "version": 11,
        "classes": list(classes),
        "fixed_future_shapes": np.asarray(shapes, dtype=np.float64),
        "horizon": HORIZON,
        "target_columns": list(TARGET_COLUMNS),
    }


def predict_fixed_future_shape(prob, bank):
    shapes = np.asarray(bank["fixed_future_shapes"], dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    if prob.shape[1] != shapes.shape[0]:
        raise ValueError(f"prob/classes 不一致: {prob.shape} vs {shapes.shape}")
    return np.einsum("ns,shv->nhv", prob, shapes)


def confidence_gate(prob, threshold, power=1.0, use_margin=False):
    prob = np.asarray(prob, dtype=np.float64)
    top1 = np.max(prob, axis=1)
    if use_margin and prob.shape[1] >= 2:
        sorted_prob = np.sort(prob, axis=1)
        conf = top1 * np.sqrt(np.maximum(top1 - sorted_prob[:, -2], 0.0))
    else:
        conf = top1

    threshold = float(threshold)
    if threshold < 0.0:
        gate = np.ones_like(conf)
    else:
        denom = max(1.0 - threshold, 1e-6)
        gate = np.clip((conf - threshold) / denom, 0.0, 1.0)
    gate = gate ** max(float(power), 0.25)
    return gate


def save_teacher(path: Path, model, bank):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"classifier": model, "bank": bank}, f)
