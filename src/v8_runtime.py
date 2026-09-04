import pickle
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.trajectory_fusion import endpoint_zero_highpass


_PCA_MODEL = None
_PCA_PREPROCESS = None
_ROBUST_PCA_MODEL = None
_ROBUST_PCA_PREPROCESS = None

PCA_MODEL_PATH = MODEL_DIR / "model_pca_xgb.pkl"
PCA_PREPROCESS_PATH = MODEL_DIR / "preprocess_pca_xgb.pkl"
ROBUST_PCA_MODEL_PATH = MODEL_DIR / "model_pca_robust_v15.pkl"
ROBUST_PCA_PREPROCESS_PATH = MODEL_DIR / "preprocess_pca_robust_v15.pkl"


def _as_float_vector(config: Dict, key: str, default: float = 0.0) -> np.ndarray:
    n_targets = len(TARGET_COLUMNS)
    values = np.asarray(
        config.get(key, [default] * n_targets),
        dtype=np.float64,
    )
    if values.shape != (n_targets,):
        values = np.full(n_targets, default, dtype=np.float64)
    return values


def _as_int_vector(config: Dict, key: str, default: int = 9) -> np.ndarray:
    n_targets = len(TARGET_COLUMNS)
    values = np.asarray(
        config.get(key, [default] * n_targets),
        dtype=np.int32,
    )
    if values.shape != (n_targets,):
        values = np.full(n_targets, default, dtype=np.int32)
    values = np.maximum(values, 1)
    values += (values % 2 == 0).astype(np.int32)
    return values


def _as_source_list(config: Dict, key: str) -> List[str]:
    n_targets = len(TARGET_COLUMNS)
    raw = config.get(key, ["v3"] * n_targets)
    if not isinstance(raw, (list, tuple)) or len(raw) != n_targets:
        raw = ["v3"] * n_targets

    out = []
    for value in raw:
        value = str(value).strip().lower()
        if value not in {"v3", "lgb", "xgb"}:
            value = "v3"
        out.append(value)
    return out


def v8_enabled(config: Dict) -> bool:
    if int(config.get("version", 1)) < 8:
        return False
    model_name = str(config.get("trajectory_model", "")).strip().lower()
    return model_name.startswith("pca_xgb")


def v15_enabled(config: Dict) -> bool:
    if int(config.get("version", 1)) < 15:
        return False
    model_name = str(config.get("trajectory_model", "")).strip().lower()
    if "robust_blend_v15" not in model_name:
        return False
    alphas = _as_float_vector(config, "robust_pca_alphas", 0.0)
    return bool(np.any(alphas > 1e-12))


def v15_alphas(config: Dict) -> np.ndarray:
    return np.clip(
        _as_float_vector(config, "robust_pca_alphas", 0.0),
        0.0,
        1.0,
    )


def v8_parameters(config: Dict) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """读取 V8 在线融合参数，形状全部严格校验。"""
    weights = np.clip(
        _as_float_vector(config, "pca_blend_weights", 0.0),
        0.0,
        1.0,
    )
    gains = np.clip(
        _as_float_vector(config, "v8_highpass_gains", 0.0),
        0.0,
        2.0,
    )
    sources = _as_source_list(config, "v8_highpass_sources")
    windows = _as_int_vector(config, "v8_highpass_windows", 9)
    return weights, gains, sources, windows


def required_lgb_targets_for_v8(config: Dict) -> List[int]:
    """
    V3 为节省时间会跳过 LGB 权重为 0 的输出。

    V8 若某变量把 LGB 作为高频来源，即使 V3 主融合中该变量 LGB 权重为 0，
    仍必须计算该变量完整 96 步 LGB 轨迹，否则离线 V8 与在线 V8 会不一致。
    """
    if not v8_enabled(config):
        return []

    _, gains, sources, _ = v8_parameters(config)
    return [
        j
        for j, source in enumerate(sources)
        if source == "lgb" and float(gains[j]) > 1e-12
    ]


def _validate_preprocess(preprocess: Dict, label: str):
    target_columns = list(preprocess.get("target_columns", []))
    horizon = int(preprocess.get("horizon", -1))
    pcas = preprocess.get("pcas")

    if target_columns != list(TARGET_COLUMNS):
        raise RuntimeError(
            f"{label} target_columns 不一致: {target_columns} vs {TARGET_COLUMNS}"
        )
    if horizon != HORIZON:
        raise RuntimeError(f"{label} horizon 不一致: {horizon} vs {HORIZON}")
    if not isinstance(pcas, (list, tuple)) or len(pcas) != len(TARGET_COLUMNS):
        raise RuntimeError(f"{label} pcas 数量不正确")
    return pcas


def load_pca_runtime():
    global _PCA_MODEL, _PCA_PREPROCESS

    if _PCA_MODEL is None:
        if not PCA_MODEL_PATH.exists():
            raise FileNotFoundError(f"缺少 V8 PCA 模型: {PCA_MODEL_PATH}")
        with open(PCA_MODEL_PATH, "rb") as f:
            _PCA_MODEL = pickle.load(f)

    if _PCA_PREPROCESS is None:
        if not PCA_PREPROCESS_PATH.exists():
            raise FileNotFoundError(f"缺少 V8 PCA 预处理文件: {PCA_PREPROCESS_PATH}")
        with open(PCA_PREPROCESS_PATH, "rb") as f:
            _PCA_PREPROCESS = pickle.load(f)

    _validate_preprocess(_PCA_PREPROCESS, "V8 PCA")
    return _PCA_MODEL, _PCA_PREPROCESS


def load_robust_pca_runtime():
    global _ROBUST_PCA_MODEL, _ROBUST_PCA_PREPROCESS

    if _ROBUST_PCA_MODEL is None:
        if not ROBUST_PCA_MODEL_PATH.exists():
            raise FileNotFoundError(f"缺少 V15 robust PCA 模型: {ROBUST_PCA_MODEL_PATH}")
        with open(ROBUST_PCA_MODEL_PATH, "rb") as f:
            _ROBUST_PCA_MODEL = pickle.load(f)

    if _ROBUST_PCA_PREPROCESS is None:
        if not ROBUST_PCA_PREPROCESS_PATH.exists():
            raise FileNotFoundError(
                f"缺少 V15 robust PCA 预处理文件: {ROBUST_PCA_PREPROCESS_PATH}"
            )
        with open(ROBUST_PCA_PREPROCESS_PATH, "rb") as f:
            _ROBUST_PCA_PREPROCESS = pickle.load(f)

    _validate_preprocess(_ROBUST_PCA_PREPROCESS, "V15 robust PCA")
    return _ROBUST_PCA_MODEL, _ROBUST_PCA_PREPROCESS


def preload_v8_runtime(config: Dict) -> None:
    """API 启动时预加载 PCA 模型，避免第一条官网请求承担模型加载时间。"""
    if v8_enabled(config):
        load_pca_runtime()
    if v15_enabled(config):
        load_robust_pca_runtime()


def _decode_pca_scores(scores: np.ndarray, pcas: Sequence) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    n = scores.shape[0]
    out = np.empty(
        (n, HORIZON, len(TARGET_COLUMNS)),
        dtype=np.float64,
    )

    offset = 0
    for j, pca in enumerate(pcas):
        k = int(pca.n_components_)
        end = offset + k
        if end > scores.shape[1]:
            raise RuntimeError(
                f"PCA score 维度不足: need={end}, actual={scores.shape[1]}"
            )
        out[:, :, j] = pca.inverse_transform(scores[:, offset:end])
        offset = end

    if offset != scores.shape[1]:
        raise RuntimeError(
            f"PCA score 维度不一致: used={offset}, actual={scores.shape[1]}"
        )

    return out


def _predict_pca_pack(model, preprocess, features, last_values):
    scaler_X = preprocess["scaler_X"]
    scaler_y = preprocess["scaler_y"]
    pcas = preprocess["pcas"]

    X = scaler_X.transform(np.asarray(features, dtype=np.float64).reshape(1, -1))
    pred_scaled = np.asarray(model.predict(X))
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(1, -1)

    pred_scores = scaler_y.inverse_transform(pred_scaled)
    pred_delta = _decode_pca_scores(pred_scores, pcas)[0]
    last = np.asarray(last_values, dtype=np.float64).reshape(1, -1)
    return pred_delta + last


def predict_pca_trajectory(features: np.ndarray, last_values: np.ndarray):
    started = time.perf_counter()
    model, preprocess = load_pca_runtime()
    pred_abs = _predict_pca_pack(model, preprocess, features, last_values)
    return pred_abs, time.perf_counter() - started


def predict_robust_pca_trajectory(features: np.ndarray, last_values: np.ndarray):
    started = time.perf_counter()
    model, preprocess = load_robust_pca_runtime()
    pred_abs = _predict_pca_pack(model, preprocess, features, last_values)
    return pred_abs, time.perf_counter() - started


def apply_v8_runtime(
    features: np.ndarray,
    last_values: np.ndarray,
    pred_v3: np.ndarray,
    pred_lgb: np.ndarray,
    pred_xgb: np.ndarray,
    config: Dict,
):
    """
    V8 在线公式：
      low_rank = (1-w)*V3 + w*PCA
      final    = low_rank + gamma*HighPass(source)

    V15 只替换 PCA 低频项：
      PCA* = (1-alpha)*clean_PCA + alpha*robust_PCA

    V8 的 V3/LGB/XGB 高频来源、窗口和 gain 完全保持不变，因此 callback/API
    无需任何改动；V15 只是低频 PCA 分支的保守增强。
    """
    if not v8_enabled(config):
        return np.asarray(pred_v3, dtype=np.float64), 0.0, 0

    weights, gains, sources, windows = v8_parameters(config)
    pca_pred, pca_seconds = predict_pca_trajectory(features, last_values)

    if v15_enabled(config):
        robust_pred, robust_seconds = predict_robust_pca_trajectory(
            features,
            last_values,
        )
        alphas = v15_alphas(config)
        pca_pred = (
            (1.0 - alphas.reshape(1, -1)) * pca_pred
            + alphas.reshape(1, -1) * robust_pred
        )
        pca_seconds += robust_seconds

    pred_v3 = np.asarray(pred_v3, dtype=np.float64)
    pred_lgb = np.asarray(pred_lgb, dtype=np.float64)
    pred_xgb = np.asarray(pred_xgb, dtype=np.float64)

    if pca_pred.shape != pred_v3.shape:
        raise RuntimeError(
            f"PCA 输出 shape 不一致: {pca_pred.shape} vs {pred_v3.shape}"
        )

    source_map = {
        "v3": pred_v3,
        "lgb": pred_lgb,
        "xgb": pred_xgb,
    }

    out = np.empty_like(pred_v3, dtype=np.float64)
    for j in range(len(TARGET_COLUMNS)):
        w = float(weights[j])
        gamma = float(gains[j])
        source_name = sources[j]
        window = int(windows[j])

        low_rank = (1.0 - w) * pred_v3[:, j] + w * pca_pred[:, j]
        if gamma > 1e-12:
            highpass = endpoint_zero_highpass(
                source_map[source_name][:, j],
                window,
            )
            out[:, j] = low_rank + gamma * highpass
        else:
            out[:, j] = low_rank

    return out, pca_seconds, int(np.count_nonzero(weights > 1e-12))
