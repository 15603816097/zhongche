import os
import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS, XGB_DEVICE, XGB_PARAMS
from src.dataset_builder import (
    load_all_data,
    sample_weights,
    temporal_train_val_indices,
)


PCA_COMPONENTS = 12
MODEL_PATH = MODEL_DIR / "model_pca_xgb.pkl"
PREPROCESS_PATH = MODEL_DIR / "preprocess_pca_xgb.pkl"
VAL_PATH = MODEL_DIR / "val_pred_pca_xgb.npz"


def _params():
    """低秩轨迹模型：输出维度只有 6*12=72，适合当前小样本数据。"""
    params = XGB_PARAMS.copy()
    params.update(
        {
            "n_estimators": 520,
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 4.0,
            "subsample": 0.90,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.20,
            "reg_lambda": 1.8,
        }
    )
    return params


def _delta_trajectories(y_abs, last_values):
    y_abs = np.asarray(y_abs, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)
    return y_abs - last_values[:, None, :]


def _fit_pcas(delta, fit_indices):
    pcas = []
    explained = []
    n_components = min(PCA_COMPONENTS, len(fit_indices), HORIZON)

    for j, col in enumerate(TARGET_COLUMNS):
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(delta[fit_indices, :, j])
        pcas.append(pca)
        explained.append(float(np.sum(pca.explained_variance_ratio_)))
        print(
            f"  PCA {col:16s}: components={n_components} "
            f"explained={explained[-1]:.4f}"
        )

    return pcas, explained


def _encode(delta, pcas):
    blocks = [
        pca.transform(delta[:, :, j])
        for j, pca in enumerate(pcas)
    ]
    return np.concatenate(blocks, axis=1)


def _decode(scores, pcas):
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    out = np.empty((n, HORIZON, len(TARGET_COLUMNS)), dtype=np.float64)

    offset = 0
    for j, pca in enumerate(pcas):
        k = int(pca.n_components_)
        out[:, :, j] = pca.inverse_transform(scores[:, offset:offset + k])
        offset += k

    if offset != scores.shape[1]:
        raise RuntimeError(
            f"PCA score 维度不一致: used={offset}, actual={scores.shape[1]}"
        )
    return out


def _fit_model(X, y, weights):
    model = XGBRegressor(**_params())
    model.fit(X, y, sample_weight=weights)
    return model


def train():
    print("=" * 82)
    print("V6 Low-Rank PCA XGBoost：用低维未来轨迹基学习 96 步形状")
    print("=" * 82)
    print(f"XGBoost device: {XGB_DEVICE}")
    print(f"PCA components per variable: {PCA_COMPONENTS}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_all_data()
    train_idx, val_idx = temporal_train_val_indices(bundle)
    weights = sample_weights(bundle)

    delta = _delta_trajectories(bundle.y_abs, bundle.last_values)

    print(
        f"samples={len(bundle.X)} train={len(train_idx)} val={len(val_idx)} "
        f"feature_dim={bundle.X.shape[1]}"
    )

    # ------------------------------------------------------------------
    # A. 防泄漏验证：PCA / scaler / XGB 都只在 train 段拟合。
    # ------------------------------------------------------------------
    print("\n[A] 拟合验证 PCA ...")
    pcas_val, explained_val = _fit_pcas(delta, train_idx)
    scores_all_val_basis = _encode(delta, pcas_val)

    scaler_X_val = StandardScaler()
    scaler_y_val = StandardScaler()
    X_train = scaler_X_val.fit_transform(bundle.X[train_idx])
    X_val = scaler_X_val.transform(bundle.X[val_idx])
    y_train = scaler_y_val.fit_transform(scores_all_val_basis[train_idx])

    print(f"开始训练验证 PCA-XGBoost，output_dim={y_train.shape[1]} ...")
    model_val = _fit_model(X_train, y_train, weights[train_idx])

    pred_scaled = np.asarray(model_val.predict(X_val))
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(len(val_idx), -1)
    pred_scores = scaler_y_val.inverse_transform(pred_scaled)
    pred_delta = _decode(pred_scores, pcas_val)
    pred_abs = pred_delta + bundle.last_values[val_idx, None, :]

    rmse = float(
        np.sqrt(
            mean_squared_error(
                bundle.y_abs[val_idx].reshape(len(val_idx), -1),
                pred_abs.reshape(len(val_idx), -1),
            )
        )
    )
    print(f"PCA-XGBoost validation absolute RMSE: {rmse:.6f}")

    np.savez_compressed(
        VAL_PATH,
        pred_abs=pred_abs.astype(np.float32),
        y_abs=bundle.y_abs[val_idx].astype(np.float32),
        last_values=bundle.last_values[val_idx].astype(np.float32),
        sequence_names=bundle.sequence_names[val_idx],
        starts=bundle.starts[val_idx],
        rmse=np.asarray(rmse),
        explained_variance=np.asarray(explained_val, dtype=np.float64),
        components=np.asarray([int(p.n_components_) for p in pcas_val], dtype=np.int32),
    )
    print(f"保存: {VAL_PATH}")

    # ------------------------------------------------------------------
    # B. 最终模型：全部样本重新拟合 PCA / scaler / XGB。
    # 当前 API 不会自动使用，只有 V6 候选通过后才启用。
    # ------------------------------------------------------------------
    print("\n[B] 全量拟合最终 PCA ...")
    all_idx = np.arange(len(bundle.X), dtype=np.int64)
    pcas_final, explained_final = _fit_pcas(delta, all_idx)
    scores_all = _encode(delta, pcas_final)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_all = scaler_X.fit_transform(bundle.X)
    y_all = scaler_y.fit_transform(scores_all)

    print(f"开始全量训练最终 PCA-XGBoost，output_dim={y_all.shape[1]} ...")
    model_final = _fit_model(X_all, y_all, weights)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_final, f)
    with open(PREPROCESS_PATH, "wb") as f:
        pickle.dump(
            {
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "pcas": pcas_final,
                "target_columns": list(TARGET_COLUMNS),
                "horizon": HORIZON,
                "components": [int(p.n_components_) for p in pcas_final],
                "explained_variance": explained_final,
            },
            f,
        )

    print(f"保存: {MODEL_PATH}")
    print(f"保存: {PREPROCESS_PATH}")
    print("V6 先只生成候选，不会覆盖当前 ensemble_config.pkl。")


if __name__ == "__main__":
    train()
