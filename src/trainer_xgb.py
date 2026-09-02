import os
import pickle
import sys

import numpy as np
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


def _fit_model(X, y, weights):
    """
    使用 XGBoost 原生多输出回归。

    旧版本使用 MultiOutputRegressor，会在 Python 层创建 576 个独立模型，
    对 GPU 利用率和训练启动开销都不理想。XGBoost 3.x 已支持二维 y，
    因此直接一次 fit 全部 96*6 个输出。
    """
    params = XGB_PARAMS.copy()
    model = XGBRegressor(**params)
    model.fit(X, y, sample_weight=weights)
    return model


def _absolute_from_delta(delta, last_values):
    delta = np.asarray(delta, dtype=np.float64)
    return (
        delta.reshape(-1, HORIZON, len(TARGET_COLUMNS))
        + last_values[:, None, :]
    )


def train():
    print("=" * 70)
    print("XGBoost 训练：连续时序标签 + 防泄漏验证 + GPU 原生多输出")
    print("=" * 70)
    print(f"XGBoost device: {XGB_DEVICE}")
    print(
        f"tree_method={XGB_PARAMS.get('tree_method')}, "
        f"n_estimators={XGB_PARAMS.get('n_estimators')}, "
        f"n_jobs={XGB_PARAMS.get('n_jobs')}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_all_data()
    train_idx, val_idx = temporal_train_val_indices(bundle)
    weights = sample_weights(bundle)

    print(
        f"总样本={len(bundle.X)}, "
        f"train={len(train_idx)}, val={len(val_idx)}, "
        f"feature_dim={bundle.X.shape[1]}, "
        f"output_dim={bundle.y_delta.shape[1]}"
    )

    # 阶段A：验证模型。
    # scaler 只在训练子集 fit，避免验证集信息泄漏。
    scaler_X_val = StandardScaler()
    scaler_y_val = StandardScaler()

    X_train = scaler_X_val.fit_transform(bundle.X[train_idx])
    y_train = scaler_y_val.fit_transform(bundle.y_delta[train_idx])
    X_val = scaler_X_val.transform(bundle.X[val_idx])

    print("开始训练验证 XGBoost ...")
    model_val = _fit_model(
        X_train,
        y_train,
        weights[train_idx],
    )

    pred_scaled_val = np.asarray(model_val.predict(X_val))
    if pred_scaled_val.ndim == 1:
        pred_scaled_val = pred_scaled_val.reshape(len(val_idx), -1)

    pred_delta_val = scaler_y_val.inverse_transform(pred_scaled_val)
    pred_abs_val = _absolute_from_delta(
        pred_delta_val,
        bundle.last_values[val_idx],
    )

    rmse = np.sqrt(
        mean_squared_error(
            bundle.y_abs[val_idx].reshape(len(val_idx), -1),
            pred_abs_val.reshape(len(val_idx), -1),
        )
    )
    print(f"验证集绝对值 RMSE: {rmse:.4f}")

    np.savez_compressed(
        MODEL_DIR / "val_pred_xgb.npz",
        pred_abs=pred_abs_val.astype(np.float32),
        y_abs=bundle.y_abs[val_idx].astype(np.float32),
        last_values=bundle.last_values[val_idx].astype(np.float32),
        baseline_abs=bundle.baseline_abs[val_idx].astype(np.float32),
        sequence_names=bundle.sequence_names[val_idx],
        starts=bundle.starts[val_idx],
    )

    # 阶段B：最终全量重训。
    print("开始全量重训最终 XGBoost ...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_all = scaler_X.fit_transform(bundle.X)
    y_all = scaler_y.fit_transform(bundle.y_delta)

    model_final = _fit_model(
        X_all,
        y_all,
        weights,
    )

    with open(MODEL_DIR / "scaler_xgb.pkl", "wb") as f:
        pickle.dump(
            {"scaler_X": scaler_X, "scaler_y": scaler_y},
            f,
        )

    with open(MODEL_DIR / "model_xgb.pkl", "wb") as f:
        pickle.dump(model_final, f)

    print(f"保存: {MODEL_DIR / 'model_xgb.pkl'}")
    print(f"保存: {MODEL_DIR / 'scaler_xgb.pkl'}")
    print(f"保存: {MODEL_DIR / 'val_pred_xgb.npz'}")


if __name__ == "__main__":
    train()
