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


TREND_MODEL_PATH = MODEL_DIR / "model_trend_xgb.pkl"
TREND_SCALER_PATH = MODEL_DIR / "scaler_trend_xgb.pkl"
TREND_VAL_PATH = MODEL_DIR / "val_pred_trend_xgb.npz"


def _trend_params():
    """趋势模型比数值模型略保守，降低小数据过拟合。"""
    params = XGB_PARAMS.copy()
    params.update(
        {
            "n_estimators": 460,
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 4.0,
            "subsample": 0.90,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.18,
            "reg_lambda": 1.8,
        }
    )
    return params


def _future_step_differences(y_abs, last_values):
    """
    把未来绝对轨迹转换成真正的一阶差分：
      d1 = y1 - last_history
      d2 = y2 - y1
      ...
    这样模型直接学习官网趋势指标最关心的逐步变化。
    """
    y_abs = np.asarray(y_abs, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)
    extended = np.concatenate(
        [last_values[:, None, :], y_abs],
        axis=1,
    )
    return np.diff(extended, axis=1)


def _absolute_from_step_diff(step_diff, last_values):
    step_diff = np.asarray(step_diff, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)
    return last_values[:, None, :] + np.cumsum(step_diff, axis=1)


def _fit_model(X, y, weights):
    model = XGBRegressor(**_trend_params())
    model.fit(X, y, sample_weight=weights)
    return model


def _diff_corr(y_true_diff, y_pred_diff):
    a = np.asarray(y_true_diff, dtype=np.float64).reshape(-1)
    b = np.asarray(y_pred_diff, dtype=np.float64).reshape(-1)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def train():
    print("=" * 78)
    print("V5 Trend XGBoost：直接预测未来 96x6 一阶差分")
    print("=" * 78)
    print(f"XGBoost device: {XGB_DEVICE}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_all_data()
    train_idx, val_idx = temporal_train_val_indices(bundle)
    weights = sample_weights(bundle)

    y_step_diff = _future_step_differences(
        bundle.y_abs,
        bundle.last_values,
    ).astype(np.float32)
    y_flat = y_step_diff.reshape(len(y_step_diff), -1)

    print(
        f"samples={len(bundle.X)} train={len(train_idx)} val={len(val_idx)} "
        f"feature_dim={bundle.X.shape[1]} output_dim={y_flat.shape[1]}"
    )

    # ------------------------------------------------------------------
    # A. 验证模型：scaler 只看 train，防止验证泄漏。
    # ------------------------------------------------------------------
    scaler_X_val = StandardScaler()
    scaler_y_val = StandardScaler()

    X_train = scaler_X_val.fit_transform(bundle.X[train_idx])
    X_val = scaler_X_val.transform(bundle.X[val_idx])
    y_train = scaler_y_val.fit_transform(y_flat[train_idx])

    print("开始训练验证 Trend XGBoost ...")
    model_val = _fit_model(
        X_train,
        y_train,
        weights[train_idx],
    )

    pred_scaled = np.asarray(model_val.predict(X_val))
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(len(val_idx), -1)

    pred_step_diff = scaler_y_val.inverse_transform(pred_scaled).reshape(
        len(val_idx), HORIZON, len(TARGET_COLUMNS)
    )
    true_step_diff = y_step_diff[val_idx].astype(np.float64)

    pred_abs = _absolute_from_step_diff(
        pred_step_diff,
        bundle.last_values[val_idx],
    )

    absolute_rmse = float(
        np.sqrt(
            mean_squared_error(
                bundle.y_abs[val_idx].reshape(len(val_idx), -1),
                pred_abs.reshape(len(val_idx), -1),
            )
        )
    )
    diff_rmse = float(np.sqrt(np.mean((true_step_diff - pred_step_diff) ** 2)))
    diff_corr = _diff_corr(true_step_diff, pred_step_diff)

    print(f"Trend model absolute RMSE: {absolute_rmse:.6f}")
    print(f"Trend model diff RMSE    : {diff_rmse:.6f}")
    print(f"Trend model diff Corr    : {diff_corr:.6f}")

    np.savez_compressed(
        TREND_VAL_PATH,
        pred_abs=pred_abs.astype(np.float32),
        pred_step_diff=pred_step_diff.astype(np.float32),
        true_step_diff=true_step_diff.astype(np.float32),
        y_abs=bundle.y_abs[val_idx].astype(np.float32),
        last_values=bundle.last_values[val_idx].astype(np.float32),
        sequence_names=bundle.sequence_names[val_idx],
        starts=bundle.starts[val_idx],
        absolute_rmse=np.asarray(absolute_rmse),
        diff_rmse=np.asarray(diff_rmse),
        diff_corr=np.asarray(diff_corr),
    )
    print(f"保存: {TREND_VAL_PATH}")

    # ------------------------------------------------------------------
    # B. 全量重训：只有 V5 搜索真正接受时，inference 才会使用这个模型。
    # ------------------------------------------------------------------
    print("开始全量重训最终 Trend XGBoost ...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_all = scaler_X.fit_transform(bundle.X)
    y_all = scaler_y.fit_transform(y_flat)

    model_final = _fit_model(X_all, y_all, weights)

    with open(TREND_SCALER_PATH, "wb") as f:
        pickle.dump(
            {"scaler_X": scaler_X, "scaler_y": scaler_y},
            f,
        )
    with open(TREND_MODEL_PATH, "wb") as f:
        pickle.dump(model_final, f)

    print(f"保存: {TREND_MODEL_PATH}")
    print(f"保存: {TREND_SCALER_PATH}")
    print("V5 搜索未通过之前，当前 ensemble_config 不会启用该模型。")


if __name__ == "__main__":
    train()
