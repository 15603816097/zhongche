import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.feature_engineer import (
    extract_inference_features,
    robust_trend_forecast,
)


_model_lgb = None
_model_xgb = None
_scalers_lgb = None
_scalers_xgb = None
_ensemble_config = None

DEFAULT_LGB_WEIGHTS = np.array(
    [0.65] * len(TARGET_COLUMNS),
    dtype=np.float64,
)
DEFAULT_BASELINE_WEIGHTS = np.array(
    [0.15] * len(TARGET_COLUMNS),
    dtype=np.float64,
)


def load_models():
    global _model_lgb, _model_xgb
    global _scalers_lgb, _scalers_xgb, _ensemble_config

    if _model_lgb is None:
        with open(MODEL_DIR / "model_lgb.pkl", "rb") as f:
            _model_lgb = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            _scalers_lgb = pickle.load(f)

    if _model_xgb is None:
        with open(MODEL_DIR / "model_xgb.pkl", "rb") as f:
            _model_xgb = pickle.load(f)
        with open(MODEL_DIR / "scaler_xgb.pkl", "rb") as f:
            _scalers_xgb = pickle.load(f)

    if _ensemble_config is None:
        config_path = MODEL_DIR / "ensemble_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                _ensemble_config = pickle.load(f)
        else:
            _ensemble_config = {
                "version": 1,
                "lgb_weights": DEFAULT_LGB_WEIGHTS.tolist(),
                "baseline_weights": DEFAULT_BASELINE_WEIGHTS.tolist(),
                "target_columns": list(TARGET_COLUMNS),
            }

    return (
        _model_lgb,
        _model_xgb,
        _scalers_lgb,
        _scalers_xgb,
        _ensemble_config,
    )


def _weights_from_config(config):
    lgb_weights = np.asarray(
        config.get("lgb_weights", DEFAULT_LGB_WEIGHTS),
        dtype=np.float64,
    )
    baseline_weights = np.asarray(
        config.get("baseline_weights", DEFAULT_BASELINE_WEIGHTS),
        dtype=np.float64,
    )

    if lgb_weights.shape != (len(TARGET_COLUMNS),):
        lgb_weights = DEFAULT_LGB_WEIGHTS.copy()
    if baseline_weights.shape != (len(TARGET_COLUMNS),):
        baseline_weights = DEFAULT_BASELINE_WEIGHTS.copy()

    return (
        np.clip(lgb_weights, 0.0, 1.0),
        np.clip(baseline_weights, 0.0, 0.8),
    )


def predict_future(history_df: pd.DataFrame) -> np.ndarray:
    """
    预测未来 96 步绝对值。

    最终预测 =
      (LGB/XGB 逐变量加权) 与 稳健趋势基线 再融合。
    """
    history_clean = clean_sequence(history_df)
    features = extract_inference_features(history_clean)

    (
        model_lgb,
        model_xgb,
        scalers_lgb,
        scalers_xgb,
        ensemble_config,
    ) = load_models()

    X_lgb = scalers_lgb["scaler_X"].transform(
        features.reshape(1, -1)
    )
    delta_lgb_scaled = model_lgb.predict(X_lgb)
    delta_lgb = scalers_lgb["scaler_y"].inverse_transform(
        delta_lgb_scaled
    )[0].reshape(HORIZON, len(TARGET_COLUMNS))

    X_xgb = scalers_xgb["scaler_X"].transform(
        features.reshape(1, -1)
    )
    delta_xgb_scaled = model_xgb.predict(X_xgb)
    delta_xgb = scalers_xgb["scaler_y"].inverse_transform(
        delta_xgb_scaled
    )[0].reshape(HORIZON, len(TARGET_COLUMNS))

    last = history_clean.iloc[-1][TARGET_COLUMNS].to_numpy(
        dtype=np.float64
    )
    pred_lgb = delta_lgb + last.reshape(1, -1)
    pred_xgb = delta_xgb + last.reshape(1, -1)

    lgb_weights, baseline_weights = _weights_from_config(
        ensemble_config
    )

    ml_pred = (
        lgb_weights.reshape(1, -1) * pred_lgb
        + (1.0 - lgb_weights.reshape(1, -1)) * pred_xgb
    )

    baseline = robust_trend_forecast(
        history_clean,
        HORIZON,
    )

    pred = (
        (1.0 - baseline_weights.reshape(1, -1)) * ml_pred
        + baseline_weights.reshape(1, -1) * baseline
    )

    pred = np.asarray(pred, dtype=np.float64)
    bad = ~np.isfinite(pred)
    if np.any(bad):
        fallback = np.tile(last, (HORIZON, 1))
        pred[bad] = fallback[bad]

    return pred
