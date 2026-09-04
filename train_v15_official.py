import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS, XGB_PARAMS
from find_best_weight import evaluate_multivariate, print_report
from src.dataset_builder import load_all_data, sample_weights, temporal_train_val_indices
from src.robust_augmentation import PERTURBATIONS, perturb_features
from src.v8_runtime import v8_parameters


ALPHAS = np.asarray([1.0, 1.0, 1.0, 0.15, 1.0, 0.5], dtype=np.float64)
AUGMENT_WEIGHT = 0.55
PCA_COMPONENTS = 12

ACTIVE_CONFIG_PATH = MODEL_DIR / "ensemble_config.pkl"
V8_VAL_PATH = MODEL_DIR / "val_pred_candidate_v8.npz"
CLEAN_PCA_VAL_PATH = MODEL_DIR / "val_pred_pca_xgb.npz"
CLEAN_PCA_PREPROCESS_PATH = MODEL_DIR / "preprocess_pca_xgb.pkl"

ROBUST_MODEL_PATH = MODEL_DIR / "model_pca_robust_v15.pkl"
ROBUST_PREPROCESS_PATH = MODEL_DIR / "preprocess_pca_robust_v15.pkl"
CANDIDATE_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v15.pkl"
VAL_ARTIFACT_PATH = MODEL_DIR / "val_pred_candidate_v15.pkl"

EPS = 1e-12


def _params():
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
            "verbosity": 0,
        }
    )
    return params


def _delta(y_abs, last_values):
    return np.asarray(y_abs, dtype=np.float64) - np.asarray(
        last_values, dtype=np.float64
    )[:, None, :]


def _fit_pcas(delta, idx):
    pcas = []
    n_components = min(PCA_COMPONENTS, len(idx), HORIZON)
    for j, col in enumerate(TARGET_COLUMNS):
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(delta[idx, :, j])
        pcas.append(pca)
        print(
            f"  PCA {col:16s}: comp={n_components:2d} "
            f"explained={float(np.sum(pca.explained_variance_ratio_)):.4f}"
        )
    return pcas


def _encode(delta, pcas):
    return np.concatenate(
        [pca.transform(delta[:, :, j]) for j, pca in enumerate(pcas)],
        axis=1,
    )


def _decode(scores, pcas):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    out = np.empty(
        (len(scores), HORIZON, len(TARGET_COLUMNS)),
        dtype=np.float64,
    )
    offset = 0
    for j, pca in enumerate(pcas):
        k = int(pca.n_components_)
        out[:, :, j] = pca.inverse_transform(scores[:, offset : offset + k])
        offset += k

    if offset != scores.shape[1]:
        raise RuntimeError(
            f"PCA score 维度异常: used={offset}, actual={scores.shape[1]}"
        )
    return out


def _fit_robust_model(X_clean, scores, weights, seed):
    X_aug = perturb_features(X_clean, PERTURBATIONS, seed=seed)

    X_raw = np.concatenate([X_clean, X_aug], axis=0)
    y_raw = np.concatenate([scores, scores], axis=0)
    w_raw = np.concatenate([weights, AUGMENT_WEIGHT * weights], axis=0)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    model = XGBRegressor(**_params())
    model.fit(X_scaled, y_scaled, sample_weight=w_raw)
    return model, scaler_x, scaler_y


def _predict(model, scaler_x, scaler_y, pcas, X, last_values):
    X_scaled = scaler_x.transform(np.asarray(X, dtype=np.float64))
    pred_scaled = np.asarray(model.predict(X_scaled))
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(len(X), -1)
    pred_scores = scaler_y.inverse_transform(pred_scaled)
    pred_delta = _decode(pred_scores, pcas)
    return pred_delta + np.asarray(last_values, dtype=np.float64)[:, None, :]


def _load_npz_pred(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    if "pred_abs" not in data:
        raise RuntimeError(f"{path} 缺少 pred_abs")
    return data["pred_abs"].astype(np.float64)


def _validate_existing_clean_preprocess(preprocess):
    if list(preprocess.get("target_columns", [])) != list(TARGET_COLUMNS):
        raise RuntimeError("clean PCA preprocess target_columns 不一致")
    if int(preprocess.get("horizon", -1)) != HORIZON:
        raise RuntimeError("clean PCA preprocess horizon 不一致")
    pcas = preprocess.get("pcas")
    if not isinstance(pcas, (list, tuple)) or len(pcas) != len(TARGET_COLUMNS):
        raise RuntimeError("clean PCA preprocess pcas 数量异常")
    return pcas


def main():
    print("=" * 124)
    print("V15 Official Robust-PCA Preparation")
    print("先做 exact V8 integration temporal gate；通过后才训练全量 robust PCA 并生成候选配置。")
    print("不会自动激活 ensemble_config.pkl，也不会修改 app.py/callback。")
    print("=" * 124)
    print(f"locked alphas      : {ALPHAS.tolist()}")
    print(f"augment weight     : {AUGMENT_WEIGHT}")
    print(f"perturbations      : {PERTURBATIONS}")
    print(f"XGB estimators     : {_params().get('n_estimators')}")

    for path in (
        ACTIVE_CONFIG_PATH,
        V8_VAL_PATH,
        CLEAN_PCA_VAL_PATH,
        CLEAN_PCA_PREPROCESS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"缺少文件: {path}")

    with open(ACTIVE_CONFIG_PATH, "rb") as f:
        active = pickle.load(f)

    if int(active.get("version", -1)) != 8:
        raise RuntimeError(
            f"正式准备前要求线上仍为 V8，当前 version={active.get('version')}"
        )

    pca_weights, _, _, _ = v8_parameters(active)
    print(f"V8 PCA weights     : {pca_weights.tolist()}")

    bundle = load_all_data()
    weights_all = sample_weights(bundle)
    train_idx, val_idx = temporal_train_val_indices(bundle)
    delta_all = _delta(bundle.y_abs, bundle.last_values)

    # ------------------------------------------------------------------
    # A. Exact V8 integration temporal gate.
    # ------------------------------------------------------------------
    print("\n[A] Train-only robust PCA for exact V8 integration gate")
    started = time.perf_counter()
    pcas_val = _fit_pcas(delta_all, train_idx)
    scores_all = _encode(delta_all, pcas_val)

    model_val, sx_val, sy_val = _fit_robust_model(
        bundle.X[train_idx],
        scores_all[train_idx],
        weights_all[train_idx],
        seed=15101,
    )
    robust_val = _predict(
        model_val,
        sx_val,
        sy_val,
        pcas_val,
        bundle.X[val_idx],
        bundle.last_values[val_idx],
    )

    clean_pca_val = _load_npz_pred(CLEAN_PCA_VAL_PATH)
    v8_val = _load_npz_pred(V8_VAL_PATH)
    y_true = bundle.y_abs[val_idx].astype(np.float64)
    last_values = bundle.last_values[val_idx].astype(np.float64)

    if clean_pca_val.shape != y_true.shape:
        raise RuntimeError(
            f"clean PCA val shape mismatch: {clean_pca_val.shape} vs {y_true.shape}"
        )
    if v8_val.shape != y_true.shape:
        raise RuntimeError(f"V8 val shape mismatch: {v8_val.shape} vs {y_true.shape}")

    alpha3 = ALPHAS.reshape(1, 1, -1)
    pca_blend_val = (1.0 - alpha3) * clean_pca_val + alpha3 * robust_val

    # V8 high-frequency source不依赖PCA，所以只需替换低频PCA项即可得到精确融合结果。
    candidate_val = v8_val + pca_weights.reshape(1, 1, -1) * (
        pca_blend_val - clean_pca_val
    )

    ref_report = evaluate_multivariate(y_true, v8_val, last_values)
    cand_report = evaluate_multivariate(y_true, candidate_val, last_values)
    print_report("Current official V8 temporal reference", ref_report)
    print_report("V15 exact V8 + clean/robust PCA blend", cand_report)

    ref = ref_report["mean"]
    cand = cand_report["mean"]
    rmse_ok = cand["flat_rmse"] <= ref["flat_rmse"] * 1.004
    proxy_ok = cand["proxy_loss"] <= ref["proxy_loss"] + 0.0008
    diff_ok = cand["diff_corr"] >= ref["diff_corr"] - 0.0003
    peak_ok = cand["peak_f1"] >= ref["peak_f1"] - 0.0020
    vol_ok = cand["volatility_fit"] >= ref["volatility_fit"] - 0.0050
    passed = bool(rmse_ok and proxy_ok and diff_ok and peak_ok and vol_ok)

    print("\nExact integration gates:")
    print(
        f"  RMSE  : {rmse_ok} "
        f"({(cand['flat_rmse']/ref['flat_rmse']-1.0)*100:+.3f}%)"
    )
    print(
        f"  Proxy : {proxy_ok} "
        f"({cand['proxy_loss']-ref['proxy_loss']:+.6f})"
    )
    print(
        f"  Diff  : {diff_ok} "
        f"({cand['diff_corr']-ref['diff_corr']:+.6f})"
    )
    print(
        f"  Peak  : {peak_ok} "
        f"({cand['peak_f1']-ref['peak_f1']:+.6f})"
    )
    print(
        f"  Vol   : {vol_ok} "
        f"({cand['volatility_fit']-ref['volatility_fit']:+.6f})"
    )
    print(f"  validation seconds: {time.perf_counter()-started:.1f}")

    with open(VAL_ARTIFACT_PATH, "wb") as f:
        pickle.dump(
            {
                "pred_abs": candidate_val.astype(np.float32),
                "robust_pca_pred_abs": robust_val.astype(np.float32),
                "alphas": ALPHAS.tolist(),
                "passed": passed,
                "reference": ref,
                "candidate": cand,
            },
            f,
        )

    if not passed:
        print("\nREJECT V15 EXACT INTEGRATION GATE：不训练正式模型，不生成激活候选。")
        print("线上 V8 完全未修改。")
        return

    # ------------------------------------------------------------------
    # B. Train final robust model using the exact final PCA basis already used by V8.
    # ------------------------------------------------------------------
    print("\n[B] Train final all-data robust PCA model")
    final_started = time.perf_counter()
    with open(CLEAN_PCA_PREPROCESS_PATH, "rb") as f:
        clean_preprocess = pickle.load(f)
    final_pcas = _validate_existing_clean_preprocess(clean_preprocess)
    final_scores = _encode(delta_all, final_pcas)

    model_final, sx_final, sy_final = _fit_robust_model(
        bundle.X,
        final_scores,
        weights_all,
        seed=15201,
    )

    with open(ROBUST_MODEL_PATH, "wb") as f:
        pickle.dump(model_final, f)
    with open(ROBUST_PREPROCESS_PATH, "wb") as f:
        pickle.dump(
            {
                "scaler_X": sx_final,
                "scaler_y": sy_final,
                "pcas": final_pcas,
                "target_columns": list(TARGET_COLUMNS),
                "horizon": HORIZON,
                "robust_pca_alphas": ALPHAS.tolist(),
                "augment_weight": AUGMENT_WEIGHT,
                "perturbations": list(PERTURBATIONS),
                "base_clean_preprocess": CLEAN_PCA_PREPROCESS_PATH.name,
            },
            f,
        )

    candidate_config = dict(active)
    candidate_config.update(
        {
            "version": 15,
            "candidate_only": True,
            "base_version": 8,
            "trajectory_model": "pca_xgb_source_aware_hf_robust_blend_v15",
            "robust_pca_alphas": ALPHAS.tolist(),
            "robust_pca_model": ROBUST_MODEL_PATH.name,
            "robust_pca_preprocess": ROBUST_PREPROCESS_PATH.name,
            "v15_v16_stress_validated": True,
            "candidate_passed_local_gate": True,
            "validation_rmse": float(cand["flat_rmse"]),
            "validation_proxy_loss": float(cand["proxy_loss"]),
            "validation_diff_corr": float(cand["diff_corr"]),
            "validation_peak_f1": float(cand["peak_f1"]),
            "validation_volatility_fit": float(cand["volatility_fit"]),
            "validation_direction_accuracy": float(cand["direction_accuracy"]),
        }
    )
    with open(CANDIDATE_CONFIG_PATH, "wb") as f:
        pickle.dump(candidate_config, f)

    print(f"final model       : {ROBUST_MODEL_PATH}")
    print(f"final preprocess  : {ROBUST_PREPROCESS_PATH}")
    print(f"candidate config  : {CANDIDATE_CONFIG_PATH}")
    print(f"final train seconds: {time.perf_counter()-final_started:.1f}")
    print("\nPASS V15 OFFICIAL PREPARATION")
    print("仍未修改 models/ensemble_config.pkl；下一步运行 python activate_v15.py。")


if __name__ == "__main__":
    main()
