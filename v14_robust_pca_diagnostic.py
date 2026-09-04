import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import HORIZON, LOOKBACK, MODEL_DIR, TARGET_COLUMNS, XGB_DEVICE, XGB_PARAMS
from find_best_weight import evaluate_multivariate
from src.dataset_builder import load_all_data, sample_weights, temporal_train_val_indices
from src.feature_engineer import extract_features_from_array


ARTIFACT_PATH = MODEL_DIR / "v14_robust_pca_diagnostic.npz"
PCA_COMPONENTS = 12
ESTIMATORS = int(os.getenv("V14_ESTIMATORS", "320"))
AUGMENT_WEIGHT = 0.55
EPS = 1e-9

PERTURBATIONS = (
    "noise",
    "missing_random",
    "missing_block",
    "bias",
    "drift",
)


def _params():
    params = XGB_PARAMS.copy()
    params.update(
        {
            "n_estimators": ESTIMATORS,
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


def _raw_windows(X):
    raw_dim = LOOKBACK * len(TARGET_COLUMNS)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < raw_dim:
        raise ValueError(f"X shape 异常: {X.shape}")
    return X[:, :raw_dim].reshape(-1, LOOKBACK, len(TARGET_COLUMNS)).copy()


def _interp_nan_column(x):
    x = np.asarray(x, dtype=np.float64).copy()
    good = np.isfinite(x)
    if np.all(good):
        return x
    if not np.any(good):
        return np.zeros_like(x)
    idx = np.arange(len(x), dtype=np.float64)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _repair_missing(raw):
    out = np.asarray(raw, dtype=np.float64).copy()
    for j in range(out.shape[1]):
        out[:, j] = _interp_nan_column(out[:, j])
    return out


def _scales(raw):
    raw = np.asarray(raw, dtype=np.float64)
    recent = raw[-48:]
    level = np.maximum(np.median(np.abs(recent), axis=0), 1.0)
    value_std = np.std(recent, axis=0)
    diff_std = np.std(np.diff(recent, axis=0), axis=0)
    local = np.maximum(diff_std, 0.02 * value_std)
    local = np.maximum(local, 2e-4 * level)
    return level, value_std, local


def perturb_window(raw, kind, rng):
    out = np.asarray(raw, dtype=np.float64).copy()
    _, value_std, local = _scales(out)
    n, d = out.shape

    if kind == "noise":
        sigma = 0.30 * local
        out += rng.normal(0.0, 1.0, size=out.shape) * sigma[None, :]

    elif kind == "missing_random":
        mask = rng.random(out.shape) < 0.045
        # 保住最后一点，避免把任务锚点本身直接抹掉。
        mask[-1, :] = False
        out[mask] = np.nan
        out = _repair_missing(out)

    elif kind == "missing_block":
        n_vars = int(rng.integers(1, min(4, d) + 1))
        cols = rng.choice(d, size=n_vars, replace=False)
        block = int(rng.integers(6, 17))
        start = int(rng.integers(max(0, n - 64), max(1, n - block)))
        end = min(n - 1, start + block)
        out[start:end, cols] = np.nan
        out = _repair_missing(out)

    elif kind == "bias":
        n_vars = int(rng.integers(1, min(4, d) + 1))
        cols = rng.choice(d, size=n_vars, replace=False)
        sign = rng.choice([-1.0, 1.0], size=n_vars)
        magnitude = np.maximum(0.10 * value_std[cols], 1.2 * local[cols])
        out[:, cols] += sign[None, :] * magnitude[None, :]

    elif kind == "drift":
        n_vars = int(rng.integers(1, min(4, d) + 1))
        cols = rng.choice(d, size=n_vars, replace=False)
        sign = rng.choice([-1.0, 1.0], size=n_vars)
        end_mag = np.maximum(0.16 * value_std[cols], 1.8 * local[cols])
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
        out[:, cols] += ramp * sign[None, :] * end_mag[None, :]

    else:
        raise ValueError(f"未知 perturbation: {kind}")

    return np.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)


def perturb_features(X, kinds, seed):
    raw = _raw_windows(X)
    rng = np.random.default_rng(seed)
    out = np.empty((len(raw), 1242), dtype=np.float32)
    for i in range(len(raw)):
        kind = kinds[i % len(kinds)]
        perturbed = perturb_window(raw[i], kind, rng)
        out[i] = extract_features_from_array(perturbed)
    return out


def stress_features(X, seed):
    blocks = []
    labels = []
    for k, kind in enumerate(PERTURBATIONS):
        kinds = [kind]
        feat = perturb_features(X, kinds, seed + 1000 * (k + 1))
        blocks.append(feat)
        labels.extend([kind] * len(feat))
    return np.concatenate(blocks, axis=0), np.asarray(labels, dtype=object)


def _delta(y_abs, last_values):
    return np.asarray(y_abs, dtype=np.float64) - np.asarray(last_values, dtype=np.float64)[:, None, :]


def _fit_pcas(delta, idx):
    pcas = []
    n_components = min(PCA_COMPONENTS, len(idx), HORIZON)
    for j, col in enumerate(TARGET_COLUMNS):
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(delta[idx, :, j])
        pcas.append(pca)
        print(
            f"    PCA {col:16s}: comp={n_components:2d} "
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
    out = np.empty((len(scores), HORIZON, len(TARGET_COLUMNS)), dtype=np.float64)
    offset = 0
    for j, pca in enumerate(pcas):
        k = int(pca.n_components_)
        out[:, :, j] = pca.inverse_transform(scores[:, offset:offset + k])
        offset += k
    if offset != scores.shape[1]:
        raise RuntimeError(f"PCA score 维度异常: used={offset}, actual={scores.shape[1]}")
    return out


def _fit_pair(bundle, train_idx, pcas, base_weights, seed):
    scores = _encode(_delta(bundle.y_abs, bundle.last_values), pcas)

    # Clean baseline.
    scaler_x_clean = StandardScaler()
    scaler_y_clean = StandardScaler()
    X_clean = scaler_x_clean.fit_transform(bundle.X[train_idx])
    y_clean = scaler_y_clean.fit_transform(scores[train_idx])
    model_clean = XGBRegressor(**_params())
    model_clean.fit(X_clean, y_clean, sample_weight=base_weights[train_idx])

    # Robust model: clean + one deterministic perturbation per training sample.
    aug_features = perturb_features(
        bundle.X[train_idx],
        PERTURBATIONS,
        seed=seed,
    )
    X_robust_raw = np.concatenate([bundle.X[train_idx], aug_features], axis=0)
    y_robust_raw = np.concatenate([scores[train_idx], scores[train_idx]], axis=0)
    w_clean = base_weights[train_idx]
    w_robust = np.concatenate([w_clean, AUGMENT_WEIGHT * w_clean], axis=0)

    scaler_x_robust = StandardScaler()
    scaler_y_robust = StandardScaler()
    X_robust = scaler_x_robust.fit_transform(X_robust_raw)
    y_robust = scaler_y_robust.fit_transform(y_robust_raw)
    model_robust = XGBRegressor(**_params())
    model_robust.fit(X_robust, y_robust, sample_weight=w_robust)

    return {
        "pcas": pcas,
        "clean": (model_clean, scaler_x_clean, scaler_y_clean),
        "robust": (model_robust, scaler_x_robust, scaler_y_robust),
    }


def _predict(model_pack, features, last_values):
    model, scaler_x, scaler_y = model_pack
    X = scaler_x.transform(np.asarray(features, dtype=np.float64))
    pred_scaled = np.asarray(model.predict(X))
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(len(features), -1)
    scores = scaler_y.inverse_transform(pred_scaled)
    pred_delta = _decode(scores, _predict.pcas)
    return pred_delta + np.asarray(last_values, dtype=np.float64)[:, None, :]


def _predict_with_pcas(model_pack, pcas, features, last_values):
    model, scaler_x, scaler_y = model_pack
    X = scaler_x.transform(np.asarray(features, dtype=np.float64))
    pred_scaled = np.asarray(model.predict(X))
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(len(features), -1)
    scores = scaler_y.inverse_transform(pred_scaled)
    pred_delta = _decode(scores, pcas)
    return pred_delta + np.asarray(last_values, dtype=np.float64)[:, None, :]


def _repeat_targets(y, last, repeats):
    return np.concatenate([y] * repeats, axis=0), np.concatenate([last] * repeats, axis=0)


def _short(report):
    m = report["mean"]
    return (
        f"RMSE={m['flat_rmse']:.4f} meanRMSE={m['rmse']:.4f} "
        f"Diff={m['diff_corr']:.4f} Peak={m['peak_f1']:.4f} "
        f"Vol={m['volatility_fit']:.4f} Proxy={m['proxy_loss']:.4f}"
    )


def _degradation(clean_report, stress_report):
    c = clean_report["mean"]
    s = stress_report["mean"]
    return float(s["flat_rmse"] / max(c["flat_rmse"], EPS) - 1.0)


def main():
    print("=" * 124)
    print("V14 Robust PCA-XGBoost Augmentation Diagnostic")
    print("训练时仅扰动输入窗口，未来标签保持不变；每次扰动后重新计算完整1242维特征。")
    print("同时检查 temporal clean/stress + LOSO clean/stress + unseen boundary stress。")
    print("本脚本不修改 V8、API、callback 或 ensemble_config.pkl。")
    print("=" * 124)
    print(f"device={XGB_DEVICE} estimators={ESTIMATORS} augment_weight={AUGMENT_WEIGHT}")
    print(f"perturbations={PERTURBATIONS}")

    bundle = load_all_data()
    weights = sample_weights(bundle)
    delta_all = _delta(bundle.y_abs, bundle.last_values)
    train_idx, val_idx = temporal_train_val_indices(bundle)

    # ------------------------------------------------------------------
    # A. Temporal clean/stress comparison.
    # ------------------------------------------------------------------
    print("\n[A] Temporal robust diagnostic")
    pcas_t = _fit_pcas(delta_all, train_idx)
    pair_t = _fit_pair(bundle, train_idx, pcas_t, weights, seed=14001)

    clean_base = _predict_with_pcas(pair_t["clean"], pcas_t, bundle.X[val_idx], bundle.last_values[val_idx])
    clean_rob = _predict_with_pcas(pair_t["robust"], pcas_t, bundle.X[val_idx], bundle.last_values[val_idx])

    stress_X_t, _ = stress_features(bundle.X[val_idx], seed=14101)
    y_stress_t, last_stress_t = _repeat_targets(
        bundle.y_abs[val_idx], bundle.last_values[val_idx], len(PERTURBATIONS)
    )
    stress_base = _predict_with_pcas(pair_t["clean"], pcas_t, stress_X_t, last_stress_t)
    stress_rob = _predict_with_pcas(pair_t["robust"], pcas_t, stress_X_t, last_stress_t)

    t_clean_base_r = evaluate_multivariate(bundle.y_abs[val_idx], clean_base, bundle.last_values[val_idx])
    t_clean_rob_r = evaluate_multivariate(bundle.y_abs[val_idx], clean_rob, bundle.last_values[val_idx])
    t_stress_base_r = evaluate_multivariate(y_stress_t, stress_base, last_stress_t)
    t_stress_rob_r = evaluate_multivariate(y_stress_t, stress_rob, last_stress_t)

    print(f"  temporal clean baseline: {_short(t_clean_base_r)}")
    print(f"  temporal clean robust  : {_short(t_clean_rob_r)}")
    print(f"  temporal stress baseline: {_short(t_stress_base_r)}")
    print(f"  temporal stress robust  : {_short(t_stress_rob_r)}")
    print(
        f"  stress degradation baseline={_degradation(t_clean_base_r, t_stress_base_r)*100:+.2f}% "
        f"robust={_degradation(t_clean_rob_r, t_stress_rob_r)*100:+.2f}%"
    )

    # ------------------------------------------------------------------
    # B. LOSO clean/stress comparison.
    # ------------------------------------------------------------------
    print("\n[B] LOSO robust diagnostic")
    sequences = sorted(np.unique(bundle.sequence_names).tolist())
    n = len(bundle.X)
    clean_oof_base = np.full_like(bundle.y_abs, np.nan, dtype=np.float64)
    clean_oof_rob = np.full_like(bundle.y_abs, np.nan, dtype=np.float64)
    stress_oof_base = []
    stress_oof_rob = []
    stress_oof_y = []
    stress_oof_last = []

    boundary_idx_all = []
    boundary_stress_base = []
    boundary_stress_rob = []
    boundary_stress_y = []
    boundary_stress_last = []

    started = time.perf_counter()
    for fold, heldout in enumerate(sequences):
        test_idx = np.where(bundle.sequence_names == heldout)[0]
        train_fold = np.where(bundle.sequence_names != heldout)[0]
        print("\n" + "-" * 124)
        print(f"[FOLD {fold+1}/{len(sequences)}] heldout={heldout} train={len(train_fold)} test={len(test_idx)}")
        fold_started = time.perf_counter()

        pcas = _fit_pcas(delta_all, train_fold)
        pair = _fit_pair(bundle, train_fold, pcas, weights, seed=14200 + fold)

        pred_base = _predict_with_pcas(pair["clean"], pcas, bundle.X[test_idx], bundle.last_values[test_idx])
        pred_rob = _predict_with_pcas(pair["robust"], pcas, bundle.X[test_idx], bundle.last_values[test_idx])
        clean_oof_base[test_idx] = pred_base
        clean_oof_rob[test_idx] = pred_rob

        stress_X, _ = stress_features(bundle.X[test_idx], seed=14300 + fold)
        y_stress, last_stress = _repeat_targets(
            bundle.y_abs[test_idx], bundle.last_values[test_idx], len(PERTURBATIONS)
        )
        stress_oof_base.append(_predict_with_pcas(pair["clean"], pcas, stress_X, last_stress))
        stress_oof_rob.append(_predict_with_pcas(pair["robust"], pcas, stress_X, last_stress))
        stress_oof_y.append(y_stress)
        stress_oof_last.append(last_stress)

        bidx = test_idx[bundle.starts[test_idx] == np.max(bundle.starts[test_idx])]
        boundary_idx_all.extend(bidx.tolist())
        bX, _ = stress_features(bundle.X[bidx], seed=14400 + fold)
        by, blast = _repeat_targets(
            bundle.y_abs[bidx], bundle.last_values[bidx], len(PERTURBATIONS)
        )
        boundary_stress_base.append(_predict_with_pcas(pair["clean"], pcas, bX, blast))
        boundary_stress_rob.append(_predict_with_pcas(pair["robust"], pcas, bX, blast))
        boundary_stress_y.append(by)
        boundary_stress_last.append(blast)

        fold_clean_base = evaluate_multivariate(bundle.y_abs[test_idx], pred_base, bundle.last_values[test_idx])
        fold_clean_rob = evaluate_multivariate(bundle.y_abs[test_idx], pred_rob, bundle.last_values[test_idx])
        print(f"  clean baseline: {_short(fold_clean_base)}")
        print(f"  clean robust  : {_short(fold_clean_rob)}")
        print(f"  fold seconds={time.perf_counter()-fold_started:.1f}")

    if np.any(~np.isfinite(clean_oof_base)) or np.any(~np.isfinite(clean_oof_rob)):
        raise RuntimeError("LOSO OOF prediction incomplete")

    loso_clean_base_r = evaluate_multivariate(bundle.y_abs, clean_oof_base, bundle.last_values)
    loso_clean_rob_r = evaluate_multivariate(bundle.y_abs, clean_oof_rob, bundle.last_values)

    stress_y = np.concatenate(stress_oof_y, axis=0)
    stress_last = np.concatenate(stress_oof_last, axis=0)
    stress_base_all = np.concatenate(stress_oof_base, axis=0)
    stress_rob_all = np.concatenate(stress_oof_rob, axis=0)
    loso_stress_base_r = evaluate_multivariate(stress_y, stress_base_all, stress_last)
    loso_stress_rob_r = evaluate_multivariate(stress_y, stress_rob_all, stress_last)

    boundary_idx_all = np.asarray(boundary_idx_all, dtype=np.int64)
    b_clean_base_r = evaluate_multivariate(
        bundle.y_abs[boundary_idx_all], clean_oof_base[boundary_idx_all], bundle.last_values[boundary_idx_all]
    )
    b_clean_rob_r = evaluate_multivariate(
        bundle.y_abs[boundary_idx_all], clean_oof_rob[boundary_idx_all], bundle.last_values[boundary_idx_all]
    )
    b_stress_y = np.concatenate(boundary_stress_y, axis=0)
    b_stress_last = np.concatenate(boundary_stress_last, axis=0)
    b_stress_base = np.concatenate(boundary_stress_base, axis=0)
    b_stress_rob = np.concatenate(boundary_stress_rob, axis=0)
    b_stress_base_r = evaluate_multivariate(b_stress_y, b_stress_base, b_stress_last)
    b_stress_rob_r = evaluate_multivariate(b_stress_y, b_stress_rob, b_stress_last)

    print("\n" + "=" * 124)
    print("V14 dual-validation conclusion")
    print("=" * 124)
    print(f"temporal clean baseline : {_short(t_clean_base_r)}")
    print(f"temporal clean robust   : {_short(t_clean_rob_r)}")
    print(f"temporal stress baseline: {_short(t_stress_base_r)}")
    print(f"temporal stress robust  : {_short(t_stress_rob_r)}")
    print(f"LOSO clean baseline     : {_short(loso_clean_base_r)}")
    print(f"LOSO clean robust       : {_short(loso_clean_rob_r)}")
    print(f"LOSO stress baseline    : {_short(loso_stress_base_r)}")
    print(f"LOSO stress robust      : {_short(loso_stress_rob_r)}")
    print(f"boundary clean baseline : {_short(b_clean_base_r)}")
    print(f"boundary clean robust   : {_short(b_clean_rob_r)}")
    print(f"boundary stress baseline: {_short(b_stress_base_r)}")
    print(f"boundary stress robust  : {_short(b_stress_rob_r)}")

    tc0 = t_clean_base_r["mean"]["flat_rmse"]
    tc1 = t_clean_rob_r["mean"]["flat_rmse"]
    ts0 = t_stress_base_r["mean"]["flat_rmse"]
    ts1 = t_stress_rob_r["mean"]["flat_rmse"]
    lc0 = loso_clean_base_r["mean"]["flat_rmse"]
    lc1 = loso_clean_rob_r["mean"]["flat_rmse"]
    ls0 = loso_stress_base_r["mean"]["flat_rmse"]
    ls1 = loso_stress_rob_r["mean"]["flat_rmse"]
    bc0 = b_clean_base_r["mean"]["flat_rmse"]
    bc1 = b_clean_rob_r["mean"]["flat_rmse"]
    bs0 = b_stress_base_r["mean"]["flat_rmse"]
    bs1 = b_stress_rob_r["mean"]["flat_rmse"]

    temporal_clean_ok = tc1 <= tc0 * 1.01
    temporal_stress_ok = ts1 <= ts0 * 0.985
    loso_clean_ok = lc1 <= lc0 * 1.02
    loso_stress_ok = ls1 <= ls0 * 0.985
    boundary_clean_ok = bc1 <= bc0 * 1.02
    boundary_stress_ok = bs1 <= bs0 * 0.985

    print("\nGates:")
    print(f"  temporal clean gate : {temporal_clean_ok} ({(tc1/tc0-1)*100:+.2f}%)")
    print(f"  temporal stress gate: {temporal_stress_ok} ({(ts1/ts0-1)*100:+.2f}%)")
    print(f"  LOSO clean gate     : {loso_clean_ok} ({(lc1/lc0-1)*100:+.2f}%)")
    print(f"  LOSO stress gate    : {loso_stress_ok} ({(ls1/ls0-1)*100:+.2f}%)")
    print(f"  boundary clean gate : {boundary_clean_ok} ({(bc1/bc0-1)*100:+.2f}%)")
    print(f"  boundary stress gate: {boundary_stress_ok} ({(bs1/bs0-1)*100:+.2f}%)")

    passed = all(
        [
            temporal_clean_ok,
            temporal_stress_ok,
            loso_clean_ok,
            loso_stress_ok,
            boundary_clean_ok,
            boundary_stress_ok,
        ]
    )

    np.savez_compressed(
        ARTIFACT_PATH,
        temporal_clean_base=clean_base.astype(np.float32),
        temporal_clean_robust=clean_rob.astype(np.float32),
        loso_clean_base=clean_oof_base.astype(np.float32),
        loso_clean_robust=clean_oof_rob.astype(np.float32),
        boundary_idx=boundary_idx_all,
        passed=np.asarray(passed),
    )
    print(f"artifact          : {ARTIFACT_PATH}")
    print(f"LOSO total seconds: {time.perf_counter()-started:.1f}")
    if passed:
        print("PASS V14 ROBUST DUAL GATE：说明输入增强值得进入正式候选训练。")
    else:
        print("REJECT V14 ROBUST DUAL GATE：不要上线，继续保留当前官网 V8。")
    print("V14 仅诊断，不修改线上 V8。")


if __name__ == "__main__":
    main()
