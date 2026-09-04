import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import HORIZON, LOOKBACK, MODEL_DIR, TARGET_COLUMNS, XGB_DEVICE, XGB_PARAMS
from find_best_weight import (
    evaluate_multivariate,
    print_report,
    safe_corr,
    variable_metrics,
)
from src.dataset_builder import (
    load_all_data,
    sample_weights,
    temporal_train_val_indices,
)
from src.trajectory_fusion import endpoint_zero_highpass


V8_TEMPORAL_PATH = MODEL_DIR / "val_pred_candidate_v8.npz"
V12_LOSO_PATH = MODEL_DIR / "loso_pca_v12.npz"
ARTIFACT_PATH = MODEL_DIR / "v13_highfreq_pca_diagnostic.npz"

RAW_DIM = LOOKBACK * len(TARGET_COLUMNS)
RECENT = 48
HF_WINDOW = 9
HF_COMPONENTS = 6
GAIN_GRID = np.asarray(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00],
    dtype=np.float64,
)
EPS = 1e-9

# 只做“未知 sequence 可泛化”的高频残差，不再使用 sequence id/template。
# 模型输出仅 6*6=36 维 PCA score，并且 target 按历史差分尺度归一化。


def _trend_score(report):
    m = report["mean"]
    diff_score = float(np.clip((m["diff_corr"] + 1.0) / 2.0, 0.0, 1.0))
    return float(
        0.45 * diff_score
        + 0.25 * np.clip(m["peak_f1"], 0.0, 1.0)
        + 0.20 * np.clip(m["volatility_fit"], 0.0, 1.0)
        + 0.10 * np.clip(m["direction_accuracy"], 0.0, 1.0)
    )


def _params():
    params = XGB_PARAMS.copy()
    params.update(
        {
            "n_estimators": int(os.getenv("V13_ESTIMATORS", "340")),
            "learning_rate": 0.030,
            "max_depth": 3,
            "min_child_weight": 8.0,
            "subsample": 0.86,
            "colsample_bytree": 0.72,
            "reg_alpha": 0.30,
            "reg_lambda": 2.5,
            "verbosity": 0,
        }
    )
    return params


def _raw_windows(bundle):
    X = np.asarray(bundle.X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < RAW_DIM:
        raise ValueError(f"bundle.X shape 异常: {X.shape}")
    return X[:, :RAW_DIM].reshape(-1, LOOKBACK, len(TARGET_COLUMNS))


def _history_scale(raw):
    recent = np.asarray(raw, dtype=np.float64)[:, -RECENT:, :]
    d = np.diff(recent, axis=1)
    std = np.std(d, axis=1)
    med = np.median(d, axis=1)
    mad = np.median(np.abs(d - med[:, None, :]), axis=1)
    robust = 1.4826 * mad
    scale = np.maximum(std, robust)

    # 防止近乎静止窗口的 scale 过小导致归一化爆炸。
    level = np.maximum(np.median(np.abs(recent), axis=1), 1.0)
    floor = np.maximum(1e-6, 1e-5 * level)
    return np.maximum(scale, floor)


def _dynamic_features(raw):
    """
    面向未知 sequence 的高频特征：去绝对水平、按各自历史波动归一化。
    避免模型靠 sequence 的绝对量纲/均值记忆类别。
    """
    recent = np.asarray(raw, dtype=np.float64)[:, -RECENT:, :]
    scale = _history_scale(raw)
    last = recent[:, -1:, :]

    level_norm = (recent - last) / scale[:, None, :]
    diff_norm = np.diff(recent, axis=1) / scale[:, None, :]

    blocks = [
        level_norm.reshape(len(raw), -1),
        diff_norm.reshape(len(raw), -1),
    ]

    for width in (12, 24, 48):
        seg = recent[:, -width:, :]
        seg_diff = np.diff(seg, axis=1)
        mean = np.mean(seg, axis=1)
        std = np.std(seg, axis=1)
        span = np.max(seg, axis=1) - np.min(seg, axis=1)
        diff_std = np.std(seg_diff, axis=1)
        last_minus_mean = seg[:, -1, :] - mean

        x = np.arange(width, dtype=np.float64)
        xc = x - x.mean()
        denom = float(np.sum(xc ** 2)) + EPS
        slope = np.sum((seg - mean[:, None, :]) * xc[None, :, None], axis=1) / denom

        blocks.extend(
            [
                std / scale,
                span / scale,
                diff_std / scale,
                last_minus_mean / scale,
                slope / scale,
            ]
        )

    feat = np.concatenate(blocks, axis=1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0), scale


def _highfreq_target(y_abs, hist_scale):
    y_abs = np.asarray(y_abs, dtype=np.float64)
    hist_scale = np.asarray(hist_scale, dtype=np.float64)
    out = np.empty_like(y_abs, dtype=np.float64)
    for j in range(len(TARGET_COLUMNS)):
        hp = endpoint_zero_highpass(y_abs[:, :, j], HF_WINDOW)
        out[:, :, j] = hp / hist_scale[:, None, j]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_pcas(norm_hf, train_idx):
    pcas = []
    explained = []
    caps = []
    k = min(HF_COMPONENTS, len(train_idx), HORIZON)

    for j, col in enumerate(TARGET_COLUMNS):
        block = norm_hf[train_idx, :, j]
        pca = PCA(n_components=k, random_state=42)
        pca.fit(block)
        pcas.append(pca)
        explained.append(float(np.sum(pca.explained_variance_ratio_)))
        cap = float(np.quantile(np.abs(block), 0.995))
        caps.append(max(0.50, min(cap, 12.0)))
        print(
            f"    HF-PCA {col:16s}: comp={k} explained={explained[-1]:.4f} "
            f"norm_cap={caps[-1]:.3f}"
        )
    return pcas, np.asarray(caps, dtype=np.float64), explained


def _encode(norm_hf, pcas):
    return np.concatenate(
        [pca.transform(norm_hf[:, :, j]) for j, pca in enumerate(pcas)],
        axis=1,
    )


def _decode(scores, pcas, caps, hist_scale):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    out_norm = np.empty(
        (len(scores), HORIZON, len(TARGET_COLUMNS)), dtype=np.float64
    )
    offset = 0
    for j, pca in enumerate(pcas):
        k = int(pca.n_components_)
        block = pca.inverse_transform(scores[:, offset:offset + k])
        block = np.clip(block, -float(caps[j]), float(caps[j]))
        out_norm[:, :, j] = block
        offset += k

    if offset != scores.shape[1]:
        raise RuntimeError(
            f"HF score 维度不一致: used={offset}, actual={scores.shape[1]}"
        )
    return out_norm * np.asarray(hist_scale, dtype=np.float64)[:, None, :]


def _fit_predict(features, norm_hf, hist_scale, train_idx, test_idx, weights):
    pcas, caps, explained = _fit_pcas(norm_hf, train_idx)
    scores = _encode(norm_hf, pcas)

    sx = StandardScaler()
    sy = StandardScaler()
    X_train = sx.fit_transform(features[train_idx])
    X_test = sx.transform(features[test_idx])
    y_train = sy.fit_transform(scores[train_idx])

    model = XGBRegressor(**_params())
    fit_weights = np.sqrt(np.asarray(weights[train_idx], dtype=np.float64))
    model.fit(X_train, y_train, sample_weight=fit_weights)

    pred_scaled = np.asarray(model.predict(X_test))
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(len(test_idx), -1)
    pred_scores = sy.inverse_transform(pred_scaled)
    pred_hp = _decode(pred_scores, pcas, caps, hist_scale[test_idx])
    return pred_hp, explained


def _short(report):
    m = report["mean"]
    return (
        f"RMSE={m['flat_rmse']:.4f} meanRMSE={m['rmse']:.4f} "
        f"Diff={m['diff_corr']:.4f} Peak={m['peak_f1']:.4f} "
        f"Vol={m['volatility_fit']:.4f} Trend={_trend_score(report):.4f} "
        f"Proxy={m['proxy_loss']:.4f}"
    )


def _correlations(true_hp, pred_hp):
    vals = []
    for j in range(len(TARGET_COLUMNS)):
        vals.append(safe_corr(true_hp[:, :, j], pred_hp[:, :, j]))
    return np.asarray(vals, dtype=np.float64)


def _search_temporal_gains(y_true, last, base_pred, pred_hp):
    gains = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)

    print("\n" + "=" * 124)
    print("V13 temporal gain search：只在 temporal validation 选择 gain；LOSO 绝不参与调参。")
    print("=" * 124)

    for j, col in enumerate(TARGET_COLUMNS):
        yt = y_true[:, :, j]
        anchor = last[:, j]
        ref = base_pred[:, :, j]
        hp = pred_hp[:, :, j]
        ref_m = variable_metrics(yt, ref, anchor)
        rmse_cap = ref_m["rmse"] * 1.005
        diff_floor = ref_m["diff_corr"] - 0.0005

        def loss(m):
            rmse_ratio = m["rmse"] / max(ref_m["rmse"], EPS)
            diff_loss = (1.0 - m["diff_corr"]) / 2.0
            peak_loss = 1.0 - m["peak_f1"]
            vol_loss = 1.0 - m["volatility_fit"]
            return (
                0.35 * rmse_ratio
                + 0.25 * diff_loss
                + 0.20 * peak_loss
                + 0.20 * vol_loss
            )

        best = (loss(ref_m), ref_m["rmse"], 0.0, ref_m)
        for gain in GAIN_GRID:
            pred = ref + float(gain) * hp
            m = variable_metrics(yt, pred, anchor)
            if m["rmse"] > rmse_cap + EPS:
                continue
            if m["diff_corr"] < diff_floor:
                continue
            cand = (loss(m), m["rmse"], float(gain), m)
            if cand[:3] < best[:3]:
                best = cand

        _, _, gain, m = best
        gains[j] = gain
        print(
            f"{col:16s} GAIN={gain:.2f} RMSE={m['rmse']:.4f} "
            f"Diff={m['diff_corr']:.4f} Peak={m['peak_f1']:.4f} "
            f"Vol={m['volatility_fit']:.4f} Dir={m['direction_accuracy']:.4f}"
        )

    return gains


def main():
    print("=" * 124)
    print("V13 Cross-Sequence High-Frequency PCA Residual Diagnostic")
    print("低频继续由 V8/V12 PCA 负责；高频模型只学习去水平、按历史波动归一化后的局部残差。")
    print("gain 只用 temporal validation 搜索，再原样检查 LOSO，避免用 LOSO 调参。")
    print("本脚本不修改线上 V8、API、callback 或 ensemble_config.pkl。")
    print("=" * 124)
    print(
        f"device={XGB_DEVICE} estimators={_params()['n_estimators']} "
        f"HF_WINDOW={HF_WINDOW} components/var={HF_COMPONENTS}"
    )

    if not V8_TEMPORAL_PATH.exists():
        raise FileNotFoundError(f"缺少 {V8_TEMPORAL_PATH}")
    if not V12_LOSO_PATH.exists():
        raise FileNotFoundError(f"缺少 {V12_LOSO_PATH}，请先运行 V12")

    bundle = load_all_data()
    weights = sample_weights(bundle)
    raw = _raw_windows(bundle)
    dyn_features, hist_scale = _dynamic_features(raw)
    norm_hf = _highfreq_target(bundle.y_abs, hist_scale)
    true_hp = norm_hf * hist_scale[:, None, :]

    train_idx, val_idx = temporal_train_val_indices(bundle)
    with np.load(V8_TEMPORAL_PATH, allow_pickle=True) as data:
        v8_val = data["pred_abs"].astype(np.float64)
    if len(v8_val) != len(val_idx):
        raise RuntimeError(f"V8 val 数量不一致: {len(v8_val)} vs {len(val_idx)}")

    started = time.perf_counter()

    print("\n[A] Temporal high-frequency model")
    temporal_hp, temporal_explained = _fit_predict(
        dyn_features,
        norm_hf,
        hist_scale,
        train_idx,
        val_idx,
        weights,
    )
    temporal_corr = _correlations(true_hp[val_idx], temporal_hp)
    print("  HF corr by variable:", np.round(temporal_corr, 4).tolist())
    print(f"  HF corr mean       : {float(np.mean(temporal_corr)):.4f}")

    gains = _search_temporal_gains(
        bundle.y_abs[val_idx].astype(np.float64),
        bundle.last_values[val_idx].astype(np.float64),
        v8_val,
        temporal_hp,
    )

    temporal_candidate = v8_val + temporal_hp * gains.reshape(1, 1, -1)
    temporal_ref_report = evaluate_multivariate(
        bundle.y_abs[val_idx], v8_val, bundle.last_values[val_idx]
    )
    temporal_cand_report = evaluate_multivariate(
        bundle.y_abs[val_idx], temporal_candidate, bundle.last_values[val_idx]
    )
    print_report("V13 temporal V8 reference", temporal_ref_report)
    print_report("V13 temporal V8 + predicted HF", temporal_cand_report)

    print("\n[B] LOSO high-frequency OOF")
    sequences = sorted(np.unique(bundle.sequence_names).tolist())
    loso_hp = np.full_like(bundle.y_abs, np.nan, dtype=np.float64)
    explained_by_fold = []

    for fold, heldout in enumerate(sequences):
        test_idx = np.where(bundle.sequence_names == heldout)[0]
        fold_train = np.where(bundle.sequence_names != heldout)[0]
        fold_started = time.perf_counter()
        print("\n" + "-" * 124)
        print(
            f"[HF FOLD {fold + 1}/{len(sequences)}] heldout={heldout} "
            f"train={len(fold_train)} test={len(test_idx)}"
        )
        pred_hp, explained = _fit_predict(
            dyn_features,
            norm_hf,
            hist_scale,
            fold_train,
            test_idx,
            weights,
        )
        loso_hp[test_idx] = pred_hp
        explained_by_fold.append(explained)
        corr = _correlations(true_hp[test_idx], pred_hp)
        print(
            f"  HF corr mean={float(np.mean(corr)):.4f} "
            f"by_var={np.round(corr, 4).tolist()}"
        )
        print(f"  fold seconds={time.perf_counter() - fold_started:.1f}")

    if np.any(~np.isfinite(loso_hp)):
        raise RuntimeError("LOSO HF prediction incomplete")

    with np.load(V12_LOSO_PATH, allow_pickle=True) as data:
        loso_base = data["pred_abs"].astype(np.float64)
        loso_y = data["y_abs"].astype(np.float64)
        loso_last = data["last_values"].astype(np.float64)
        boundary_idx = data["boundary_idx"].astype(np.int64)

    if loso_base.shape != bundle.y_abs.shape:
        raise RuntimeError(f"V12 LOSO shape 异常: {loso_base.shape}")

    loso_candidate = loso_base + loso_hp * gains.reshape(1, 1, -1)
    loso_ref_report = evaluate_multivariate(loso_y, loso_base, loso_last)
    loso_cand_report = evaluate_multivariate(loso_y, loso_candidate, loso_last)
    boundary_ref_report = evaluate_multivariate(
        loso_y[boundary_idx], loso_base[boundary_idx], loso_last[boundary_idx]
    )
    boundary_cand_report = evaluate_multivariate(
        loso_y[boundary_idx], loso_candidate[boundary_idx], loso_last[boundary_idx]
    )

    print_report("V13 LOSO V12 low-frequency reference", loso_ref_report)
    print_report("V13 LOSO V12 + predicted HF", loso_cand_report)
    print_report("V13 LOSO boundary reference", boundary_ref_report)
    print_report("V13 LOSO boundary + predicted HF", boundary_cand_report)

    tr = temporal_ref_report["mean"]
    tc = temporal_cand_report["mean"]
    lr = loso_ref_report["mean"]
    lc = loso_cand_report["mean"]
    br = boundary_ref_report["mean"]
    bc = boundary_cand_report["mean"]

    temporal_ok = bool(
        tc["flat_rmse"] <= tr["flat_rmse"] * 1.005
        and tc["diff_corr"] >= tr["diff_corr"] - 0.0002
        and _trend_score(temporal_cand_report) >= _trend_score(temporal_ref_report) + 0.002
        and tc["peak_f1"] >= tr["peak_f1"] + 0.003
        and tc["volatility_fit"] >= tr["volatility_fit"] + 0.004
    )
    loso_ok = bool(
        lc["flat_rmse"] <= lr["flat_rmse"] * 1.01
        and lc["diff_corr"] >= lr["diff_corr"] - 0.002
        and _trend_score(loso_cand_report) >= _trend_score(loso_ref_report) + 0.003
        and lc["peak_f1"] >= lr["peak_f1"] + 0.002
        and lc["volatility_fit"] >= lr["volatility_fit"] + 0.008
    )
    boundary_ok = bool(
        bc["flat_rmse"] <= br["flat_rmse"] * 1.02
        and _trend_score(boundary_cand_report) >= _trend_score(boundary_ref_report)
    )
    passed = bool(temporal_ok and loso_ok and boundary_ok and np.any(gains > 0.0))

    np.savez_compressed(
        ARTIFACT_PATH,
        temporal_pred_hp=temporal_hp.astype(np.float32),
        temporal_candidate=temporal_candidate.astype(np.float32),
        loso_pred_hp=loso_hp.astype(np.float32),
        loso_candidate=loso_candidate.astype(np.float32),
        gains=gains.astype(np.float64),
        temporal_hf_corr=temporal_corr.astype(np.float64),
        temporal_explained=np.asarray(temporal_explained, dtype=np.float64),
        loso_explained=np.asarray(explained_by_fold, dtype=np.float64),
        boundary_idx=boundary_idx,
        passed=np.asarray(passed),
    )

    print("\n" + "=" * 124)
    print("V13 dual-validation conclusion")
    print("=" * 124)
    print(f"gains             : {gains.tolist()}")
    print(f"temporal HF corr  : {float(np.mean(temporal_corr)):.4f}")
    print(f"temporal reference: {_short(temporal_ref_report)}")
    print(f"temporal candidate: {_short(temporal_cand_report)}")
    print(f"LOSO reference    : {_short(loso_ref_report)}")
    print(f"LOSO candidate    : {_short(loso_cand_report)}")
    print(f"boundary reference: {_short(boundary_ref_report)}")
    print(f"boundary candidate: {_short(boundary_cand_report)}")
    print(f"temporal gate     : {temporal_ok}")
    print(f"LOSO gate         : {loso_ok}")
    print(f"boundary gate     : {boundary_ok}")
    print(f"artifact          : {ARTIFACT_PATH}")
    print(f"total seconds     : {time.perf_counter() - started:.1f}")
    if passed:
        print("PASS V13 DUAL GATE：高频残差在 temporal + LOSO 同时有效，才值得继续做在线版本。")
    else:
        print("REJECT V13 DUAL GATE：不要上线；保留当前官网 V8。")


if __name__ == "__main__":
    main()
