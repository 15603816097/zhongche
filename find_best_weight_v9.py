import pickle

import numpy as np

from config import MODEL_DIR, TARGET_COLUMNS
from find_best_weight import (
    evaluate_multivariate,
    load_validation_arrays,
    print_report,
    variable_metrics,
)
from src.dataset_builder import load_all_data, temporal_train_val_indices
from src.template_shape import (
    build_template_bank,
    predict_template_shapes_from_features,
    sequence_match_accuracy,
)


V8_PRED_PATH = MODEL_DIR / "val_pred_candidate_v8.npz"
V9_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v9.pkl"
V9_PRED_PATH = MODEL_DIR / "val_pred_candidate_v9.npz"
V9_BANK_PATH = MODEL_DIR / "template_shape_bank_v9.pkl"

GAIN_GRID = np.asarray(
    [-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65],
    dtype=np.float64,
)

# 只有一天一次官网机会时，V9 必须非常保守：保 Accuracy，换取明确 Trend 增益。
MAX_VARIABLE_RMSE_DEGRADATION = 0.010
MAX_GLOBAL_RMSE_DEGRADATION = 0.006
MAX_GLOBAL_DIFF_DROP = 0.0002
MIN_GLOBAL_TREND_GAIN = 0.0060
MIN_GLOBAL_PEAK_GAIN = 0.0040
MIN_GLOBAL_VOL_GAIN = 0.0060
MAX_PROXY_DEGRADATION = 0.0015
EPS = 1e-12


def _trend_score(report):
    m = report["mean"]
    diff_score = float(np.clip((m["diff_corr"] + 1.0) / 2.0, 0.0, 1.0))
    return float(
        0.45 * diff_score
        + 0.25 * np.clip(m["peak_f1"], 0.0, 1.0)
        + 0.20 * np.clip(m["volatility_fit"], 0.0, 1.0)
        + 0.10 * np.clip(m["direction_accuracy"], 0.0, 1.0)
    )


def _variable_loss(metrics):
    """40% 数值 + 60% 趋势；RMSE 另有硬安全闸。越低越好。"""
    rmse_ratio = metrics["rmse"] / max(metrics["persistence_rmse"], EPS)
    mae_ratio = metrics["mae"] / max(metrics["persistence_mae"], EPS)
    diff_loss = float(np.clip((1.0 - metrics["diff_corr"]) / 2.0, 0.0, 1.0))
    peak_loss = 1.0 - float(np.clip(metrics["peak_f1"], 0.0, 1.0))
    vol_loss = 1.0 - float(np.clip(metrics["volatility_fit"], 0.0, 1.0))
    dir_loss = 1.0 - float(np.clip(metrics["direction_accuracy"], 0.0, 1.0))
    return float(
        0.25 * rmse_ratio
        + 0.15 * mae_ratio
        + 0.25 * diff_loss
        + 0.15 * peak_loss
        + 0.15 * vol_loss
        + 0.05 * dir_loss
    )


def _load_v8_pred():
    if not V8_PRED_PATH.exists():
        raise FileNotFoundError(
            f"缺少 {V8_PRED_PATH}，请先保留 V8 离线候选产物"
        )
    data = np.load(V8_PRED_PATH, allow_pickle=True)
    return data["pred_abs"].astype(np.float64)


def main():
    y_true, last_values, _, _, _ = load_validation_arrays()
    v8_pred = _load_v8_pred()
    if v8_pred.shape != y_true.shape:
        raise ValueError(f"V8 pred shape 不一致: {v8_pred.shape} vs {y_true.shape}")

    bundle = load_all_data()
    train_idx, val_idx = temporal_train_val_indices(bundle)

    if len(val_idx) != len(y_true):
        raise RuntimeError(
            f"验证样本数量不一致: bundle val={len(val_idx)} vs pred={len(y_true)}"
        )
    if not np.allclose(bundle.y_abs[val_idx], y_true, rtol=1e-5, atol=1e-5):
        raise RuntimeError("bundle.y_abs[val_idx] 与 val_pred_lgb.npz 顺序不一致")
    if not np.allclose(bundle.last_values[val_idx], last_values, rtol=1e-5, atol=1e-5):
        raise RuntimeError("bundle.last_values[val_idx] 与验证数组顺序不一致")

    print("=" * 120)
    print("V9 Sequence Template Shape Fusion：从最初 55.97 的趋势能力中提取‘形状模板’")
    print("模板只用 train 段构建，不使用 val future；线上 V8 不会被修改。")
    print("=" * 120)

    bank = build_template_bank(bundle, train_idx)
    template_shape, match_weights, distances = predict_template_shapes_from_features(
        bundle.X[val_idx],
        bank,
    )
    match_acc = sequence_match_accuracy(
        match_weights,
        bundle.sequence_names[val_idx],
        bank,
    )

    print(f"sequence matcher accuracy: {match_acc:.4f}")
    print("template bank sequences:", bank["sequences"])
    print("template train counts  :", bank["sample_counts"])
    print(
        "mean max soft weight    : "
        f"{float(np.mean(np.max(match_weights, axis=1))):.4f}"
    )
    print(
        "mean min distance       : "
        f"{float(np.mean(np.min(distances, axis=1))):.4f}"
    )

    v8_report = evaluate_multivariate(y_true, v8_pred, last_values)
    print_report("V8 官方已验证 baseline（离线 validation）", v8_report)

    gains = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)

    print("\n" + "=" * 120)
    print("V9：每变量搜索 endpoint-zero template shape gain")
    print(
        f"单变量 RMSE 最多退化 {MAX_VARIABLE_RMSE_DEGRADATION*100:.1f}%；"
        "模板不修改 V8 的整体终点，只补局部峰谷/波动。"
    )
    print("=" * 120)

    for j, col in enumerate(TARGET_COLUMNS):
        yt = y_true[:, :, j]
        anchor = last_values[:, j]
        ref = v8_pred[:, :, j]
        shape = template_shape[:, :, j]

        ref_metrics = variable_metrics(yt, ref, anchor)
        rmse_cap = ref_metrics["rmse"] * (1.0 + MAX_VARIABLE_RMSE_DEGRADATION)

        best = (
            _variable_loss(ref_metrics),
            ref_metrics["rmse"],
            -ref_metrics["diff_corr"],
            -ref_metrics["peak_f1"],
            0.0,
            ref_metrics,
        )

        for gain in GAIN_GRID:
            pred = ref + float(gain) * shape
            metrics = variable_metrics(yt, pred, anchor)
            if metrics["rmse"] > rmse_cap + EPS:
                continue

            # 单变量不允许明显牺牲 DiffCorr，避免再次出现 V7 的问题。
            if metrics["diff_corr"] < ref_metrics["diff_corr"] - 0.0010:
                continue

            candidate = (
                _variable_loss(metrics),
                metrics["rmse"],
                -metrics["diff_corr"],
                -metrics["peak_f1"],
                float(gain),
                metrics,
            )
            if candidate[:5] < best[:5]:
                best = candidate

        best_loss, best_rmse, _, _, best_gain, best_metrics = best
        gains[j] = best_gain

        print(
            f"{col:16s} GAIN={best_gain:+.2f} "
            f"RMSE={best_rmse:.4f} Diff={best_metrics['diff_corr']:.4f} "
            f"Peak={best_metrics['peak_f1']:.4f} "
            f"Vol={best_metrics['volatility_fit']:.4f} "
            f"Dir={best_metrics['direction_accuracy']:.4f} "
            f"Loss={best_loss:.4f}"
        )

    candidate_pred = v8_pred + template_shape * gains.reshape(1, 1, -1)
    candidate_report = evaluate_multivariate(y_true, candidate_pred, last_values)
    print_report("V9 template-shape candidate", candidate_report)

    ref = v8_report["mean"]
    cand = candidate_report["mean"]
    ref_rmse = float(ref["flat_rmse"])
    cand_rmse = float(cand["flat_rmse"])
    ref_trend = _trend_score(v8_report)
    cand_trend = _trend_score(candidate_report)
    ref_proxy = float(ref["proxy_loss"])
    cand_proxy = float(cand["proxy_loss"])

    rmse_ok = cand_rmse <= ref_rmse * (1.0 + MAX_GLOBAL_RMSE_DEGRADATION)
    diff_ok = cand["diff_corr"] >= ref["diff_corr"] - MAX_GLOBAL_DIFF_DROP
    trend_ok = cand_trend >= ref_trend + MIN_GLOBAL_TREND_GAIN
    peak_ok = cand["peak_f1"] >= ref["peak_f1"] + MIN_GLOBAL_PEAK_GAIN
    vol_ok = cand["volatility_fit"] >= ref["volatility_fit"] + MIN_GLOBAL_VOL_GAIN
    proxy_ok = cand_proxy <= ref_proxy + MAX_PROXY_DEGRADATION
    any_gain = bool(np.any(np.abs(gains) > 1e-12))

    passed = bool(
        rmse_ok and diff_ok and trend_ok and peak_ok and vol_ok and proxy_ok and any_gain
    )

    print("\n" + "=" * 120)
    print("V9 候选结论")
    print("=" * 120)
    print(f"matcher accuracy : {match_acc:.4f}")
    print(f"V8 flat RMSE     : {ref_rmse:.6f}")
    print(
        f"V9 flat RMSE     : {cand_rmse:.6f} "
        f"({(cand_rmse/ref_rmse-1.0)*100:+.2f}%)"
    )
    print(f"V8 proxy loss    : {ref_proxy:.6f}")
    print(
        f"V9 proxy loss    : {cand_proxy:.6f} "
        f"({(cand_proxy/ref_proxy-1.0)*100:+.2f}%)"
    )
    print(f"V8 trendScore    : {ref_trend:.6f}")
    print(
        f"V9 trendScore    : {cand_trend:.6f} "
        f"({(cand_trend/ref_trend-1.0)*100:+.2f}%)"
    )
    print(f"V9 DiffCorr      : {cand['diff_corr']:.6f}")
    print(f"V9 PeakF1        : {cand['peak_f1']:.6f}")
    print(f"V9 VolFit        : {cand['volatility_fit']:.6f}")
    print(f"V9 DirAcc        : {cand['direction_accuracy']:.6f}")
    print(f"shape gains      : {gains.tolist()}")

    np.savez_compressed(
        V9_PRED_PATH,
        pred_abs=candidate_pred.astype(np.float32),
        template_shape=template_shape.astype(np.float32),
        shape_gains=gains,
        match_weights=match_weights.astype(np.float32),
        val_sequence_names=bundle.sequence_names[val_idx],
        val_starts=bundle.starts[val_idx],
    )

    with open(V9_BANK_PATH, "wb") as f:
        pickle.dump(bank, f)

    candidate_config = {
        "version": 9,
        "candidate_only": True,
        "base_version": 8,
        "trajectory_model": "v8_plus_causal_sequence_template_shape_v1",
        "template_shape_gains": gains.tolist(),
        "template_match_temperature": float(bank["match_temperature"]),
        "template_sequences": list(bank["sequences"]),
        "template_match_accuracy": match_acc,
        "validation_rmse": cand_rmse,
        "validation_proxy_loss": cand_proxy,
        "validation_diff_corr": float(cand["diff_corr"]),
        "validation_peak_f1": float(cand["peak_f1"]),
        "validation_volatility_fit": float(cand["volatility_fit"]),
        "validation_direction_accuracy": float(cand["direction_accuracy"]),
        "validation_trend_score": cand_trend,
        "candidate_passed_local_gate": passed,
    }
    with open(V9_CONFIG_PATH, "wb") as f:
        pickle.dump(candidate_config, f)

    if passed:
        print("PASS V9 LOCAL GATE：今天只保留离线候选，明天再决定是否接入 API。")
    else:
        print("REJECT V9 LOCAL GATE：不要上线，继续用当前官网已验证 V8。")
    print(f"candidate config: {V9_CONFIG_PATH}")
    print(f"template bank   : {V9_BANK_PATH}")
    print("当前 models/ensemble_config.pkl 完全不会被 V9 修改。")


if __name__ == "__main__":
    main()
