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
V10_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v10.pkl"
V10_PRED_PATH = MODEL_DIR / "val_pred_candidate_v10.npz"
V10_BANK_PATH = MODEL_DIR / "template_shape_bank_v10.pkl"

TEMPERATURE_GRID = (0.15, 0.20, 0.25, 0.35, 0.50, 0.75)
GAIN_GRID = np.asarray(
    [-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15,
     0.20, 0.30, 0.40, 0.50, 0.65, 0.80],
    dtype=np.float64,
)
# -1 表示不做置信度门控，完整注入；其他值表示 confidence threshold。
CONF_THRESHOLD_GRID = (-1.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
CONF_POWER_GRID = (1.0, 2.0)

# V10 目标：修复 V9 的两个问题：
# 1) mean max soft weight 只有约 0.52，低置信样本也被固定 gain 注入；
# 2) V9 Peak/Vol 提升，但 RMSE 与 DiffCorr 轻微变差。
MAX_VARIABLE_RMSE_DEGRADATION = 0.008
MAX_VARIABLE_DIFF_DROP = 0.00035
MAX_VARIABLE_PEAK_DROP = 0.002
MAX_VARIABLE_VOL_DROP = 0.004

MAX_GLOBAL_RMSE_DEGRADATION = 0.004
MAX_GLOBAL_DIFF_DROP = 0.00010
MIN_GLOBAL_TREND_GAIN = 0.0040
MIN_GLOBAL_PEAK_GAIN = 0.0060
MIN_GLOBAL_VOL_GAIN = 0.0060
MAX_PROXY_DEGRADATION = 0.0005
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
    """35% Accuracy + 65% Trend；硬门槛负责约束 RMSE / DiffCorr。"""
    rmse_ratio = metrics["rmse"] / max(metrics["persistence_rmse"], EPS)
    mae_ratio = metrics["mae"] / max(metrics["persistence_mae"], EPS)
    diff_loss = float(np.clip((1.0 - metrics["diff_corr"]) / 2.0, 0.0, 1.0))
    peak_loss = 1.0 - float(np.clip(metrics["peak_f1"], 0.0, 1.0))
    vol_loss = 1.0 - float(np.clip(metrics["volatility_fit"], 0.0, 1.0))
    dir_loss = 1.0 - float(np.clip(metrics["direction_accuracy"], 0.0, 1.0))
    return float(
        0.22 * rmse_ratio
        + 0.13 * mae_ratio
        + 0.28 * diff_loss
        + 0.16 * peak_loss
        + 0.16 * vol_loss
        + 0.05 * dir_loss
    )


def _load_v8_pred():
    if not V8_PRED_PATH.exists():
        raise FileNotFoundError(f"缺少 {V8_PRED_PATH}")
    data = np.load(V8_PRED_PATH, allow_pickle=True)
    return data["pred_abs"].astype(np.float64)


def _confidence_from_weights(weights: np.ndarray) -> np.ndarray:
    """
    把 soft sequence weights 转为 0~1 confidence。

    同时考虑：
      - top1 权重相对均匀分布的提升
      - top1-top2 margin
      - softmax entropy
    """
    w = np.asarray(weights, dtype=np.float64)
    n_seq = w.shape[1]
    sorted_w = np.sort(w, axis=1)
    top1 = sorted_w[:, -1]
    top2 = sorted_w[:, -2] if n_seq > 1 else np.zeros_like(top1)

    uniform = 1.0 / float(n_seq)
    top1_norm = np.clip((top1 - uniform) / max(1.0 - uniform, EPS), 0.0, 1.0)
    margin = np.clip(top1 - top2, 0.0, 1.0)

    entropy = -np.sum(w * np.log(np.maximum(w, EPS)), axis=1)
    entropy_conf = 1.0 - entropy / max(np.log(float(n_seq)), EPS)
    entropy_conf = np.clip(entropy_conf, 0.0, 1.0)

    confidence = 0.45 * top1_norm + 0.25 * margin + 0.30 * entropy_conf
    return np.clip(confidence, 0.0, 1.0)


def _confidence_gate(confidence: np.ndarray, threshold: float, power: float) -> np.ndarray:
    if threshold < 0.0:
        return np.ones_like(confidence, dtype=np.float64)
    threshold = float(np.clip(threshold, 0.0, 0.95))
    gate = np.clip(
        (confidence - threshold) / max(1.0 - threshold, EPS),
        0.0,
        1.0,
    )
    return gate ** float(power)


def _print_confidence_calibration(weights, true_names, bank, confidence):
    seqs = np.asarray(bank["sequences"], dtype=object)
    pred = seqs[np.argmax(weights, axis=1)]
    true = np.asarray(true_names, dtype=object)
    correct = pred == true

    print("confidence calibration:")
    bins = ((0.00, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.70), (0.70, 1.01))
    for lo, hi in bins:
        mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        print(
            f"  conf [{lo:.2f},{hi:.2f}) n={int(np.sum(mask)):3d} "
            f"match_acc={float(np.mean(correct[mask])):.4f} "
            f"mean_conf={float(np.mean(confidence[mask])):.4f}"
        )


def _search_one_temperature(
    temperature,
    bundle,
    val_idx,
    bank,
    y_true,
    last_values,
    v8_pred,
    v8_report,
):
    template_shape, match_weights, distances = predict_template_shapes_from_features(
        bundle.X[val_idx],
        bank,
        temperature=float(temperature),
    )
    confidence = _confidence_from_weights(match_weights)
    match_acc = sequence_match_accuracy(
        match_weights,
        bundle.sequence_names[val_idx],
        bank,
    )

    gains = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)
    thresholds = np.full(len(TARGET_COLUMNS), -1.0, dtype=np.float64)
    powers = np.ones(len(TARGET_COLUMNS), dtype=np.float64)
    mean_gates = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)

    for j, col in enumerate(TARGET_COLUMNS):
        yt = y_true[:, :, j]
        anchor = last_values[:, j]
        ref = v8_pred[:, :, j]
        shape = template_shape[:, :, j]
        ref_metrics = variable_metrics(yt, ref, anchor)

        rmse_cap = ref_metrics["rmse"] * (1.0 + MAX_VARIABLE_RMSE_DEGRADATION)
        diff_floor = ref_metrics["diff_corr"] - MAX_VARIABLE_DIFF_DROP
        peak_floor = max(0.0, ref_metrics["peak_f1"] - MAX_VARIABLE_PEAK_DROP)
        vol_floor = max(0.0, ref_metrics["volatility_fit"] - MAX_VARIABLE_VOL_DROP)

        best_key = (
            _variable_loss(ref_metrics),
            ref_metrics["rmse"],
            -ref_metrics["diff_corr"],
            -ref_metrics["peak_f1"],
            0.0,
            -1.0,
            1.0,
        )
        best_metrics = ref_metrics
        best_gate = np.zeros(len(confidence), dtype=np.float64)

        for threshold in CONF_THRESHOLD_GRID:
            for power in CONF_POWER_GRID:
                gate = _confidence_gate(confidence, threshold, power)
                gated_shape = shape * gate[:, None]

                for gain in GAIN_GRID:
                    if abs(float(gain)) <= EPS:
                        pred = ref
                    else:
                        pred = ref + float(gain) * gated_shape
                    metrics = variable_metrics(yt, pred, anchor)

                    if metrics["rmse"] > rmse_cap + EPS:
                        continue
                    if metrics["diff_corr"] < diff_floor:
                        continue
                    if metrics["peak_f1"] < peak_floor:
                        continue
                    if metrics["volatility_fit"] < vol_floor:
                        continue

                    key = (
                        _variable_loss(metrics),
                        metrics["rmse"],
                        -metrics["diff_corr"],
                        -metrics["peak_f1"],
                        abs(float(gain)),
                        float(threshold),
                        float(power),
                    )
                    if key < best_key:
                        best_key = key
                        best_metrics = metrics
                        best_gate = gate
                        gains[j] = float(gain)
                        thresholds[j] = float(threshold)
                        powers[j] = float(power)

        mean_gates[j] = float(np.mean(best_gate)) if abs(gains[j]) > EPS else 0.0

    candidate_pred = v8_pred.copy()
    gate_matrix = np.zeros((len(val_idx), len(TARGET_COLUMNS)), dtype=np.float64)
    for j in range(len(TARGET_COLUMNS)):
        gate = _confidence_gate(confidence, thresholds[j], powers[j])
        if abs(gains[j]) <= EPS:
            gate = np.zeros_like(gate)
        gate_matrix[:, j] = gate
        candidate_pred[:, :, j] += (
            gains[j] * gate[:, None] * template_shape[:, :, j]
        )

    report = evaluate_multivariate(y_true, candidate_pred, last_values)
    ref = v8_report["mean"]
    cand = report["mean"]
    ref_trend = _trend_score(v8_report)
    cand_trend = _trend_score(report)

    # 用 proxy 为主，兼顾趋势和 RMSE；这里只用于温度之间排序，不是通过门槛。
    selection_score = float(
        cand["proxy_loss"]
        + 0.35 * max(0.0, cand["flat_rmse"] / ref["flat_rmse"] - 1.0)
        - 0.40 * max(0.0, cand_trend - ref_trend)
    )

    return {
        "temperature": float(temperature),
        "template_shape": template_shape,
        "weights": match_weights,
        "distances": distances,
        "confidence": confidence,
        "match_acc": float(match_acc),
        "gains": gains.copy(),
        "thresholds": thresholds.copy(),
        "powers": powers.copy(),
        "mean_gates": mean_gates.copy(),
        "gate_matrix": gate_matrix,
        "pred": candidate_pred,
        "report": report,
        "trend": cand_trend,
        "selection_score": selection_score,
    }


def main():
    y_true, last_values, _, _, _ = load_validation_arrays()
    v8_pred = _load_v8_pred()
    if v8_pred.shape != y_true.shape:
        raise ValueError(f"V8 pred shape 不一致: {v8_pred.shape} vs {y_true.shape}")

    bundle = load_all_data()
    train_idx, val_idx = temporal_train_val_indices(bundle)
    if len(val_idx) != len(y_true):
        raise RuntimeError(f"bundle val={len(val_idx)} vs y_true={len(y_true)}")

    bank = build_template_bank(bundle, train_idx)
    v8_report = evaluate_multivariate(y_true, v8_pred, last_values)

    print("=" * 128)
    print("V10 Confidence-Gated Template Shape Fusion")
    print("V9 的模板路线保留，但低置信匹配样本自动弱化/关闭 shape 注入。")
    print("V8 仍是当前官网 baseline；本脚本绝不覆盖 ensemble_config.pkl。")
    print("=" * 128)
    print_report("V8 official baseline proxy", v8_report)

    results = []
    for temperature in TEMPERATURE_GRID:
        result = _search_one_temperature(
            temperature=temperature,
            bundle=bundle,
            val_idx=val_idx,
            bank=bank,
            y_true=y_true,
            last_values=last_values,
            v8_pred=v8_pred,
            v8_report=v8_report,
        )
        results.append(result)
        m = result["report"]["mean"]
        print(
            f"TEMP={temperature:.2f} match={result['match_acc']:.4f} "
            f"mean_conf={float(np.mean(result['confidence'])):.4f} "
            f"RMSE={m['flat_rmse']:.6f} Diff={m['diff_corr']:.6f} "
            f"Peak={m['peak_f1']:.6f} Vol={m['volatility_fit']:.6f} "
            f"Trend={result['trend']:.6f} Proxy={m['proxy_loss']:.6f}"
        )

    best = min(results, key=lambda x: x["selection_score"])
    candidate_report = best["report"]
    print("\n" + "=" * 128)
    print(f"V10 selected temperature = {best['temperature']:.2f}")
    print("=" * 128)
    _print_confidence_calibration(
        best["weights"],
        bundle.sequence_names[val_idx],
        bank,
        best["confidence"],
    )

    for j, col in enumerate(TARGET_COLUMNS):
        m = candidate_report["per_variable"][col]
        print(
            f"{col:16s} GAIN={best['gains'][j]:+.2f} "
            f"THR={best['thresholds'][j]:+.2f} POW={best['powers'][j]:.0f} "
            f"mean_gate={best['mean_gates'][j]:.3f} "
            f"RMSE={m['rmse']:.4f} Diff={m['diff_corr']:.4f} "
            f"Peak={m['peak_f1']:.4f} Vol={m['volatility_fit']:.4f}"
        )

    print_report("V10 confidence-gated template candidate", candidate_report)

    ref = v8_report["mean"]
    cand = candidate_report["mean"]
    ref_rmse = float(ref["flat_rmse"])
    cand_rmse = float(cand["flat_rmse"])
    ref_proxy = float(ref["proxy_loss"])
    cand_proxy = float(cand["proxy_loss"])
    ref_trend = _trend_score(v8_report)
    cand_trend = _trend_score(candidate_report)

    rmse_ok = cand_rmse <= ref_rmse * (1.0 + MAX_GLOBAL_RMSE_DEGRADATION)
    diff_ok = cand["diff_corr"] >= ref["diff_corr"] - MAX_GLOBAL_DIFF_DROP
    trend_ok = cand_trend >= ref_trend + MIN_GLOBAL_TREND_GAIN
    peak_ok = cand["peak_f1"] >= ref["peak_f1"] + MIN_GLOBAL_PEAK_GAIN
    vol_ok = cand["volatility_fit"] >= ref["volatility_fit"] + MIN_GLOBAL_VOL_GAIN
    proxy_ok = cand_proxy <= ref_proxy + MAX_PROXY_DEGRADATION
    any_gain = bool(np.any(np.abs(best["gains"]) > EPS))
    passed = bool(
        rmse_ok and diff_ok and trend_ok and peak_ok and vol_ok and proxy_ok and any_gain
    )

    print("\n" + "=" * 128)
    print("V10 candidate conclusion")
    print("=" * 128)
    print(f"temperature       : {best['temperature']:.2f}")
    print(f"matcher accuracy  : {best['match_acc']:.4f}")
    print(f"mean confidence   : {float(np.mean(best['confidence'])):.4f}")
    print(f"V8 flat RMSE      : {ref_rmse:.6f}")
    print(f"V10 flat RMSE     : {cand_rmse:.6f} ({(cand_rmse/ref_rmse-1)*100:+.2f}%)")
    print(f"V8 proxy loss     : {ref_proxy:.6f}")
    print(f"V10 proxy loss    : {cand_proxy:.6f} ({(cand_proxy/ref_proxy-1)*100:+.2f}%)")
    print(f"V8 trendScore     : {ref_trend:.6f}")
    print(f"V10 trendScore    : {cand_trend:.6f} ({(cand_trend/ref_trend-1)*100:+.2f}%)")
    print(f"V10 DiffCorr      : {cand['diff_corr']:.6f}")
    print(f"V10 PeakF1        : {cand['peak_f1']:.6f}")
    print(f"V10 VolFit        : {cand['volatility_fit']:.6f}")
    print(f"V10 DirAcc        : {cand['direction_accuracy']:.6f}")
    print(f"shape gains       : {best['gains'].tolist()}")
    print(f"conf thresholds   : {best['thresholds'].tolist()}")
    print(f"conf powers       : {best['powers'].tolist()}")
    print(f"mean gates        : {best['mean_gates'].tolist()}")

    np.savez_compressed(
        V10_PRED_PATH,
        pred_abs=best["pred"].astype(np.float32),
        template_shape=best["template_shape"].astype(np.float32),
        shape_gains=best["gains"],
        conf_thresholds=best["thresholds"],
        conf_powers=best["powers"],
        confidence=best["confidence"].astype(np.float32),
        gate_matrix=best["gate_matrix"].astype(np.float32),
        match_weights=best["weights"].astype(np.float32),
        val_sequence_names=bundle.sequence_names[val_idx],
        val_starts=bundle.starts[val_idx],
    )

    bank = dict(bank)
    bank["match_temperature"] = float(best["temperature"])
    with open(V10_BANK_PATH, "wb") as f:
        pickle.dump(bank, f)

    candidate_config = {
        "version": 10,
        "candidate_only": True,
        "base_version": 8,
        "trajectory_model": "v8_plus_confidence_gated_template_shape_v1",
        "template_shape_gains": best["gains"].tolist(),
        "template_conf_thresholds": best["thresholds"].tolist(),
        "template_conf_powers": best["powers"].tolist(),
        "template_match_temperature": float(best["temperature"]),
        "template_sequences": list(bank["sequences"]),
        "template_match_accuracy": float(best["match_acc"]),
        "validation_rmse": cand_rmse,
        "validation_proxy_loss": cand_proxy,
        "validation_diff_corr": float(cand["diff_corr"]),
        "validation_peak_f1": float(cand["peak_f1"]),
        "validation_volatility_fit": float(cand["volatility_fit"]),
        "validation_direction_accuracy": float(cand["direction_accuracy"]),
        "validation_trend_score": cand_trend,
        "candidate_passed_local_gate": passed,
    }
    with open(V10_CONFIG_PATH, "wb") as f:
        pickle.dump(candidate_config, f)

    if passed:
        print("PASS V10 LOCAL GATE：保留为明天官网候选，但今天仍不要切换线上 V8。")
    else:
        print("REJECT V10 LOCAL GATE：继续保留 V8，不要上线。")
    print(f"candidate config: {V10_CONFIG_PATH}")
    print(f"template bank   : {V10_BANK_PATH}")
    print("models/ensemble_config.pkl 未修改。")


if __name__ == "__main__":
    main()
