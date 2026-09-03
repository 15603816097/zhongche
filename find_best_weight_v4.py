import pickle
import shutil

import numpy as np

from config import HORIZON, LOOKBACK, MODEL_DIR, TARGET_COLUMNS
from find_best_weight import (
    apply_ensemble_config,
    competition_proxy_loss,
    evaluate_multivariate,
    load_existing_config,
    load_validation_arrays,
    print_report,
    variable_metrics,
)
from src.dataset_builder import load_all_data, temporal_train_val_indices
from src.trend_pattern import adaptive_pattern_forecast_array


HORIZON_SEGMENTS = ((0, 32), (32, 64), (64, 96))
PATTERN_GRID = np.asarray(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    dtype=np.float64,
)

# V4 的目标是补趋势，不再大幅动 V3 数值预测。
# 每段每变量最多允许 RMSE 比 V3 退化 2.5%。
MAX_SEGMENT_RMSE_DEGRADATION = 0.025
MAX_GLOBAL_RMSE_DEGRADATION = 0.015
EPS = 1e-12


def _trend_priority_loss(metrics):
    """越小越好。40% 数值 + 60% 趋势，用 RMSE cap 保住 Accuracy。"""
    rmse_ratio = min(
        3.0,
        metrics["rmse"] / max(metrics["persistence_rmse"], EPS),
    )
    mae_ratio = min(
        3.0,
        metrics["mae"] / max(metrics["persistence_mae"], EPS),
    )
    diff_loss = float(
        np.clip((1.0 - metrics["diff_corr"]) / 2.0, 0.0, 1.0)
    )
    peak_loss = 1.0 - float(np.clip(metrics["peak_f1"], 0.0, 1.0))
    vol_loss = 1.0 - float(np.clip(metrics["volatility_fit"], 0.0, 1.0))
    dir_loss = 1.0 - float(
        np.clip(metrics["direction_accuracy"], 0.0, 1.0)
    )

    return float(
        0.28 * rmse_ratio
        + 0.12 * mae_ratio
        + 0.25 * diff_loss
        + 0.20 * peak_loss
        + 0.10 * vol_loss
        + 0.05 * dir_loss
    )


def _trend_score(report):
    """越大越好，仅用于 V3/V4 同一验证集比较。"""
    m = report["mean"]
    diff_score = float(np.clip((m["diff_corr"] + 1.0) / 2.0, 0.0, 1.0))
    return float(
        0.40 * diff_score
        + 0.30 * np.clip(m["peak_f1"], 0.0, 1.0)
        + 0.20 * np.clip(m["volatility_fit"], 0.0, 1.0)
        + 0.10 * np.clip(m["direction_accuracy"], 0.0, 1.0)
    )


def _build_pattern_validation():
    """
    从训练缓存第一段 144x6 原始窗口还原验证输入，并计算 V4 pattern baseline。
    这样不需要重新训练 LGB/XGB。
    """
    cache_path = MODEL_DIR / "val_pred_pattern_v4.npz"

    bundle = load_all_data(use_cache=True)
    _, val_idx = temporal_train_val_indices(bundle)

    raw_dim = LOOKBACK * len(TARGET_COLUMNS)
    if bundle.X.shape[1] < raw_dim:
        raise RuntimeError(
            f"特征维度不足，无法还原原始窗口: {bundle.X.shape[1]} < {raw_dim}"
        )

    windows = bundle.X[val_idx, :raw_dim].astype(np.float64).reshape(
        len(val_idx), LOOKBACK, len(TARGET_COLUMNS)
    )

    pattern = np.empty(
        (len(val_idx), HORIZON, len(TARGET_COLUMNS)),
        dtype=np.float64,
    )

    for i, window in enumerate(windows):
        pattern[i] = adaptive_pattern_forecast_array(window, HORIZON)
        if (i + 1) % 50 == 0 or i + 1 == len(windows):
            print(f"  pattern baseline: {i + 1}/{len(windows)}")

    np.savez_compressed(
        cache_path,
        pred_abs=pattern.astype(np.float32),
        starts=bundle.starts[val_idx].astype(np.int32),
        sequence_names=bundle.sequence_names[val_idx].astype(str),
    )
    print(f"已保存 pattern 验证预测: {cache_path}")
    return pattern


def _expand_pattern_weights(config):
    weights = np.zeros((HORIZON, len(TARGET_COLUMNS)), dtype=np.float64)
    segments = config.get("horizon_segments_v4", HORIZON_SEGMENTS)
    by_segment = np.asarray(
        config.get("pattern_weights_by_segment", []),
        dtype=np.float64,
    )

    expected = (len(segments), len(TARGET_COLUMNS))
    if by_segment.shape != expected:
        return weights

    for k, pair in enumerate(segments):
        start, end = int(pair[0]), int(pair[1])
        start = max(0, min(HORIZON, start))
        end = max(start, min(HORIZON, end))
        weights[start:end] = by_segment[k]
    return np.clip(weights, 0.0, 0.8)


def apply_v4_pattern(v3_pred, pattern_pred, config):
    w = _expand_pattern_weights(config)
    return (
        (1.0 - w[None, :, :]) * np.asarray(v3_pred, dtype=np.float64)
        + w[None, :, :] * np.asarray(pattern_pred, dtype=np.float64)
    )


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    current_config = load_existing_config()

    if int(current_config.get("version", 1)) < 3:
        raise RuntimeError(
            "请先运行 bash run_optimize_v3.sh，得到 ensemble version=3 后再运行 V4。"
        )

    v3_pred = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        current_config,
    )
    v3_report = evaluate_multivariate(y_true, v3_pred, last_values)
    print_report("V3 当前基准", v3_report)

    print("\n构建自适应 pattern trend baseline ...")
    pattern = _build_pattern_validation()
    pattern_report = evaluate_multivariate(y_true, pattern, last_values)
    print_report("Adaptive pattern baseline", pattern_report)

    n_segments = len(HORIZON_SEGMENTS)
    n_targets = len(TARGET_COLUMNS)
    pattern_weights = np.zeros((n_segments, n_targets), dtype=np.float64)

    print("\n" + "=" * 112)
    print("V4：在 V3 上做保守 pattern trend 注入")
    print(
        "目标：40% Accuracy proxy + 60% Trend proxy；"
        f"每段每变量 RMSE 最多退化 {MAX_SEGMENT_RMSE_DEGRADATION*100:.1f}%"
    )
    print("=" * 112)

    for seg_idx, (start, end) in enumerate(HORIZON_SEGMENTS):
        print(f"\n[SEGMENT {seg_idx + 1}] step {start + 1}-{end}")

        for j, col in enumerate(TARGET_COLUMNS):
            yt = y_true[:, start:end, j]
            ref = v3_pred[:, start:end, j]
            pp = pattern[:, start:end, j]
            anchor = (
                last_values[:, j]
                if start == 0
                else y_true[:, start - 1, j]
            )

            ref_metrics = variable_metrics(yt, ref, anchor)
            rmse_cap = (
                ref_metrics["rmse"]
                * (1.0 + MAX_SEGMENT_RMSE_DEGRADATION)
                + EPS
            )

            best = (
                _trend_priority_loss(ref_metrics),
                ref_metrics["rmse"],
                -ref_metrics["diff_corr"],
                -ref_metrics["peak_f1"],
                0.0,
                ref_metrics,
            )

            for w in PATTERN_GRID:
                pred = (1.0 - w) * ref + w * pp
                metrics = variable_metrics(yt, pred, anchor)
                if metrics["rmse"] > rmse_cap:
                    continue

                candidate = (
                    _trend_priority_loss(metrics),
                    metrics["rmse"],
                    -metrics["diff_corr"],
                    -metrics["peak_f1"],
                    float(w),
                    metrics,
                )
                if candidate[:5] < best[:5]:
                    best = candidate

            best_loss, best_rmse, _, _, best_w, best_metrics = best
            pattern_weights[seg_idx, j] = best_w

            print(
                f"  {col:16s} PATTERN={best_w:.2f} "
                f"RMSE={best_rmse:.4f} "
                f"Diff={best_metrics['diff_corr']:.4f} "
                f"Peak={best_metrics['peak_f1']:.4f} "
                f"Vol={best_metrics['volatility_fit']:.4f} "
                f"Dir={best_metrics['direction_accuracy']:.4f} "
                f"Loss={best_loss:.4f}"
            )

    config_v4 = dict(current_config)
    config_v4["version"] = 4
    config_v4["horizon_segments_v4"] = [list(x) for x in HORIZON_SEGMENTS]
    config_v4["pattern_weights_by_segment"] = pattern_weights.tolist()
    config_v4["pattern_baseline"] = "adaptive_repeated_diff_residual_v1"

    v4_pred = apply_v4_pattern(v3_pred, pattern, config_v4)
    v4_report = evaluate_multivariate(y_true, v4_pred, last_values)
    print_report("V4 pattern-aware ensemble", v4_report)

    v3_rmse = float(v3_report["mean"]["flat_rmse"])
    v4_rmse = float(v4_report["mean"]["flat_rmse"])
    v3_trend = _trend_score(v3_report)
    v4_trend = _trend_score(v4_report)
    v3_proxy = float(v3_report["mean"]["proxy_loss"])
    v4_proxy = float(v4_report["mean"]["proxy_loss"])

    print("\n" + "=" * 112)
    print("V4 搜索结论")
    print("=" * 112)
    print(f"V3 flat RMSE : {v3_rmse:.6f}")
    print(f"V4 flat RMSE : {v4_rmse:.6f} ({(v4_rmse/v3_rmse-1)*100:+.2f}%)")
    print(f"V3 proxy loss: {v3_proxy:.6f}")
    print(f"V4 proxy loss: {v4_proxy:.6f} ({(v4_proxy/v3_proxy-1)*100:+.2f}%)")
    print(f"V3 trendScore: {v3_trend:.6f}")
    print(f"V4 trendScore: {v4_trend:.6f} ({(v4_trend/v3_trend-1)*100:+.2f}%)")
    print(f"V4 DiffCorr  : {v4_report['mean']['diff_corr']:.6f}")
    print(f"V4 PeakF1    : {v4_report['mean']['peak_f1']:.6f}")
    print(f"V4 VolFit    : {v4_report['mean']['volatility_fit']:.6f}")
    print(f"V4 DirAcc    : {v4_report['mean']['direction_accuracy']:.6f}")

    # 全局安全闸：趋势必须进步，且总体 RMSE 不允许明显恶化。
    accept = (
        v4_trend > v3_trend + 1e-6
        and v4_rmse <= v3_rmse * (1.0 + MAX_GLOBAL_RMSE_DEGRADATION)
    )

    config_path = MODEL_DIR / "ensemble_config.pkl"
    backup_path = MODEL_DIR / "ensemble_config_before_pattern_v4.pkl"

    if accept:
        if config_path.exists() and not backup_path.exists():
            shutil.copy2(config_path, backup_path)
            print(f"已备份 V3 配置: {backup_path}")

        config_v4["validation_rmse"] = v4_rmse
        config_v4["validation_direction_accuracy"] = float(
            v4_report["mean"]["direction_accuracy"]
        )
        config_v4["validation_diff_corr"] = float(v4_report["mean"]["diff_corr"])
        config_v4["validation_peak_f1"] = float(v4_report["mean"]["peak_f1"])
        config_v4["validation_volatility_fit"] = float(
            v4_report["mean"]["volatility_fit"]
        )
        config_v4["validation_proxy_loss"] = v4_proxy
        config_v4["validation_trend_score"] = v4_trend

        with open(config_path, "wb") as f:
            pickle.dump(config_v4, f)
        print(f"ACCEPT V4：已保存 {config_path}")
    else:
        print("REJECT V4：没有同时满足趋势提升 + RMSE 安全闸，继续保留 V3。")

    print("注意：所有 proxy 都只是本地排序信号，不等于官网真实分数。")


if __name__ == "__main__":
    main()
