import pickle
import shutil
from pathlib import Path

import numpy as np

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS


# 三段预测区间分别搜索融合参数，避免 1~96 步共用一套权重。
HORIZON_SEGMENTS = ((0, 32), (32, 64), (64, 96))
LGB_GRID = np.arange(0.0, 1.01, 0.1)
BASE_GRID = np.arange(0.0, 0.51, 0.1)
GAIN_GRID = np.asarray([0.90, 0.95, 1.00, 1.05, 1.10, 1.15], dtype=np.float64)

# 趋势优化允许局部 RMSE 相比当前官网基准融合最多退化 6%。
# 这样可以用少量 Accuracy 换 Trend，但避免再次出现“趋势涨、准确度崩”。
MAX_RMSE_DEGRADATION = 0.06
EPS = 1e-12


DEFAULT_LGB_WEIGHTS = np.asarray([0.65] * len(TARGET_COLUMNS), dtype=np.float64)
DEFAULT_BASELINE_WEIGHTS = np.asarray([0.15] * len(TARGET_COLUMNS), dtype=np.float64)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def robust_mape(y_true, y_pred):
    median_abs = float(np.median(np.abs(y_true)))
    floor = max(1e-6, 0.05 * median_abs)
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered ** 2))
    if ss_tot <= EPS:
        return 1.0 if ss_res <= EPS else 0.0
    return float(1.0 - ss_res / ss_tot)


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or a.size != b.size:
        return 0.0

    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std <= EPS and b_std <= EPS:
        return 1.0
    if a_std <= EPS or b_std <= EPS:
        return 0.0

    value = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def step_differences(values, anchor):
    values = np.asarray(values, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"values 必须为二维，实际 {values.shape}")
    if len(anchor) != len(values):
        raise ValueError("anchor 数量与样本数不一致")
    extended = np.concatenate([anchor[:, None], values], axis=1)
    return np.diff(extended, axis=1)


def differential_correlation(y_true, y_pred, anchor):
    return safe_corr(
        step_differences(y_true, anchor),
        step_differences(y_pred, anchor),
    )


def direction_accuracy(y_true, y_pred, anchor):
    true_diff = step_differences(y_true, anchor)
    pred_diff = step_differences(y_pred, anchor)
    scale = float(np.std(true_diff))
    deadband = max(1e-12, 1e-4 * scale)

    true_sign = np.where(np.abs(true_diff) <= deadband, 0.0, np.sign(true_diff))
    pred_sign = np.where(np.abs(pred_diff) <= deadband, 0.0, np.sign(pred_diff))
    return float(np.mean(true_sign == pred_sign))


def volatility_fit(y_true, y_pred, anchor):
    true_diff = step_differences(y_true, anchor)
    pred_diff = step_differences(y_pred, anchor)
    true_std = float(np.std(true_diff))
    pred_std = float(np.std(pred_diff))

    if true_std <= EPS and pred_std <= EPS:
        return 1.0
    if true_std <= EPS or pred_std <= EPS:
        return 0.0

    ratio = (pred_std + EPS) / (true_std + EPS)
    return float(np.clip(np.exp(-abs(np.log(ratio))), 0.0, 1.0))


def _peak_candidates(values, threshold):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 3:
        return []

    prev = values[:-2]
    center = values[1:-1]
    nxt = values[2:]
    maxima = (center > prev) & (center >= nxt)
    minima = (center < prev) & (center <= nxt)
    prominence = np.minimum(np.abs(center - prev), np.abs(center - nxt))
    keep = (maxima | minima) & (prominence >= threshold)

    result = []
    for local_idx in np.where(keep)[0]:
        idx = int(local_idx + 1)
        kind = 1 if bool(maxima[local_idx]) else -1
        result.append((idx, kind))
    return result


def peak_f1(y_true, y_pred, tolerance=2):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("peak_f1 输入 shape 异常")

    tp = 0
    fp = 0
    fn = 0

    for true_row, pred_row in zip(y_true, y_pred):
        diff = np.diff(true_row)
        if len(diff) == 0:
            continue

        med = float(np.median(diff))
        mad = float(np.median(np.abs(diff - med)))
        robust_sigma = 1.4826 * mad
        std = float(np.std(diff))
        threshold = max(1e-10, 0.35 * std, 0.50 * robust_sigma)

        true_peaks = _peak_candidates(true_row, threshold)
        pred_peaks = _peak_candidates(pred_row, threshold)

        used = set()
        local_tp = 0
        for true_idx, true_kind in true_peaks:
            candidates = [
                (abs(pred_idx - true_idx), k)
                for k, (pred_idx, pred_kind) in enumerate(pred_peaks)
                if k not in used
                and pred_kind == true_kind
                and abs(pred_idx - true_idx) <= tolerance
            ]
            if candidates:
                _, best_k = min(candidates)
                used.add(best_k)
                local_tp += 1

        tp += local_tp
        fn += len(true_peaks) - local_tp
        fp += len(pred_peaks) - local_tp

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall <= EPS:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def variable_metrics(y_true, y_pred, anchor):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)

    persistence = np.repeat(anchor[:, None], y_true.shape[1], axis=1)
    pred_rmse = rmse(y_true, y_pred)
    pred_mae = mae(y_true, y_pred)
    persistence_rmse = rmse(y_true, persistence)
    persistence_mae = mae(y_true, persistence)

    return {
        "rmse": pred_rmse,
        "mae": pred_mae,
        "mape": robust_mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "persistence_rmse": persistence_rmse,
        "persistence_mae": persistence_mae,
        "persistence_gain": float(1.0 - pred_rmse / max(persistence_rmse, EPS)),
        "diff_corr": differential_correlation(y_true, y_pred, anchor),
        "peak_f1": peak_f1(y_true, y_pred),
        "volatility_fit": volatility_fit(y_true, y_pred, anchor),
        "direction_accuracy": direction_accuracy(y_true, y_pred, anchor),
    }


def competition_proxy_loss(metrics):
    """
    本地排序代理损失，越小越好，不等于官网真实分数。

    55% 数值准确度：RMSE + MAE（均相对 persistence 归一化）
    45% 趋势质量：差分相关 + 波动拟合 + 峰值F1 + 方向一致
    """
    rmse_ratio = min(
        3.0,
        metrics["rmse"] / max(metrics["persistence_rmse"], EPS),
    )
    mae_ratio = min(
        3.0,
        metrics["mae"] / max(metrics["persistence_mae"], EPS),
    )
    diff_loss = float(np.clip((1.0 - metrics["diff_corr"]) / 2.0, 0.0, 1.0))
    vol_loss = 1.0 - float(np.clip(metrics["volatility_fit"], 0.0, 1.0))
    peak_loss = 1.0 - float(np.clip(metrics["peak_f1"], 0.0, 1.0))
    dir_loss = 1.0 - float(np.clip(metrics["direction_accuracy"], 0.0, 1.0))

    return float(
        0.38 * rmse_ratio
        + 0.17 * mae_ratio
        + 0.18 * diff_loss
        + 0.12 * vol_loss
        + 0.10 * peak_loss
        + 0.05 * dir_loss
    )


def load_validation_arrays():
    lgb_path = MODEL_DIR / "val_pred_lgb.npz"
    xgb_path = MODEL_DIR / "val_pred_xgb.npz"

    if not lgb_path.exists() or not xgb_path.exists():
        raise FileNotFoundError(
            "请先运行 python src/trainer.py 和 python src/trainer_xgb.py"
        )

    lgb = np.load(lgb_path, allow_pickle=True)
    xgb = np.load(xgb_path, allow_pickle=True)

    y_true = lgb["y_abs"].astype(np.float64)
    last_values = lgb["last_values"].astype(np.float64)
    baseline = lgb["baseline_abs"].astype(np.float64)
    pred_lgb = lgb["pred_abs"].astype(np.float64)
    pred_xgb = xgb["pred_abs"].astype(np.float64)

    if not (y_true.shape == pred_lgb.shape == pred_xgb.shape == baseline.shape):
        raise ValueError("LGB/XGB/baseline 验证预测 shape 不一致")

    return y_true, last_values, baseline, pred_lgb, pred_xgb


def load_existing_config():
    path = MODEL_DIR / "ensemble_config.pkl"
    if not path.exists():
        return {
            "version": 1,
            "lgb_weights": DEFAULT_LGB_WEIGHTS.tolist(),
            "baseline_weights": DEFAULT_BASELINE_WEIGHTS.tolist(),
            "target_columns": list(TARGET_COLUMNS),
        }
    with open(path, "rb") as f:
        return pickle.load(f)


def stepwise_parameters_from_config(config):
    n_targets = len(TARGET_COLUMNS)

    old_lgb = np.asarray(
        config.get("lgb_weights", DEFAULT_LGB_WEIGHTS),
        dtype=np.float64,
    )
    old_base = np.asarray(
        config.get("baseline_weights", DEFAULT_BASELINE_WEIGHTS),
        dtype=np.float64,
    )
    if old_lgb.shape != (n_targets,):
        old_lgb = DEFAULT_LGB_WEIGHTS.copy()
    if old_base.shape != (n_targets,):
        old_base = DEFAULT_BASELINE_WEIGHTS.copy()

    lgb_step = np.repeat(old_lgb[None, :], HORIZON, axis=0)
    base_step = np.repeat(old_base[None, :], HORIZON, axis=0)
    gain_step = np.ones((HORIZON, n_targets), dtype=np.float64)

    segments = config.get("horizon_segments")
    lgb_seg = np.asarray(config.get("lgb_weights_by_segment", []), dtype=np.float64)
    base_seg = np.asarray(config.get("baseline_weights_by_segment", []), dtype=np.float64)
    gain_seg = np.asarray(config.get("delta_gain_by_segment", []), dtype=np.float64)

    expected_shape = (len(HORIZON_SEGMENTS), n_targets)
    if (
        isinstance(segments, (list, tuple))
        and len(segments) == len(HORIZON_SEGMENTS)
        and lgb_seg.shape == expected_shape
        and base_seg.shape == expected_shape
        and gain_seg.shape == expected_shape
    ):
        for k, pair in enumerate(segments):
            start, end = int(pair[0]), int(pair[1])
            start = max(0, min(HORIZON, start))
            end = max(start, min(HORIZON, end))
            lgb_step[start:end] = lgb_seg[k]
            base_step[start:end] = base_seg[k]
            gain_step[start:end] = gain_seg[k]

    return (
        np.clip(lgb_step, 0.0, 1.0),
        np.clip(base_step, 0.0, 0.8),
        np.clip(gain_step, 0.75, 1.35),
    )


def apply_ensemble_config(
    pred_lgb,
    pred_xgb,
    baseline,
    last_values,
    config,
):
    lgb_step, base_step, gain_step = stepwise_parameters_from_config(config)

    ml_pred = (
        lgb_step[None, :, :] * pred_lgb
        + (1.0 - lgb_step[None, :, :]) * pred_xgb
    )
    pred = (
        (1.0 - base_step[None, :, :]) * ml_pred
        + base_step[None, :, :] * baseline
    )

    last = np.asarray(last_values, dtype=np.float64)[:, None, :]
    pred = last + gain_step[None, :, :] * (pred - last)
    return np.asarray(pred, dtype=np.float64)


def evaluate_multivariate(y_true, y_pred, last_values):
    per_variable = {}
    for j, col in enumerate(TARGET_COLUMNS):
        per_variable[col] = variable_metrics(
            y_true[:, :, j],
            y_pred[:, :, j],
            last_values[:, j],
        )

    mean = {}
    metric_names = list(next(iter(per_variable.values())).keys())
    for name in metric_names:
        mean[name] = float(
            np.mean([per_variable[col][name] for col in TARGET_COLUMNS])
        )

    mean["proxy_loss"] = float(
        np.mean([
            competition_proxy_loss(per_variable[col])
            for col in TARGET_COLUMNS
        ])
    )
    mean["flat_rmse"] = rmse(y_true, y_pred)

    return {"per_variable": per_variable, "mean": mean}


def print_report(title, report):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    print(
        f"{'variable':16s} {'RMSE':>10s} {'MAE':>10s} {'R2':>9s} "
        f"{'PersistGain':>12s} {'DiffCorr':>10s} {'PeakF1':>9s} "
        f"{'VolFit':>9s} {'DirAcc':>9s}"
    )
    print("-" * 112)

    for col in TARGET_COLUMNS:
        m = report["per_variable"][col]
        print(
            f"{col:16s} {m['rmse']:10.4f} {m['mae']:10.4f} {m['r2']:9.4f} "
            f"{m['persistence_gain']:12.4f} {m['diff_corr']:10.4f} "
            f"{m['peak_f1']:9.4f} {m['volatility_fit']:9.4f} "
            f"{m['direction_accuracy']:9.4f}"
        )

    m = report["mean"]
    print("-" * 112)
    print(
        f"MEAN             {m['rmse']:10.4f} {m['mae']:10.4f} {m['r2']:9.4f} "
        f"{m['persistence_gain']:12.4f} {m['diff_corr']:10.4f} "
        f"{m['peak_f1']:9.4f} {m['volatility_fit']:9.4f} "
        f"{m['direction_accuracy']:9.4f}"
    )
    print(
        f"flat_RMSE={m['flat_rmse']:.6f}  proxy_loss={m['proxy_loss']:.6f} "
        "(proxy_loss 越低越好；不是官网真实分数)"
    )


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    current_config = load_existing_config()
    reference_pred = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        current_config,
    )
    reference_report = evaluate_multivariate(
        y_true,
        reference_pred,
        last_values,
    )

    print_report(
        f"当前融合基准（ensemble version={current_config.get('version')}）",
        reference_report,
    )

    ref_lgb_step, ref_base_step, ref_gain_step = stepwise_parameters_from_config(
        current_config
    )

    n_segments = len(HORIZON_SEGMENTS)
    n_targets = len(TARGET_COLUMNS)
    lgb_weights = np.zeros((n_segments, n_targets), dtype=np.float64)
    baseline_weights = np.zeros((n_segments, n_targets), dtype=np.float64)
    delta_gain = np.ones((n_segments, n_targets), dtype=np.float64)

    print("\n" + "=" * 112)
    print("开始趋势感知分段融合搜索")
    print(
        "目标：55% Accuracy proxy + 45% Trend proxy；"
        f"每段每变量 RMSE 最多比当前融合退化 {MAX_RMSE_DEGRADATION*100:.1f}%"
    )
    print("预测区间：1-32 / 33-64 / 65-96")
    print("=" * 112)

    for seg_idx, (start, end) in enumerate(HORIZON_SEGMENTS):
        print(f"\n[SEGMENT {seg_idx + 1}] step {start + 1}-{end}")

        for j, col in enumerate(TARGET_COLUMNS):
            yt = y_true[:, start:end, j]
            pl = pred_lgb[:, start:end, j]
            px = pred_xgb[:, start:end, j]
            pb = baseline[:, start:end, j]

            # 第一段从真实历史最后一点开始；后续分段仅用于验证度量，
            # 用真实上一时刻作为局部差分锚点，避免跨段首点差分被错误放大。
            anchor = (
                last_values[:, j]
                if start == 0
                else y_true[:, start - 1, j]
            )

            ref_segment = reference_pred[:, start:end, j]
            ref_metrics = variable_metrics(yt, ref_segment, anchor)
            rmse_cap = ref_metrics["rmse"] * (1.0 + MAX_RMSE_DEGRADATION) + EPS

            ref_w_lgb = float(np.mean(ref_lgb_step[start:end, j]))
            ref_w_base = float(np.mean(ref_base_step[start:end, j]))
            ref_gain = float(np.mean(ref_gain_step[start:end, j]))
            best = (
                competition_proxy_loss(ref_metrics),
                ref_metrics["rmse"],
                -ref_metrics["diff_corr"],
                -ref_metrics["peak_f1"],
                ref_w_lgb,
                ref_w_base,
                ref_gain,
                ref_metrics,
            )

            for w_lgb in LGB_GRID:
                ml = w_lgb * pl + (1.0 - w_lgb) * px

                for w_base in BASE_GRID:
                    raw = (1.0 - w_base) * ml + w_base * pb

                    for gain in GAIN_GRID:
                        pred = anchor[:, None] + gain * (raw - anchor[:, None])
                        metrics = variable_metrics(yt, pred, anchor)

                        if metrics["rmse"] > rmse_cap:
                            continue

                        objective = competition_proxy_loss(metrics)
                        candidate = (
                            objective,
                            metrics["rmse"],
                            -metrics["diff_corr"],
                            -metrics["peak_f1"],
                            float(w_lgb),
                            float(w_base),
                            float(gain),
                            metrics,
                        )
                        if candidate[:7] < best[:7]:
                            best = candidate

            (
                best_loss,
                best_rmse,
                _,
                _,
                best_lgb,
                best_base,
                best_gain,
                best_metrics,
            ) = best

            lgb_weights[seg_idx, j] = best_lgb
            baseline_weights[seg_idx, j] = best_base
            delta_gain[seg_idx, j] = best_gain

            print(
                f"  {col:16s} "
                f"LGB={best_lgb:.2f} XGB={1-best_lgb:.2f} "
                f"BASE={best_base:.2f} GAIN={best_gain:.2f} "
                f"RMSE={best_rmse:.4f} "
                f"Diff={best_metrics['diff_corr']:.4f} "
                f"Peak={best_metrics['peak_f1']:.4f} "
                f"Vol={best_metrics['volatility_fit']:.4f} "
                f"Loss={best_loss:.4f}"
            )

    config = {
        "version": 3,
        # 兼容旧代码/日志：保留一维平均权重。
        "lgb_weights": np.mean(lgb_weights, axis=0).tolist(),
        "baseline_weights": np.mean(baseline_weights, axis=0).tolist(),
        "target_columns": list(TARGET_COLUMNS),
        "horizon_segments": [list(x) for x in HORIZON_SEGMENTS],
        "lgb_weights_by_segment": lgb_weights.tolist(),
        "baseline_weights_by_segment": baseline_weights.tolist(),
        "delta_gain_by_segment": delta_gain.tolist(),
        "optimization": {
            "name": "trend_aware_segmented_blend",
            "max_rmse_degradation": MAX_RMSE_DEGRADATION,
            "proxy_weights": {
                "rmse_ratio": 0.38,
                "mae_ratio": 0.17,
                "diff_corr": 0.18,
                "volatility_fit": 0.12,
                "peak_f1": 0.10,
                "direction_accuracy": 0.05,
            },
        },
    }

    final_pred = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        config,
    )
    final_report = evaluate_multivariate(y_true, final_pred, last_values)
    m = final_report["mean"]

    config.update(
        {
            "validation_rmse": m["flat_rmse"],
            "validation_direction_accuracy": m["direction_accuracy"],
            "validation_diff_corr": m["diff_corr"],
            "validation_peak_f1": m["peak_f1"],
            "validation_volatility_fit": m["volatility_fit"],
            "validation_proxy_loss": m["proxy_loss"],
            "reference_validation_rmse": reference_report["mean"]["flat_rmse"],
            "reference_validation_proxy_loss": reference_report["mean"]["proxy_loss"],
        }
    )

    print_report("趋势感知分段融合 V3", final_report)

    config_path = MODEL_DIR / "ensemble_config.pkl"
    backup_path = MODEL_DIR / "ensemble_config_before_trend_v3.pkl"
    if config_path.exists() and not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"已备份旧融合配置: {backup_path}")

    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    ref_flat = reference_report["mean"]["flat_rmse"]
    new_flat = final_report["mean"]["flat_rmse"]
    ref_loss = reference_report["mean"]["proxy_loss"]
    new_loss = final_report["mean"]["proxy_loss"]

    print("\n" + "=" * 112)
    print("搜索完成")
    print("=" * 112)
    print(f"参考 flat RMSE: {ref_flat:.6f}")
    print(f"V3   flat RMSE: {new_flat:.6f} ({(new_flat/ref_flat-1)*100:+.2f}%)")
    print(f"参考 proxy loss: {ref_loss:.6f}")
    print(f"V3   proxy loss: {new_loss:.6f} ({(new_loss/ref_loss-1)*100:+.2f}%)")
    print(f"V3 DiffCorr: {m['diff_corr']:.4f}")
    print(f"V3 PeakF1:   {m['peak_f1']:.4f}")
    print(f"V3 VolFit:   {m['volatility_fit']:.4f}")
    print(f"V3 DirAcc:   {m['direction_accuracy']:.4f}")
    print(f"已保存: {config_path}")
    print("注意：proxy 仅用于本地排序，不等于官网分数。")


if __name__ == "__main__":
    main()
