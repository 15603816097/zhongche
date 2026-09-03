import pickle
import shutil

import numpy as np

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from find_best_weight import (
    apply_ensemble_config,
    evaluate_multivariate,
    load_existing_config,
    load_validation_arrays,
    print_report,
    variable_metrics,
)


TREND_VAL_PATH = MODEL_DIR / "val_pred_trend_xgb.npz"
ALPHA_GRID = np.asarray(
    [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.20],
    dtype=np.float64,
)
BETA_GRID = np.asarray(
    [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
    dtype=np.float64,
)

# V5 仍然以保住官网 Accuracy 为硬约束。
MAX_VARIABLE_RMSE_DEGRADATION = 0.03
MAX_GLOBAL_RMSE_DEGRADATION = 0.02
MIN_TREND_SCORE_GAIN = 0.005
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


def _variable_trend_score(metrics):
    diff_score = float(
        np.clip((metrics["diff_corr"] + 1.0) / 2.0, 0.0, 1.0)
    )
    return float(
        0.45 * diff_score
        + 0.25 * np.clip(metrics["peak_f1"], 0.0, 1.0)
        + 0.20 * np.clip(metrics["volatility_fit"], 0.0, 1.0)
        + 0.10 * np.clip(metrics["direction_accuracy"], 0.0, 1.0)
    )


def _load_trend_validation():
    if not TREND_VAL_PATH.exists():
        raise FileNotFoundError(
            f"缺少 {TREND_VAL_PATH}，请先运行 python src/trainer_trend_xgb.py"
        )

    data = np.load(TREND_VAL_PATH, allow_pickle=True)
    return (
        data["pred_step_diff"].astype(np.float64),
        data["pred_abs"].astype(np.float64),
        float(data["diff_corr"]),
        float(data["absolute_rmse"]),
    )


def _trend_components(pred_step_diff):
    """
    把 trend 模型输出分解成：
      linear: 只保留最终位移的线性部分
      shape : 去掉线性位移后的局部峰谷/波动，末端严格回到 0

    shape 注入不会直接改变 96 步终点，可显著降低破坏数值精度的风险。
    """
    pred_step_diff = np.asarray(pred_step_diff, dtype=np.float64)
    cumulative = np.cumsum(pred_step_diff, axis=1)
    frac = (
        np.arange(1, HORIZON + 1, dtype=np.float64) / float(HORIZON)
    ).reshape(1, HORIZON, 1)
    final_displacement = cumulative[:, -1:, :]
    linear = frac * final_displacement
    shape = cumulative - linear
    return linear, shape


def apply_v5_trend(v3_pred, last_values, trend_linear, trend_shape, alpha, beta):
    v3_pred = np.asarray(v3_pred, dtype=np.float64)
    last = np.asarray(last_values, dtype=np.float64)[:, None, :]
    alpha = np.asarray(alpha, dtype=np.float64).reshape(1, 1, -1)
    beta = np.asarray(beta, dtype=np.float64).reshape(1, 1, -1)

    base_delta = v3_pred - last
    # beta 只控制最终整体位移；alpha 专门控制零终点的局部趋势形状。
    fused_delta = (
        (1.0 - beta) * base_delta
        + beta * trend_linear
        + alpha * trend_shape
    )
    return last + fused_delta


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    current_config = load_existing_config()

    if int(current_config.get("version", 1)) < 3:
        raise RuntimeError("请先得到 V3 ensemble_config，再运行 V5。")

    v3_pred = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        current_config,
    )
    v3_report = evaluate_multivariate(y_true, v3_pred, last_values)
    print_report("V3 当前基准", v3_report)

    pred_step_diff, trend_abs, trend_diff_corr, trend_abs_rmse = _load_trend_validation()
    trend_linear, trend_shape = _trend_components(pred_step_diff)

    trend_report = evaluate_multivariate(y_true, trend_abs, last_values)
    print_report("Supervised Trend XGBoost", trend_report)
    print(
        f"Trend model raw diffCorr={trend_diff_corr:.6f} "
        f"absoluteRMSE={trend_abs_rmse:.6f}"
    )

    n_targets = len(TARGET_COLUMNS)
    alpha = np.zeros(n_targets, dtype=np.float64)
    beta = np.zeros(n_targets, dtype=np.float64)

    print("\n" + "=" * 112)
    print("V5：监督式一阶差分趋势注入搜索")
    print(
        "alpha=局部峰谷/波动注入，beta=整体位移修正；"
        f"每变量 RMSE 最多退化 {MAX_VARIABLE_RMSE_DEGRADATION*100:.1f}%"
    )
    print("=" * 112)

    for j, col in enumerate(TARGET_COLUMNS):
        yt = y_true[:, :, j]
        anchor = last_values[:, j]
        ref = v3_pred[:, :, j]
        ref_metrics = variable_metrics(yt, ref, anchor)
        ref_trend = _variable_trend_score(ref_metrics)
        rmse_cap = ref_metrics["rmse"] * (1.0 + MAX_VARIABLE_RMSE_DEGRADATION)

        best = (
            -ref_trend,
            ref_metrics["rmse"],
            0.0,
            0.0,
            ref_metrics,
        )

        for a in ALPHA_GRID:
            for b in BETA_GRID:
                candidate = (
                    anchor[:, None]
                    + (1.0 - b) * (ref - anchor[:, None])
                    + b * trend_linear[:, :, j]
                    + a * trend_shape[:, :, j]
                )
                metrics = variable_metrics(yt, candidate, anchor)
                if metrics["rmse"] > rmse_cap + EPS:
                    continue

                trend = _variable_trend_score(metrics)
                # 第一优先趋势分，第二优先 RMSE；只有在 RMSE 安全闸内才比较。
                key = (-trend, metrics["rmse"], float(a), float(b), metrics)
                if key[:4] < best[:4]:
                    best = key

        _, best_rmse, best_alpha, best_beta, best_metrics = best
        alpha[j] = best_alpha
        beta[j] = best_beta

        print(
            f"{col:16s} ALPHA={best_alpha:.2f} BETA={best_beta:.2f} "
            f"RMSE={best_rmse:.4f} "
            f"Diff={best_metrics['diff_corr']:.4f} "
            f"Peak={best_metrics['peak_f1']:.4f} "
            f"Vol={best_metrics['volatility_fit']:.4f} "
            f"Dir={best_metrics['direction_accuracy']:.4f}"
        )

    v5_pred = apply_v5_trend(
        v3_pred,
        last_values,
        trend_linear,
        trend_shape,
        alpha,
        beta,
    )
    v5_report = evaluate_multivariate(y_true, v5_pred, last_values)
    print_report("V5 supervised-trend ensemble", v5_report)

    v3_rmse = float(v3_report["mean"]["flat_rmse"])
    v5_rmse = float(v5_report["mean"]["flat_rmse"])
    v3_trend = _trend_score(v3_report)
    v5_trend = _trend_score(v5_report)

    rmse_ok = v5_rmse <= v3_rmse * (1.0 + MAX_GLOBAL_RMSE_DEGRADATION)
    trend_ok = v5_trend >= v3_trend + MIN_TREND_SCORE_GAIN
    diff_ok = (
        v5_report["mean"]["diff_corr"]
        >= v3_report["mean"]["diff_corr"] - 1e-4
    )
    any_trend_weight = bool(
        np.any(alpha > 1e-12) or np.any(beta > 1e-12)
    )
    accept = bool(rmse_ok and trend_ok and diff_ok and any_trend_weight)

    print("\n" + "=" * 112)
    print("V5 搜索结论")
    print("=" * 112)
    print(f"V3 flat RMSE : {v3_rmse:.6f}")
    print(
        f"V5 flat RMSE : {v5_rmse:.6f} "
        f"({(v5_rmse/v3_rmse-1.0)*100:+.2f}%)"
    )
    print(f"V3 trendScore: {v3_trend:.6f}")
    print(
        f"V5 trendScore: {v5_trend:.6f} "
        f"({(v5_trend/v3_trend-1.0)*100:+.2f}%)"
    )
    print(f"V5 DiffCorr  : {v5_report['mean']['diff_corr']:.6f}")
    print(f"V5 PeakF1    : {v5_report['mean']['peak_f1']:.6f}")
    print(f"V5 VolFit    : {v5_report['mean']['volatility_fit']:.6f}")
    print(f"V5 DirAcc    : {v5_report['mean']['direction_accuracy']:.6f}")
    print(f"alpha        : {alpha.tolist()}")
    print(f"beta         : {beta.tolist()}")

    if not accept:
        print("REJECT V5：没有同时满足趋势提升 + DiffCorr + RMSE 安全闸，继续保留 V3。")
        return

    config_path = MODEL_DIR / "ensemble_config.pkl"
    backup_path = MODEL_DIR / "ensemble_config_before_supervised_trend_v5.pkl"
    if config_path.exists() and not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"已备份 V3 配置: {backup_path}")

    new_config = dict(current_config)
    new_config.update(
        {
            "version": 5,
            "trend_model": "xgb_first_difference_v1",
            "trend_shape_alpha": alpha.tolist(),
            "trend_level_beta": beta.tolist(),
            "validation_rmse": v5_rmse,
            "validation_direction_accuracy": float(
                v5_report["mean"]["direction_accuracy"]
            ),
            "validation_diff_corr": float(v5_report["mean"]["diff_corr"]),
            "validation_peak_f1": float(v5_report["mean"]["peak_f1"]),
            "validation_volatility_fit": float(
                v5_report["mean"]["volatility_fit"]
            ),
            "validation_trend_score": v5_trend,
        }
    )

    with open(config_path, "wb") as f:
        pickle.dump(new_config, f)

    print("ACCEPT V5：已写入 ensemble_config.pkl。")
    print(f"已保存: {config_path}")
    print("下一步重启 API 后 inference 会自动启用 supervised trend model。")


if __name__ == "__main__":
    main()
