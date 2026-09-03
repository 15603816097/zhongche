import pickle

import numpy as np

from config import MODEL_DIR, TARGET_COLUMNS
from find_best_weight import (
    apply_ensemble_config,
    evaluate_multivariate,
    load_existing_config,
    load_validation_arrays,
    print_report,
    variable_metrics,
)


PCA_VAL_PATH = MODEL_DIR / "val_pred_pca_xgb.npz"
CANDIDATE_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v6.pkl"
CANDIDATE_PRED_PATH = MODEL_DIR / "val_pred_candidate_v6.npz"

WEIGHT_GRID = np.asarray(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80],
    dtype=np.float64,
)

# Accuracy 在官网占 50%，所以 V6 比 V5 更严格保护 RMSE。
MAX_VARIABLE_RMSE_DEGRADATION = 0.015
MAX_GLOBAL_RMSE_DEGRADATION = 0.010
MIN_PROXY_IMPROVEMENT = 0.0010
MAX_TREND_SCORE_DROP = 0.0020
MAX_DIFF_CORR_DROP = 0.0015
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
    """
    越小越好。65% 数值精度 + 35% 趋势。

    这是本地候选排序器，不等于官网真实公式；同时有 RMSE 硬安全闸。
    """
    rmse_ratio = metrics["rmse"] / max(metrics["persistence_rmse"], EPS)
    mae_ratio = metrics["mae"] / max(metrics["persistence_mae"], EPS)
    diff_loss = float(np.clip((1.0 - metrics["diff_corr"]) / 2.0, 0.0, 1.0))
    peak_loss = 1.0 - float(np.clip(metrics["peak_f1"], 0.0, 1.0))
    vol_loss = 1.0 - float(np.clip(metrics["volatility_fit"], 0.0, 1.0))
    dir_loss = 1.0 - float(np.clip(metrics["direction_accuracy"], 0.0, 1.0))

    return float(
        0.45 * rmse_ratio
        + 0.20 * mae_ratio
        + 0.14 * diff_loss
        + 0.09 * peak_loss
        + 0.08 * vol_loss
        + 0.04 * dir_loss
    )


def _load_pca_validation():
    if not PCA_VAL_PATH.exists():
        raise FileNotFoundError(
            f"缺少 {PCA_VAL_PATH}，请先运行 python src/trainer_pca_xgb.py"
        )
    data = np.load(PCA_VAL_PATH, allow_pickle=True)
    return (
        data["pred_abs"].astype(np.float64),
        float(data["rmse"]),
        data["explained_variance"].astype(np.float64),
        data["components"].astype(np.int32),
    )


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    current_config = load_existing_config()

    if int(current_config.get("version", 1)) != 3:
        print(
            f"[WARN] 当前 ensemble version={current_config.get('version')}，"
            "V6 仍以当前主融合输出作为参考，不覆盖现有配置。"
        )

    v3_pred = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        current_config,
    )
    v3_report = evaluate_multivariate(y_true, v3_pred, last_values)
    print_report("当前主融合基准", v3_report)

    pca_pred, pca_rmse, explained, components = _load_pca_validation()
    if pca_pred.shape != y_true.shape:
        raise ValueError(
            f"PCA validation shape 不一致: {pca_pred.shape} vs {y_true.shape}"
        )

    pca_report = evaluate_multivariate(y_true, pca_pred, last_values)
    print_report("Low-Rank PCA XGBoost", pca_report)
    print(f"PCA raw flat RMSE: {pca_rmse:.6f}")
    for j, col in enumerate(TARGET_COLUMNS):
        print(
            f"  {col:16s} components={int(components[j])} "
            f"explained={explained[j]:.4f}"
        )

    weights = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)

    print("\n" + "=" * 112)
    print("V6：V3 + 低秩 PCA 未来轨迹融合搜索")
    print(
        "每变量独立搜索 PCA 权重；"
        f"单变量 RMSE 最多比当前主融合退化 {MAX_VARIABLE_RMSE_DEGRADATION*100:.1f}%"
    )
    print("=" * 112)

    for j, col in enumerate(TARGET_COLUMNS):
        yt = y_true[:, :, j]
        anchor = last_values[:, j]
        ref = v3_pred[:, :, j]
        pp = pca_pred[:, :, j]

        ref_metrics = variable_metrics(yt, ref, anchor)
        ref_loss = _variable_loss(ref_metrics)
        rmse_cap = ref_metrics["rmse"] * (1.0 + MAX_VARIABLE_RMSE_DEGRADATION)

        best = (
            ref_loss,
            ref_metrics["rmse"],
            0.0,
            ref_metrics,
        )

        for w in WEIGHT_GRID:
            candidate = (1.0 - w) * ref + w * pp
            metrics = variable_metrics(yt, candidate, anchor)
            if metrics["rmse"] > rmse_cap + EPS:
                continue

            loss = _variable_loss(metrics)
            key = (loss, metrics["rmse"], float(w), metrics)
            if key[:3] < best[:3]:
                best = key

        best_loss, best_rmse, best_weight, best_metrics = best
        weights[j] = best_weight

        print(
            f"{col:16s} PCA={best_weight:.2f} "
            f"RMSE={best_rmse:.4f} "
            f"Diff={best_metrics['diff_corr']:.4f} "
            f"Peak={best_metrics['peak_f1']:.4f} "
            f"Vol={best_metrics['volatility_fit']:.4f} "
            f"Dir={best_metrics['direction_accuracy']:.4f} "
            f"Loss={best_loss:.4f}"
        )

    w = weights.reshape(1, 1, -1)
    candidate_pred = (1.0 - w) * v3_pred + w * pca_pred
    candidate_report = evaluate_multivariate(
        y_true,
        candidate_pred,
        last_values,
    )
    print_report("V6 PCA-blend candidate", candidate_report)

    v3_rmse = float(v3_report["mean"]["flat_rmse"])
    v6_rmse = float(candidate_report["mean"]["flat_rmse"])
    v3_proxy = float(v3_report["mean"]["proxy_loss"])
    v6_proxy = float(candidate_report["mean"]["proxy_loss"])
    v3_trend = _trend_score(v3_report)
    v6_trend = _trend_score(candidate_report)
    v3_diff = float(v3_report["mean"]["diff_corr"])
    v6_diff = float(candidate_report["mean"]["diff_corr"])

    rmse_ok = v6_rmse <= v3_rmse * (1.0 + MAX_GLOBAL_RMSE_DEGRADATION)
    proxy_ok = v6_proxy <= v3_proxy - MIN_PROXY_IMPROVEMENT
    trend_ok = v6_trend >= v3_trend - MAX_TREND_SCORE_DROP
    diff_ok = v6_diff >= v3_diff - MAX_DIFF_CORR_DROP
    any_weight = bool(np.any(weights > 1e-12))
    candidate_ok = bool(rmse_ok and proxy_ok and trend_ok and diff_ok and any_weight)

    print("\n" + "=" * 112)
    print("V6 候选结论")
    print("=" * 112)
    print(f"V3 flat RMSE : {v3_rmse:.6f}")
    print(
        f"V6 flat RMSE : {v6_rmse:.6f} "
        f"({(v6_rmse/v3_rmse-1.0)*100:+.2f}%)"
    )
    print(f"V3 proxy loss: {v3_proxy:.6f}")
    print(
        f"V6 proxy loss: {v6_proxy:.6f} "
        f"({(v6_proxy/v3_proxy-1.0)*100:+.2f}%)"
    )
    print(f"V3 trendScore: {v3_trend:.6f}")
    print(
        f"V6 trendScore: {v6_trend:.6f} "
        f"({(v6_trend/v3_trend-1.0)*100:+.2f}%)"
    )
    print(f"V6 DiffCorr  : {candidate_report['mean']['diff_corr']:.6f}")
    print(f"V6 PeakF1    : {candidate_report['mean']['peak_f1']:.6f}")
    print(f"V6 VolFit    : {candidate_report['mean']['volatility_fit']:.6f}")
    print(f"V6 DirAcc    : {candidate_report['mean']['direction_accuracy']:.6f}")
    print(f"PCA weights  : {weights.tolist()}")

    np.savez_compressed(
        CANDIDATE_PRED_PATH,
        pred_abs=candidate_pred.astype(np.float32),
        pca_weights=weights.astype(np.float64),
    )

    candidate_config = dict(current_config)
    candidate_config.update(
        {
            "version": 6,
            "candidate_only": True,
            "trajectory_model": "pca_xgb_v1",
            "pca_blend_weights": weights.tolist(),
            "validation_rmse": v6_rmse,
            "validation_proxy_loss": v6_proxy,
            "validation_diff_corr": float(candidate_report["mean"]["diff_corr"]),
            "validation_peak_f1": float(candidate_report["mean"]["peak_f1"]),
            "validation_volatility_fit": float(candidate_report["mean"]["volatility_fit"]),
            "validation_direction_accuracy": float(candidate_report["mean"]["direction_accuracy"]),
            "validation_trend_score": v6_trend,
            "candidate_passed_local_gate": candidate_ok,
        }
    )
    with open(CANDIDATE_CONFIG_PATH, "wb") as f:
        pickle.dump(candidate_config, f)

    if candidate_ok:
        print("PASS V6 LOCAL GATE：候选已保存，但暂不覆盖 ensemble_config.pkl。")
    else:
        print("REJECT V6 LOCAL GATE：候选未通过，当前 V3 完全不受影响。")
    print(f"候选配置: {CANDIDATE_CONFIG_PATH}")
    print("注意：无论 PASS/REJECT，本脚本都不会修改线上正在使用的 ensemble_config.pkl。")


if __name__ == "__main__":
    main()
