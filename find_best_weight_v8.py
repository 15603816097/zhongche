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
from src.trajectory_fusion import endpoint_zero_highpass


PCA_VAL_PATH = MODEL_DIR / "val_pred_pca_xgb.npz"
CANDIDATE_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v8.pkl"
CANDIDATE_PRED_PATH = MODEL_DIR / "val_pred_candidate_v8.npz"

PCA_WEIGHT_GRID = np.asarray(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90],
    dtype=np.float64,
)
HF_GAIN_GRID = np.asarray(
    [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.20],
    dtype=np.float64,
)
SMOOTH_WINDOWS = (5, 7, 9, 13, 17)
SOURCE_NAMES = ("v3", "lgb", "xgb")

# V8 核心：V7 证明“PCA低频 + 高频残差”有效，但 V3 高频残差会拖低 DiffCorr。
# 因此 V8 允许每个变量从 V3 / LGB / XGB 中选择最适合的高频来源。
MAX_VARIABLE_RMSE_DEGRADATION = 0.005
MAX_DIFF_DROP_PER_VAR = 0.0005
MAX_PEAK_DROP_PER_VAR = 0.004
MAX_VOL_DROP_PER_VAR = 0.020

# 全局门槛：Accuracy 必须明显提升，Trend 三项至少不能靠牺牲 DiffCorr 换取。
MIN_GLOBAL_RMSE_IMPROVEMENT = 0.015
MIN_PROXY_IMPROVEMENT = 0.0010
MIN_GLOBAL_TREND_SCORE_GAIN = 0.0020
MAX_GLOBAL_DIFF_DROP = 0.0002
MIN_GLOBAL_PEAK_GAIN = 0.0050
MIN_GLOBAL_VOL_GAIN = 0.0100
EPS = 1e-12


def _load_pca_validation():
    if not PCA_VAL_PATH.exists():
        raise FileNotFoundError(
            f"缺少 {PCA_VAL_PATH}，请先运行 bash run_optimize_v6.sh"
        )
    data = np.load(PCA_VAL_PATH, allow_pickle=True)
    return data["pred_abs"].astype(np.float64)


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


def _variable_loss(metrics):
    """越小越好。58% 数值 + 42% 趋势；硬门槛负责防止局部指标被交换掉。"""
    rmse_ratio = metrics["rmse"] / max(metrics["persistence_rmse"], EPS)
    mae_ratio = metrics["mae"] / max(metrics["persistence_mae"], EPS)
    diff_loss = float(np.clip((1.0 - metrics["diff_corr"]) / 2.0, 0.0, 1.0))
    peak_loss = 1.0 - float(np.clip(metrics["peak_f1"], 0.0, 1.0))
    vol_loss = 1.0 - float(np.clip(metrics["volatility_fit"], 0.0, 1.0))
    dir_loss = 1.0 - float(np.clip(metrics["direction_accuracy"], 0.0, 1.0))
    return float(
        0.38 * rmse_ratio
        + 0.20 * mae_ratio
        + 0.18 * diff_loss
        + 0.10 * peak_loss
        + 0.10 * vol_loss
        + 0.04 * dir_loss
    )


def _candidate(ref, pca, highpass, pca_weight, hf_gain):
    low_rank = (1.0 - pca_weight) * ref + pca_weight * pca
    return low_rank + hf_gain * highpass


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    config = load_existing_config()

    ref_pred = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        config,
    )
    ref_report = evaluate_multivariate(y_true, ref_pred, last_values)
    print_report("V3 当前线上基准", ref_report)

    pca_pred = _load_pca_validation()
    if pca_pred.shape != ref_pred.shape:
        raise ValueError(
            f"PCA prediction shape 不一致: {pca_pred.shape} vs {ref_pred.shape}"
        )

    source_cube = {
        "v3": ref_pred,
        "lgb": pred_lgb,
        "xgb": pred_xgb,
    }

    n_targets = len(TARGET_COLUMNS)
    pca_weights = np.zeros(n_targets, dtype=np.float64)
    hf_gains = np.zeros(n_targets, dtype=np.float64)
    smooth_windows = np.full(n_targets, 9, dtype=np.int32)
    hf_sources = np.full(n_targets, "v3", dtype="<U8")

    print("\n" + "=" * 124)
    print("V8：PCA 低频轨迹 + source-aware 高频峰谷恢复")
    print("每变量可从 V3/LGB/XGB 选择高频残差来源；目标是在保住 V7 峰谷收益的同时恢复 DiffCorr。")
    print("=" * 124)

    for j, col in enumerate(TARGET_COLUMNS):
        yt = y_true[:, :, j]
        anchor = last_values[:, j]
        ref = ref_pred[:, :, j]
        pp = pca_pred[:, :, j]

        ref_metrics = variable_metrics(yt, ref, anchor)
        rmse_cap = ref_metrics["rmse"] * (1.0 + MAX_VARIABLE_RMSE_DEGRADATION)
        diff_floor = ref_metrics["diff_corr"] - MAX_DIFF_DROP_PER_VAR
        peak_floor = max(0.0, ref_metrics["peak_f1"] - MAX_PEAK_DROP_PER_VAR)
        vol_floor = max(0.0, ref_metrics["volatility_fit"] - MAX_VOL_DROP_PER_VAR)

        best = (
            _variable_loss(ref_metrics),
            ref_metrics["rmse"],
            -_variable_trend_score(ref_metrics),
            0.0,
            0.0,
            9,
            "v3",
            ref_metrics,
        )

        for source_name in SOURCE_NAMES:
            source = source_cube[source_name][:, :, j]
            for window in SMOOTH_WINDOWS:
                hf = endpoint_zero_highpass(source, window)

                for w in PCA_WEIGHT_GRID:
                    low_rank = (1.0 - w) * ref + w * pp
                    for gamma in HF_GAIN_GRID:
                        candidate = low_rank + gamma * hf
                        metrics = variable_metrics(yt, candidate, anchor)

                        if metrics["rmse"] > rmse_cap + EPS:
                            continue
                        if metrics["diff_corr"] < diff_floor:
                            continue
                        if metrics["peak_f1"] < peak_floor:
                            continue
                        if metrics["volatility_fit"] < vol_floor:
                            continue

                        loss = _variable_loss(metrics)
                        key = (
                            loss,
                            metrics["rmse"],
                            -_variable_trend_score(metrics),
                            float(w),
                            float(gamma),
                            int(window),
                            source_name,
                            metrics,
                        )
                        if key[:7] < best[:7]:
                            best = key

        (
            best_loss,
            best_rmse,
            _,
            best_w,
            best_gamma,
            best_window,
            best_source,
            best_metrics,
        ) = best

        pca_weights[j] = best_w
        hf_gains[j] = best_gamma
        smooth_windows[j] = best_window
        hf_sources[j] = best_source

        print(
            f"{col:16s} PCA={best_w:.2f} HF={best_gamma:.2f} "
            f"SRC={best_source:3s} WIN={best_window:2d} "
            f"RMSE={best_rmse:.4f} Diff={best_metrics['diff_corr']:.4f} "
            f"Peak={best_metrics['peak_f1']:.4f} Vol={best_metrics['volatility_fit']:.4f} "
            f"Dir={best_metrics['direction_accuracy']:.4f} Loss={best_loss:.4f}"
        )

    candidate_pred = np.empty_like(ref_pred, dtype=np.float64)
    for j in range(n_targets):
        ref = ref_pred[:, :, j]
        pp = pca_pred[:, :, j]
        source = source_cube[str(hf_sources[j])][:, :, j]
        hf = endpoint_zero_highpass(source, int(smooth_windows[j]))
        candidate_pred[:, :, j] = _candidate(
            ref,
            pp,
            hf,
            float(pca_weights[j]),
            float(hf_gains[j]),
        )

    candidate_report = evaluate_multivariate(
        y_true,
        candidate_pred,
        last_values,
    )
    print_report("V8 source-aware candidate", candidate_report)

    ref_m = ref_report["mean"]
    cand_m = candidate_report["mean"]
    ref_rmse = float(ref_m["flat_rmse"])
    cand_rmse = float(cand_m["flat_rmse"])
    ref_proxy = float(ref_m["proxy_loss"])
    cand_proxy = float(cand_m["proxy_loss"])
    ref_trend = _trend_score(ref_report)
    cand_trend = _trend_score(candidate_report)

    rmse_ok = cand_rmse <= ref_rmse * (1.0 - MIN_GLOBAL_RMSE_IMPROVEMENT)
    proxy_ok = cand_proxy <= ref_proxy - MIN_PROXY_IMPROVEMENT
    trend_ok = cand_trend >= ref_trend + MIN_GLOBAL_TREND_SCORE_GAIN
    diff_ok = cand_m["diff_corr"] >= ref_m["diff_corr"] - MAX_GLOBAL_DIFF_DROP
    peak_ok = cand_m["peak_f1"] >= ref_m["peak_f1"] + MIN_GLOBAL_PEAK_GAIN
    vol_ok = cand_m["volatility_fit"] >= ref_m["volatility_fit"] + MIN_GLOBAL_VOL_GAIN
    any_pca = bool(np.any(pca_weights > 1e-12))
    passed = bool(
        rmse_ok and proxy_ok and trend_ok and diff_ok and peak_ok and vol_ok and any_pca
    )

    print("\n" + "=" * 124)
    print("V8 候选结论")
    print("=" * 124)
    print(f"V3 flat RMSE : {ref_rmse:.6f}")
    print(
        f"V8 flat RMSE : {cand_rmse:.6f} "
        f"({(cand_rmse/ref_rmse-1.0)*100:+.2f}%)"
    )
    print(f"V3 proxy loss: {ref_proxy:.6f}")
    print(
        f"V8 proxy loss: {cand_proxy:.6f} "
        f"({(cand_proxy/ref_proxy-1.0)*100:+.2f}%)"
    )
    print(f"V3 trendScore: {ref_trend:.6f}")
    print(
        f"V8 trendScore: {cand_trend:.6f} "
        f"({(cand_trend/ref_trend-1.0)*100:+.2f}%)"
    )
    print(f"V8 DiffCorr  : {cand_m['diff_corr']:.6f}")
    print(f"V8 PeakF1    : {cand_m['peak_f1']:.6f}")
    print(f"V8 VolFit    : {cand_m['volatility_fit']:.6f}")
    print(f"V8 DirAcc    : {cand_m['direction_accuracy']:.6f}")
    print(f"PCA weights  : {pca_weights.tolist()}")
    print(f"HF gains     : {hf_gains.tolist()}")
    print(f"HF sources   : {hf_sources.tolist()}")
    print(f"Smooth wins  : {smooth_windows.tolist()}")

    np.savez_compressed(
        CANDIDATE_PRED_PATH,
        pred_abs=candidate_pred.astype(np.float32),
        pca_weights=pca_weights,
        hf_gains=hf_gains,
        hf_sources=hf_sources,
        smooth_windows=smooth_windows,
    )

    candidate_config = dict(config)
    candidate_config.update(
        {
            "version": 8,
            "candidate_only": True,
            "trajectory_model": "pca_xgb_source_aware_hf_v1",
            "pca_blend_weights": pca_weights.tolist(),
            "v8_highpass_gains": hf_gains.tolist(),
            "v8_highpass_sources": hf_sources.tolist(),
            "v8_highpass_windows": smooth_windows.tolist(),
            "validation_rmse": cand_rmse,
            "validation_proxy_loss": cand_proxy,
            "validation_diff_corr": float(cand_m["diff_corr"]),
            "validation_peak_f1": float(cand_m["peak_f1"]),
            "validation_volatility_fit": float(cand_m["volatility_fit"]),
            "validation_direction_accuracy": float(cand_m["direction_accuracy"]),
            "validation_trend_score": cand_trend,
            "candidate_passed_local_gate": passed,
        }
    )
    with open(CANDIDATE_CONFIG_PATH, "wb") as f:
        pickle.dump(candidate_config, f)

    if passed:
        print("PASS V8 LOCAL GATE：Accuracy 与 Trend 同时提升，且 DiffCorr 安全闸通过。")
    else:
        print("REJECT V8 LOCAL GATE：继续保留 V3，V8 仅作为离线候选。")
    print(f"候选配置: {CANDIDATE_CONFIG_PATH}")
    print("本脚本不会覆盖 models/ensemble_config.pkl。")


if __name__ == "__main__":
    main()
