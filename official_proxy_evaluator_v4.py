import numpy as np

from config import MODEL_DIR
from find_best_weight import (
    apply_ensemble_config,
    evaluate_multivariate,
    load_existing_config,
    load_validation_arrays,
    print_report,
)
from find_best_weight_v4 import apply_v4_pattern


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    config = load_existing_config()

    v3_core = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        config,
    )

    pattern_path = MODEL_DIR / "val_pred_pattern_v4.npz"
    pattern = None
    if pattern_path.exists():
        with np.load(pattern_path, allow_pickle=False) as data:
            pattern = data["pred_abs"].astype(np.float64)

    if int(config.get("version", 1)) >= 4 and pattern is not None:
        final_pred = apply_v4_pattern(v3_core, pattern, config)
    else:
        final_pred = v3_core

    candidates = [
        ("LightGBM", pred_lgb),
        ("XGBoost", pred_xgb),
        ("Robust linear trend", baseline),
        ("V3 core", v3_core),
    ]
    if pattern is not None:
        candidates.append(("Adaptive pattern trend", pattern))
    candidates.append((f"Current ensemble version={config.get('version')}", final_pred))

    reports = {}
    for name, pred in candidates:
        report = evaluate_multivariate(y_true, pred, last_values)
        reports[name] = report
        print_report(name, report)

    print("\n" + "=" * 116)
    print("V4 横向汇总（本地代理指标，不等于官网真实分数）")
    print("=" * 116)
    print(
        f"{'model':32s} {'flatRMSE':>11s} {'DiffCorr':>10s} "
        f"{'PeakF1':>9s} {'VolFit':>9s} {'DirAcc':>9s} {'ProxyLoss':>11s}"
    )
    print("-" * 116)

    for name, _ in candidates:
        m = reports[name]["mean"]
        print(
            f"{name:32s} {m['flat_rmse']:11.4f} {m['diff_corr']:10.4f} "
            f"{m['peak_f1']:9.4f} {m['volatility_fit']:9.4f} "
            f"{m['direction_accuracy']:9.4f} {m['proxy_loss']:11.4f}"
        )

    current = reports[candidates[-1][0]]["mean"]
    print("\n当前最终融合：")
    print(f"  ensemble version: {config.get('version')}")
    print(f"  flat RMSE       : {current['flat_rmse']:.6f}")
    print(f"  DiffCorr        : {current['diff_corr']:.6f}")
    print(f"  PeakF1          : {current['peak_f1']:.6f}")
    print(f"  VolFit          : {current['volatility_fit']:.6f}")
    print(f"  DirAcc          : {current['direction_accuracy']:.6f}")
    print(f"  ProxyLoss       : {current['proxy_loss']:.6f}")
    print("\n官网真实评分仍以官网结果为准。")


if __name__ == "__main__":
    main()
