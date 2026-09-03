import numpy as np

from config import HORIZON, TARGET_COLUMNS
from find_best_weight import (
    apply_ensemble_config,
    evaluate_multivariate,
    load_existing_config,
    load_validation_arrays,
    print_report,
)


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    persistence = np.repeat(last_values[:, None, :], HORIZON, axis=1)

    config = load_existing_config()
    ensemble = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        config,
    )

    candidates = [
        ("Persistence baseline", persistence),
        ("LightGBM", pred_lgb),
        ("XGBoost", pred_xgb),
        ("Robust trend baseline", baseline),
        (f"Current ensemble version={config.get('version')}", ensemble),
    ]

    reports = {}
    for name, pred in candidates:
        report = evaluate_multivariate(y_true, pred, last_values)
        reports[name] = report
        print_report(name, report)

    print("\n" + "=" * 110)
    print("模型横向汇总（本地代理指标，不等于官网真实分数）")
    print("=" * 110)
    print(
        f"{'model':32s} {'flatRMSE':>11s} {'DiffCorr':>10s} "
        f"{'PeakF1':>9s} {'VolFit':>9s} {'DirAcc':>9s} {'ProxyLoss':>11s}"
    )
    print("-" * 110)

    for name, _ in candidates:
        m = reports[name]["mean"]
        print(
            f"{name:32s} {m['flat_rmse']:11.4f} {m['diff_corr']:10.4f} "
            f"{m['peak_f1']:9.4f} {m['volatility_fit']:9.4f} "
            f"{m['direction_accuracy']:9.4f} {m['proxy_loss']:11.4f}"
        )

    current = reports[candidates[-1][0]]["mean"]
    print("\n当前融合重点：")
    print(f"  flat RMSE : {current['flat_rmse']:.6f}")
    print(f"  DiffCorr  : {current['diff_corr']:.6f}")
    print(f"  PeakF1    : {current['peak_f1']:.6f}")
    print(f"  VolFit    : {current['volatility_fit']:.6f}")
    print(f"  DirAcc    : {current['direction_accuracy']:.6f}")
    print(f"  ProxyLoss : {current['proxy_loss']:.6f}  (越低越好)")
    print("\n官网真实评分仍以官网结果为准；这个脚本只用于本地选择更值得提交的版本。")


if __name__ == "__main__":
    main()
