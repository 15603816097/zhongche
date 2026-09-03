import numpy as np

from config import MODEL_DIR
from find_best_weight import (
    apply_ensemble_config,
    evaluate_multivariate,
    load_existing_config,
    load_validation_arrays,
    print_report,
)
from find_best_weight_v5 import (
    TREND_VAL_PATH,
    _trend_components,
    apply_v5_trend,
)


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    config = load_existing_config()

    base = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        config,
    )

    version = int(config.get("version", 1))
    final_pred = base

    if version >= 5:
        if not TREND_VAL_PATH.exists():
            raise FileNotFoundError(
                f"ensemble version={version} 已启用 V5，但缺少 {TREND_VAL_PATH}"
            )
        trend = np.load(TREND_VAL_PATH, allow_pickle=True)
        pred_step_diff = trend["pred_step_diff"].astype(np.float64)
        trend_linear, trend_shape = _trend_components(pred_step_diff)
        alpha = np.asarray(config.get("trend_shape_alpha", []), dtype=np.float64)
        beta = np.asarray(config.get("trend_level_beta", []), dtype=np.float64)
        final_pred = apply_v5_trend(
            base,
            last_values,
            trend_linear,
            trend_shape,
            alpha,
            beta,
        )

    report = evaluate_multivariate(y_true, final_pred, last_values)
    print_report(f"Final ensemble version={version}", report)

    m = report["mean"]
    print("\nV5 final summary")
    print(f"  version   : {version}")
    print(f"  flat RMSE : {m['flat_rmse']:.6f}")
    print(f"  DiffCorr  : {m['diff_corr']:.6f}")
    print(f"  PeakF1    : {m['peak_f1']:.6f}")
    print(f"  VolFit    : {m['volatility_fit']:.6f}")
    print(f"  DirAcc    : {m['direction_accuracy']:.6f}")
    print(f"  ProxyLoss : {m['proxy_loss']:.6f}")
    print("官网真实分数仍以官网评测为准。")


if __name__ == "__main__":
    main()
