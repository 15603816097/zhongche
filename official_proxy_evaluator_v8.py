import pickle

import numpy as np

from config import MODEL_DIR
from find_best_weight import (
    apply_ensemble_config,
    evaluate_multivariate,
    load_existing_config,
    load_validation_arrays,
    print_report,
)


CANDIDATE_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v8.pkl"
CANDIDATE_PRED_PATH = MODEL_DIR / "val_pred_candidate_v8.npz"


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    online_config = load_existing_config()
    online_pred = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        online_config,
    )
    online_report = evaluate_multivariate(y_true, online_pred, last_values)
    print_report(
        f"Current online ensemble version={online_config.get('version')}",
        online_report,
    )

    if not CANDIDATE_CONFIG_PATH.exists() or not CANDIDATE_PRED_PATH.exists():
        raise FileNotFoundError(
            "缺少 V8 candidate 文件，请先运行 python find_best_weight_v8.py"
        )

    with open(CANDIDATE_CONFIG_PATH, "rb") as f:
        candidate_config = pickle.load(f)
    data = np.load(CANDIDATE_PRED_PATH, allow_pickle=True)
    candidate_pred = data["pred_abs"].astype(np.float64)

    candidate_report = evaluate_multivariate(
        y_true,
        candidate_pred,
        last_values,
    )
    print_report("V8 source-aware candidate", candidate_report)

    m = candidate_report["mean"]
    print("\n" + "=" * 104)
    print("V8 final candidate summary")
    print("=" * 104)
    print(f"online version          : {online_config.get('version')}")
    print(f"candidate version       : {candidate_config.get('version')}")
    print(
        f"local gate passed       : "
        f"{candidate_config.get('candidate_passed_local_gate')}"
    )
    print(f"PCA blend weights       : {candidate_config.get('pca_blend_weights')}")
    print(f"HF gains                : {candidate_config.get('v8_highpass_gains')}")
    print(f"HF sources              : {candidate_config.get('v8_highpass_sources')}")
    print(f"HF smooth windows       : {candidate_config.get('v8_highpass_windows')}")
    print(f"candidate flat RMSE     : {m['flat_rmse']:.6f}")
    print(f"candidate DiffCorr      : {m['diff_corr']:.6f}")
    print(f"candidate PeakF1        : {m['peak_f1']:.6f}")
    print(f"candidate VolFit        : {m['volatility_fit']:.6f}")
    print(f"candidate DirAcc        : {m['direction_accuracy']:.6f}")
    print(f"candidate ProxyLoss     : {m['proxy_loss']:.6f}")
    print("\n当前线上 ensemble_config.pkl 没有被 V8 修改。")


if __name__ == "__main__":
    main()
