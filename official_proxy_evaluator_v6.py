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


CANDIDATE_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v6.pkl"
CANDIDATE_PRED_PATH = MODEL_DIR / "val_pred_candidate_v6.npz"
PCA_VAL_PATH = MODEL_DIR / "val_pred_pca_xgb.npz"


def main():
    y_true, last_values, baseline, pred_lgb, pred_xgb = load_validation_arrays()
    current_config = load_existing_config()

    current_pred = apply_ensemble_config(
        pred_lgb,
        pred_xgb,
        baseline,
        last_values,
        current_config,
    )
    current_report = evaluate_multivariate(y_true, current_pred, last_values)
    print_report(
        f"Current online ensemble version={current_config.get('version')}",
        current_report,
    )

    if PCA_VAL_PATH.exists():
        pca = np.load(PCA_VAL_PATH, allow_pickle=True)["pred_abs"].astype(np.float64)
        pca_report = evaluate_multivariate(y_true, pca, last_values)
        print_report("Low-Rank PCA XGBoost raw", pca_report)

    if not CANDIDATE_CONFIG_PATH.exists() or not CANDIDATE_PRED_PATH.exists():
        print("\nV6 candidate 尚未生成，请先运行 python find_best_weight_v6.py")
        return

    with open(CANDIDATE_CONFIG_PATH, "rb") as f:
        candidate_config = pickle.load(f)
    candidate_pred = np.load(CANDIDATE_PRED_PATH, allow_pickle=True)["pred_abs"].astype(np.float64)
    candidate_report = evaluate_multivariate(y_true, candidate_pred, last_values)
    print_report("V6 PCA-blend candidate", candidate_report)

    print("\n" + "=" * 100)
    print("V6 final candidate summary")
    print("=" * 100)
    print(f"online version          : {current_config.get('version')}")
    print(f"candidate version       : {candidate_config.get('version')}")
    print(f"local gate passed       : {candidate_config.get('candidate_passed_local_gate')}")
    print(f"PCA blend weights       : {candidate_config.get('pca_blend_weights')}")
    print(f"candidate flat RMSE     : {candidate_report['mean']['flat_rmse']:.6f}")
    print(f"candidate DiffCorr      : {candidate_report['mean']['diff_corr']:.6f}")
    print(f"candidate PeakF1        : {candidate_report['mean']['peak_f1']:.6f}")
    print(f"candidate VolFit        : {candidate_report['mean']['volatility_fit']:.6f}")
    print(f"candidate DirAcc        : {candidate_report['mean']['direction_accuracy']:.6f}")
    print(f"candidate ProxyLoss     : {candidate_report['mean']['proxy_loss']:.6f}")
    print("\n当前线上 ensemble_config.pkl 未被 V6 修改，官网提交仍然是原来的安全版本。")


if __name__ == "__main__":
    main()
