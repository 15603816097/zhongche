import pickle

import numpy as np

from config import MODEL_DIR
from find_best_weight import evaluate_multivariate, load_validation_arrays, print_report


V8_PRED_PATH = MODEL_DIR / "val_pred_candidate_v8.npz"
V10_PRED_PATH = MODEL_DIR / "val_pred_candidate_v10.npz"
V10_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v10.pkl"


def _load_pred(path):
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    data = np.load(path, allow_pickle=True)
    return data["pred_abs"].astype(np.float64)


def main():
    y_true, last_values, _, _, _ = load_validation_arrays()
    v8_pred = _load_pred(V8_PRED_PATH)
    v10_pred = _load_pred(V10_PRED_PATH)

    v8_report = evaluate_multivariate(y_true, v8_pred, last_values)
    v10_report = evaluate_multivariate(y_true, v10_pred, last_values)

    print_report("V8 official baseline proxy", v8_report)
    print_report("V10 confidence-gated template candidate", v10_report)

    if not V10_CONFIG_PATH.exists():
        raise FileNotFoundError(f"缺少 {V10_CONFIG_PATH}")
    with open(V10_CONFIG_PATH, "rb") as f:
        cfg = pickle.load(f)

    m = v10_report["mean"]
    print("\n" + "=" * 104)
    print("V10 final summary")
    print("=" * 104)
    print(f"local gate passed : {cfg.get('candidate_passed_local_gate')}")
    print(f"temperature       : {cfg.get('template_match_temperature')}")
    print(f"matcher accuracy  : {cfg.get('template_match_accuracy')}")
    print(f"shape gains       : {cfg.get('template_shape_gains')}")
    print(f"conf thresholds   : {cfg.get('template_conf_thresholds')}")
    print(f"conf powers       : {cfg.get('template_conf_powers')}")
    print(f"candidate RMSE    : {m['flat_rmse']:.6f}")
    print(f"candidate DiffCorr: {m['diff_corr']:.6f}")
    print(f"candidate PeakF1  : {m['peak_f1']:.6f}")
    print(f"candidate VolFit  : {m['volatility_fit']:.6f}")
    print(f"candidate DirAcc  : {m['direction_accuracy']:.6f}")
    print(f"candidate Proxy   : {m['proxy_loss']:.6f}")
    print("\nV10 is offline-only. Current online V8 remains unchanged.")


if __name__ == "__main__":
    main()
