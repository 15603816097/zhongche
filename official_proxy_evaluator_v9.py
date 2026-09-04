import pickle

import numpy as np

from config import MODEL_DIR
from find_best_weight import evaluate_multivariate, load_validation_arrays, print_report


V8_PRED_PATH = MODEL_DIR / "val_pred_candidate_v8.npz"
V9_PRED_PATH = MODEL_DIR / "val_pred_candidate_v9.npz"
V9_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v9.pkl"


def _load_pred(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)["pred_abs"].astype(np.float64)


def main():
    y_true, last_values, _, _, _ = load_validation_arrays()
    v8_pred = _load_pred(V8_PRED_PATH)
    v9_pred = _load_pred(V9_PRED_PATH)

    if not V9_CONFIG_PATH.exists():
        raise FileNotFoundError(V9_CONFIG_PATH)
    with open(V9_CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    v8_report = evaluate_multivariate(y_true, v8_pred, last_values)
    v9_report = evaluate_multivariate(y_true, v9_pred, last_values)

    print_report("V8 official baseline proxy", v8_report)
    print_report("V9 causal template-shape candidate", v9_report)

    a = v8_report["mean"]
    b = v9_report["mean"]

    print("\n" + "=" * 100)
    print("V9 final summary")
    print("=" * 100)
    print(f"local gate passed : {config.get('candidate_passed_local_gate')}")
    print(f"matcher accuracy  : {config.get('template_match_accuracy')}")
    print(f"shape gains       : {config.get('template_shape_gains')}")
    print(f"V8 flat RMSE      : {a['flat_rmse']:.6f}")
    print(f"V9 flat RMSE      : {b['flat_rmse']:.6f}")
    print(f"V8 DiffCorr       : {a['diff_corr']:.6f}")
    print(f"V9 DiffCorr       : {b['diff_corr']:.6f}")
    print(f"V8 PeakF1         : {a['peak_f1']:.6f}")
    print(f"V9 PeakF1         : {b['peak_f1']:.6f}")
    print(f"V8 VolFit         : {a['volatility_fit']:.6f}")
    print(f"V9 VolFit         : {b['volatility_fit']:.6f}")
    print(f"V8 ProxyLoss      : {a['proxy_loss']:.6f}")
    print(f"V9 ProxyLoss      : {b['proxy_loss']:.6f}")
    print("\nV9 is offline-only. Current online V8 remains unchanged.")


if __name__ == "__main__":
    main()
