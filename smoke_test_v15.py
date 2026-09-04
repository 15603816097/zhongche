import pickle
import time

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.inference import load_models, predict_future
from src.v8_runtime import v15_alphas


CONFIG_PATH = MODEL_DIR / "ensemble_config.pkl"
HISTORY_PATH = DATA_DIR / "sequence0001" / "history.csv"


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    print("=" * 94)
    print("V15 online inference smoke test")
    print("=" * 94)
    print(f"ensemble version: {config.get('version')}")
    print(f"trajectory_model: {config.get('trajectory_model')}")

    if int(config.get("version", -1)) != 15:
        raise RuntimeError("当前 ensemble_config 不是 V15，请先 python activate_v15.py")
    if str(config.get("trajectory_model", "")) != "pca_xgb_source_aware_hf_robust_blend_v15":
        raise RuntimeError("trajectory_model 不是 V15 robust blend")

    alphas = v15_alphas(config)
    print(f"robust alphas : {alphas.tolist()}")

    history = pd.read_csv(HISTORY_PATH)
    missing = [c for c in TARGET_COLUMNS if c not in history.columns]
    if missing:
        raise RuntimeError(f"history.csv 缺列: {missing}")
    history = history[TARGET_COLUMNS].copy()

    started = time.perf_counter()
    load_models()
    print(f"model preload: {time.perf_counter() - started:.3f}s")

    totals = []
    for run in range(1, 4):
        pred, timings = predict_future(history, return_timings=True)
        expected = (HORIZON, len(TARGET_COLUMNS))
        if pred.shape != expected:
            raise RuntimeError(f"输出 shape 错误: {pred.shape} vs {expected}")
        if not np.all(np.isfinite(pred)):
            raise RuntimeError("V15 输出存在 NaN/Inf")

        totals.append(float(timings["total"]))
        print(
            f"run={run} total={timings['total']:.3f}s "
            f"lgb={timings['lgb']:.3f}s "
            f"xgb={timings['xgb']:.3f}s "
            f"pca_total={timings.get('pca', 0.0):.3f}s "
            f"feature={timings['feature']:.3f}s "
            f"lgb_outputs={timings['lgb_outputs']} "
            f"v8_outputs={timings.get('v8_outputs', 0)}"
        )

    print("-" * 94)
    print(f"mean total (3 runs): {np.mean(totals):.3f}s")
    print(f"max total  (3 runs): {np.max(totals):.3f}s")
    if float(np.mean(totals[1:])) > 5.0:
        raise RuntimeError("V15 warm inference > 5s，提交前需要检查 runtime")
    print("V15 smoke test PASS")


if __name__ == "__main__":
    main()
