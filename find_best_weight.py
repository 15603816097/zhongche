import pickle

import numpy as np

from config import MODEL_DIR, TARGET_COLUMNS


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def direction_accuracy(y_true, y_pred, last_values):
    true_delta = y_true - last_values[:, None, :]
    pred_delta = y_pred - last_values[:, None, :]
    true_sign = np.sign(true_delta)
    pred_sign = np.sign(pred_delta)
    return float(np.mean(true_sign == pred_sign))


def main():
    lgb_path = MODEL_DIR / "val_pred_lgb.npz"
    xgb_path = MODEL_DIR / "val_pred_xgb.npz"

    if not lgb_path.exists() or not xgb_path.exists():
        raise FileNotFoundError(
            "请先运行 python src/trainer.py 和 "
            "python src/trainer_xgb.py"
        )

    lgb = np.load(lgb_path, allow_pickle=True)
    xgb = np.load(xgb_path, allow_pickle=True)

    y_true = lgb["y_abs"].astype(np.float64)
    last_values = lgb["last_values"].astype(np.float64)
    baseline = lgb["baseline_abs"].astype(np.float64)
    pred_lgb = lgb["pred_abs"].astype(np.float64)
    pred_xgb = xgb["pred_abs"].astype(np.float64)

    if not (
        y_true.shape == pred_lgb.shape == pred_xgb.shape == baseline.shape
    ):
        raise ValueError("LGB/XGB 验证预测 shape 不一致")

    n_targets = len(TARGET_COLUMNS)
    lgb_weights = np.zeros(n_targets, dtype=np.float64)
    baseline_weights = np.zeros(n_targets, dtype=np.float64)

    print("=" * 80)
    print("逐变量搜索 LGB/XGB/稳健趋势基线的融合权重")
    print("目标函数：85% 归一化RMSE + 15% 趋势方向误差")
    print("=" * 80)

    for j, col in enumerate(TARGET_COLUMNS):
        yt = y_true[:, :, j]
        pl = pred_lgb[:, :, j]
        px = pred_xgb[:, :, j]
        pb = baseline[:, :, j]
        last = last_values[:, j]

        scale = float(np.std(yt))
        if scale < 1e-8:
            scale = 1.0

        best = None

        for w_lgb in np.arange(0.0, 1.01, 0.1):
            ml = w_lgb * pl + (1.0 - w_lgb) * px

            for w_base in np.arange(0.0, 0.51, 0.1):
                pred = (1.0 - w_base) * ml + w_base * pb

                value_rmse = rmse(yt, pred)
                true_delta = yt - last[:, None]
                pred_delta = pred - last[:, None]
                dir_acc = float(
                    np.mean(np.sign(true_delta) == np.sign(pred_delta))
                )
                objective = (
                    0.85 * (value_rmse / scale)
                    + 0.15 * (1.0 - dir_acc)
                )

                candidate = (
                    objective,
                    value_rmse,
                    -dir_acc,
                    float(w_lgb),
                    float(w_base),
                )
                if best is None or candidate < best:
                    best = candidate

        _, best_rmse, neg_dir, best_lgb, best_base = best
        best_dir = -neg_dir
        lgb_weights[j] = best_lgb
        baseline_weights[j] = best_base

        print(
            f"{col:16s} "
            f"LGB={best_lgb:.2f} XGB={1-best_lgb:.2f} "
            f"BASE={best_base:.2f} "
            f"RMSE={best_rmse:.4f} "
            f"DirAcc={best_dir:.4f}"
        )

    ml = (
        lgb_weights.reshape(1, 1, -1) * pred_lgb
        + (1.0 - lgb_weights.reshape(1, 1, -1)) * pred_xgb
    )
    final_pred = (
        (1.0 - baseline_weights.reshape(1, 1, -1)) * ml
        + baseline_weights.reshape(1, 1, -1) * baseline
    )

    overall_rmse = rmse(y_true, final_pred)
    overall_dir = direction_accuracy(y_true, final_pred, last_values)

    config = {
        "version": 2,
        "lgb_weights": lgb_weights.tolist(),
        "baseline_weights": baseline_weights.tolist(),
        "target_columns": list(TARGET_COLUMNS),
        "validation_rmse": overall_rmse,
        "validation_direction_accuracy": overall_dir,
    }

    with open(MODEL_DIR / "ensemble_config.pkl", "wb") as f:
        pickle.dump(config, f)

    print("-" * 80)
    print(f"整体验证 RMSE: {overall_rmse:.4f}")
    print(f"整体趋势方向一致率: {overall_dir:.4f}")
    print(f"已保存: {MODEL_DIR / 'ensemble_config.pkl'}")


if __name__ == "__main__":
    main()
