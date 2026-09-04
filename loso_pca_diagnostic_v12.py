import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS, XGB_DEVICE, XGB_PARAMS
from find_best_weight import evaluate_multivariate, print_report
from src.dataset_builder import load_all_data, sample_weights


ARTIFACT_PATH = MODEL_DIR / "loso_pca_v12.npz"
PCA_COMPONENTS = 12


def _trend_score(report):
    m = report["mean"]
    diff_score = float(np.clip((m["diff_corr"] + 1.0) / 2.0, 0.0, 1.0))
    return float(
        0.45 * diff_score
        + 0.25 * np.clip(m["peak_f1"], 0.0, 1.0)
        + 0.20 * np.clip(m["volatility_fit"], 0.0, 1.0)
        + 0.10 * np.clip(m["direction_accuracy"], 0.0, 1.0)
    )


def _params():
    params = XGB_PARAMS.copy()
    params.update(
        {
            "n_estimators": int(os.getenv("V12_ESTIMATORS", "360")),
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 4.0,
            "subsample": 0.90,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.20,
            "reg_lambda": 1.8,
            "verbosity": 0,
        }
    )
    return params


def _delta_trajectories(y_abs, last_values):
    return np.asarray(y_abs, dtype=np.float64) - np.asarray(
        last_values, dtype=np.float64
    )[:, None, :]


def _fit_pcas(delta, train_idx):
    pcas = []
    explained = []
    n_components = min(PCA_COMPONENTS, len(train_idx), HORIZON)

    for j, col in enumerate(TARGET_COLUMNS):
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(delta[train_idx, :, j])
        pcas.append(pca)
        explained.append(float(np.sum(pca.explained_variance_ratio_)))
        print(
            f"    PCA {col:16s}: components={n_components:2d} "
            f"explained={explained[-1]:.4f}"
        )
    return pcas, explained


def _encode(delta, pcas):
    return np.concatenate(
        [pca.transform(delta[:, :, j]) for j, pca in enumerate(pcas)],
        axis=1,
    )


def _decode(scores, pcas):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    out = np.empty(
        (len(scores), HORIZON, len(TARGET_COLUMNS)),
        dtype=np.float64,
    )
    offset = 0
    for j, pca in enumerate(pcas):
        k = int(pca.n_components_)
        out[:, :, j] = pca.inverse_transform(scores[:, offset : offset + k])
        offset += k

    if offset != scores.shape[1]:
        raise RuntimeError(
            f"PCA score 维度不一致: used={offset}, actual={scores.shape[1]}"
        )
    return out


def _short(report):
    m = report["mean"]
    return (
        f"RMSE={m['flat_rmse']:.4f} meanRMSE={m['rmse']:.4f} "
        f"Diff={m['diff_corr']:.4f} Peak={m['peak_f1']:.4f} "
        f"Vol={m['volatility_fit']:.4f} Trend={_trend_score(report):.4f} "
        f"Proxy={m['proxy_loss']:.4f}"
    )


def main():
    print("=" * 124)
    print("V12 Leave-One-Sequence-Out PCA-XGBoost Generalization Diagnostic")
    print("每次完整留出 1 条 sequence，PCA/scaler/XGBoost 都只在另外 4 条 sequence 上拟合。")
    print("目的：检查当前 5 条序列内部验证是否过于乐观，构造更接近未知隐藏序列的评估。")
    print("本脚本只生成诊断产物，不修改 V8、API、callback 或 ensemble_config.pkl。")
    print("=" * 124)
    print(f"XGBoost device={XGB_DEVICE} estimators={_params()['n_estimators']}")

    bundle = load_all_data()
    weights = sample_weights(bundle)
    delta = _delta_trajectories(bundle.y_abs, bundle.last_values)
    sequences = sorted(np.unique(bundle.sequence_names).tolist())

    n = len(bundle.X)
    oof_pred = np.full_like(bundle.y_abs, np.nan, dtype=np.float64)
    fold_id = np.full(n, -1, dtype=np.int32)
    explained_by_fold = []

    total_started = time.perf_counter()

    for fold, heldout in enumerate(sequences):
        test_idx = np.where(bundle.sequence_names == heldout)[0]
        train_idx = np.where(bundle.sequence_names != heldout)[0]

        print("\n" + "-" * 124)
        print(
            f"[FOLD {fold + 1}/{len(sequences)}] heldout={heldout} "
            f"train={len(train_idx)} test={len(test_idx)}"
        )

        fold_started = time.perf_counter()
        pcas, explained = _fit_pcas(delta, train_idx)
        explained_by_fold.append(explained)
        scores = _encode(delta, pcas)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(bundle.X[train_idx])
        X_test = scaler_X.transform(bundle.X[test_idx])
        y_train = scaler_y.fit_transform(scores[train_idx])

        model = XGBRegressor(**_params())
        model.fit(X_train, y_train, sample_weight=weights[train_idx])

        pred_scaled = np.asarray(model.predict(X_test))
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled.reshape(len(test_idx), -1)
        pred_scores = scaler_y.inverse_transform(pred_scaled)
        pred_delta = _decode(pred_scores, pcas)
        pred_abs = pred_delta + bundle.last_values[test_idx, None, :]

        oof_pred[test_idx] = pred_abs
        fold_id[test_idx] = fold

        report = evaluate_multivariate(
            bundle.y_abs[test_idx],
            pred_abs,
            bundle.last_values[test_idx],
        )
        print(f"  fold result: {_short(report)}")

        boundary_idx = test_idx[
            bundle.starts[test_idx] == np.max(bundle.starts[test_idx])
        ]
        boundary_report = evaluate_multivariate(
            bundle.y_abs[boundary_idx],
            pred_abs[np.isin(test_idx, boundary_idx)],
            bundle.last_values[boundary_idx],
        )
        print(f"  heldout boundary: {_short(boundary_report)}")
        print(f"  fold seconds={time.perf_counter() - fold_started:.1f}")

    if np.any(fold_id < 0) or np.any(~np.isfinite(oof_pred)):
        raise RuntimeError("LOSO OOF prediction incomplete")

    y_true = bundle.y_abs.astype(np.float64)
    last = bundle.last_values.astype(np.float64)
    persistence = np.repeat(last[:, None, :], HORIZON, axis=1)
    baseline = bundle.baseline_abs.astype(np.float64)

    loso_report = evaluate_multivariate(y_true, oof_pred, last)
    persistence_report = evaluate_multivariate(y_true, persistence, last)
    baseline_report = evaluate_multivariate(y_true, baseline, last)

    print_report("V12 LOSO PCA-XGBoost OOF", loso_report)
    print_report("V12 LOSO persistence reference", persistence_report)
    print_report("V12 LOSO robust-trend baseline", baseline_report)

    # 只取每条 held-out sequence 的最后一个边界样本，共 5 个真正 unseen-sequence boundary。
    boundary_idx = []
    for seq in sequences:
        idx = np.where(bundle.sequence_names == seq)[0]
        boundary_idx.extend(idx[bundle.starts[idx] == np.max(bundle.starts[idx])].tolist())
    boundary_idx = np.asarray(boundary_idx, dtype=np.int64)

    boundary_loso = evaluate_multivariate(
        y_true[boundary_idx], oof_pred[boundary_idx], last[boundary_idx]
    )
    boundary_persist = evaluate_multivariate(
        y_true[boundary_idx], persistence[boundary_idx], last[boundary_idx]
    )
    boundary_baseline = evaluate_multivariate(
        y_true[boundary_idx], baseline[boundary_idx], last[boundary_idx]
    )

    print_report("V12 LOSO unseen-sequence boundary PCA-XGB", boundary_loso)
    print_report("V12 LOSO unseen-sequence boundary persistence", boundary_persist)
    print_report("V12 LOSO unseen-sequence boundary robust-trend", boundary_baseline)

    np.savez_compressed(
        ARTIFACT_PATH,
        pred_abs=oof_pred.astype(np.float32),
        y_abs=y_true.astype(np.float32),
        last_values=last.astype(np.float32),
        baseline_abs=baseline.astype(np.float32),
        sequence_names=bundle.sequence_names.astype(str),
        starts=bundle.starts.astype(np.int32),
        fold_id=fold_id,
        boundary_idx=boundary_idx,
        explained_variance=np.asarray(explained_by_fold, dtype=np.float64),
    )

    print("\n" + "=" * 124)
    print("V12 LOSO conclusion data")
    print("=" * 124)
    print(f"LOSO all       : {_short(loso_report)}")
    print(f"LOSO boundary  : {_short(boundary_loso)}")
    print(f"Robust baseline: {_short(baseline_report)}")
    print(f"artifact       : {ARTIFACT_PATH}")
    print(f"total seconds  : {time.perf_counter() - total_started:.1f}")
    print("\nInterpretation rule:")
    print("  1) 如果 LOSO 明显比当前 temporal validation 差，说明之前本地调参存在同序列过拟合。")
    print("  2) 如果 PCA 在 LOSO 仍明显胜 persistence/robust-trend，PCA 低频路线可继续。")
    print("  3) 下一版模型只允许在 temporal + LOSO 两套验证同时改善后才进入官网候选。")
    print("\nV12 is diagnostic only. Online V8 remains unchanged.")


if __name__ == "__main__":
    main()
