import numpy as np

from config import MODEL_DIR, TARGET_COLUMNS
from find_best_weight import evaluate_multivariate, variable_metrics
from src.dataset_builder import load_all_data, temporal_train_val_indices


ARTIFACT_PATH = MODEL_DIR / "v14_robust_pca_diagnostic.npz"
ALPHA_GRID = np.asarray([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0], dtype=np.float64)

# V15 不重新训练，只利用 V14 已保存的 clean / robust 预测做快速混合诊断。
# 目的：保住 temporal accuracy，同时吸收 robust 模型在 LOSO 上的泛化收益。
MAX_TEMP_VAR_RMSE_DEGRADATION = 0.010
MAX_BOUNDARY_VAR_RMSE_DEGRADATION = 0.020
MAX_TEMP_GLOBAL_RMSE_DEGRADATION = 0.007
MIN_LOSO_GLOBAL_RMSE_IMPROVEMENT = 0.005
MAX_BOUNDARY_GLOBAL_RMSE_DEGRADATION = 0.010
EPS = 1e-12


def _short(report):
    m = report["mean"]
    return (
        f"RMSE={m['flat_rmse']:.4f} meanRMSE={m['rmse']:.4f} "
        f"Diff={m['diff_corr']:.4f} Peak={m['peak_f1']:.4f} "
        f"Vol={m['volatility_fit']:.4f} Proxy={m['proxy_loss']:.4f}"
    )


def _blend(clean, robust, alpha):
    return (1.0 - alpha) * clean + alpha * robust


def main():
    print("=" * 124)
    print("V15 Clean-Robust PCA Blend Diagnostic")
    print("只利用 V14 已有预测，快速搜索 clean / robust 混合比例；不重新训练，不修改线上 V8。")
    print("目标：temporal 基本不退，同时尽量保留 V14 在 LOSO 上的泛化收益。")
    print("=" * 124)

    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"缺少 {ARTIFACT_PATH}，请先运行 bash run_v14_robust.sh")

    data = np.load(ARTIFACT_PATH, allow_pickle=True)
    t_clean = data["temporal_clean_base"].astype(np.float64)
    t_rob = data["temporal_clean_robust"].astype(np.float64)
    l_clean = data["loso_clean_base"].astype(np.float64)
    l_rob = data["loso_clean_robust"].astype(np.float64)
    boundary_idx = data["boundary_idx"].astype(np.int64)

    bundle = load_all_data()
    _, val_idx = temporal_train_val_indices(bundle)

    if len(t_clean) != len(val_idx):
        raise RuntimeError(f"temporal prediction count mismatch: {len(t_clean)} vs {len(val_idx)}")
    if len(l_clean) != len(bundle.X):
        raise RuntimeError(f"LOSO prediction count mismatch: {len(l_clean)} vs {len(bundle.X)}")

    y_t = bundle.y_abs[val_idx].astype(np.float64)
    last_t = bundle.last_values[val_idx].astype(np.float64)
    y_l = bundle.y_abs.astype(np.float64)
    last_l = bundle.last_values.astype(np.float64)

    t_ref = evaluate_multivariate(y_t, t_clean, last_t)
    l_ref = evaluate_multivariate(y_l, l_clean, last_l)
    b_ref = evaluate_multivariate(y_l[boundary_idx], l_clean[boundary_idx], last_l[boundary_idx])

    print(f"Temporal clean reference : {_short(t_ref)}")
    print(f"LOSO clean reference     : {_short(l_ref)}")
    print(f"Boundary clean reference : {_short(b_ref)}")

    alphas = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)

    print("\n" + "=" * 124)
    print("Per-variable alpha search")
    print("=" * 124)

    for j, col in enumerate(TARGET_COLUMNS):
        yt = y_t[:, :, j]
        lt = last_t[:, j]
        yl = y_l[:, :, j]
        ll = last_l[:, j]
        yb = y_l[boundary_idx, :, j]
        lb = last_l[boundary_idx, j]

        t0 = variable_metrics(yt, t_clean[:, :, j], lt)
        l0 = variable_metrics(yl, l_clean[:, :, j], ll)
        b0 = variable_metrics(yb, l_clean[boundary_idx, :, j], lb)

        best = (
            0.55 * 1.0 + 0.25 * 1.0 + 0.20 * 1.0,
            0.0,
            t0,
            l0,
            b0,
        )

        for alpha in ALPHA_GRID:
            tp = _blend(t_clean[:, :, j], t_rob[:, :, j], alpha)
            lp = _blend(l_clean[:, :, j], l_rob[:, :, j], alpha)
            bp = lp[boundary_idx]

            tm = variable_metrics(yt, tp, lt)
            lm = variable_metrics(yl, lp, ll)
            bm = variable_metrics(yb, bp, lb)

            if tm["rmse"] > t0["rmse"] * (1.0 + MAX_TEMP_VAR_RMSE_DEGRADATION) + EPS:
                continue
            if bm["rmse"] > b0["rmse"] * (1.0 + MAX_BOUNDARY_VAR_RMSE_DEGRADATION) + EPS:
                continue

            score = (
                0.55 * (lm["rmse"] / max(l0["rmse"], EPS))
                + 0.25 * (bm["rmse"] / max(b0["rmse"], EPS))
                + 0.20 * (tm["rmse"] / max(t0["rmse"], EPS))
            )
            candidate = (score, float(alpha), tm, lm, bm)
            if candidate[0] < best[0] - 1e-12 or (
                abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] < best[1]
            ):
                best = candidate

        score, alpha, tm, lm, bm = best
        alphas[j] = alpha
        print(
            f"{col:16s} alpha={alpha:4.2f} "
            f"T_RMSE={tm['rmse']:.4f} ({(tm['rmse']/t0['rmse']-1)*100:+.2f}%) "
            f"L_RMSE={lm['rmse']:.4f} ({(lm['rmse']/l0['rmse']-1)*100:+.2f}%) "
            f"B_RMSE={bm['rmse']:.4f} ({(bm['rmse']/b0['rmse']-1)*100:+.2f}%) "
            f"score={score:.4f}"
        )

    alpha3 = alphas.reshape(1, 1, -1)
    t_pred = (1.0 - alpha3) * t_clean + alpha3 * t_rob
    l_pred = (1.0 - alpha3) * l_clean + alpha3 * l_rob

    t_cand = evaluate_multivariate(y_t, t_pred, last_t)
    l_cand = evaluate_multivariate(y_l, l_pred, last_l)
    b_cand = evaluate_multivariate(y_l[boundary_idx], l_pred[boundary_idx], last_l[boundary_idx])

    print("\n" + "=" * 124)
    print("V15 blend conclusion")
    print("=" * 124)
    print(f"alphas                   : {alphas.tolist()}")
    print(f"temporal reference       : {_short(t_ref)}")
    print(f"temporal candidate       : {_short(t_cand)}")
    print(f"LOSO reference           : {_short(l_ref)}")
    print(f"LOSO candidate           : {_short(l_cand)}")
    print(f"boundary reference       : {_short(b_ref)}")
    print(f"boundary candidate       : {_short(b_cand)}")

    tr0 = t_ref["mean"]["flat_rmse"]
    tr1 = t_cand["mean"]["flat_rmse"]
    lr0 = l_ref["mean"]["flat_rmse"]
    lr1 = l_cand["mean"]["flat_rmse"]
    br0 = b_ref["mean"]["flat_rmse"]
    br1 = b_cand["mean"]["flat_rmse"]

    temporal_ok = tr1 <= tr0 * (1.0 + MAX_TEMP_GLOBAL_RMSE_DEGRADATION)
    loso_ok = lr1 <= lr0 * (1.0 - MIN_LOSO_GLOBAL_RMSE_IMPROVEMENT)
    boundary_ok = br1 <= br0 * (1.0 + MAX_BOUNDARY_GLOBAL_RMSE_DEGRADATION)
    proxy_ok = (
        t_cand["mean"]["proxy_loss"] <= t_ref["mean"]["proxy_loss"] + 0.002
        and l_cand["mean"]["proxy_loss"] <= l_ref["mean"]["proxy_loss"]
        and b_cand["mean"]["proxy_loss"] <= b_ref["mean"]["proxy_loss"]
    )
    any_alpha = bool(np.any(alphas > 1e-12))
    passed = bool(temporal_ok and loso_ok and boundary_ok and proxy_ok and any_alpha)

    print("\nGates:")
    print(f"  temporal RMSE gate : {temporal_ok} ({(tr1/tr0-1)*100:+.2f}%)")
    print(f"  LOSO RMSE gate     : {loso_ok} ({(lr1/lr0-1)*100:+.2f}%)")
    print(f"  boundary RMSE gate : {boundary_ok} ({(br1/br0-1)*100:+.2f}%)")
    print(f"  proxy gate         : {proxy_ok}")

    out_path = MODEL_DIR / "v15_blend_v14_diagnostic.npz"
    np.savez_compressed(
        out_path,
        alphas=alphas,
        temporal_pred=t_pred.astype(np.float32),
        loso_pred=l_pred.astype(np.float32),
        boundary_idx=boundary_idx,
        passed=np.asarray(passed),
    )
    print(f"artifact              : {out_path}")

    if passed:
        print("PASS V15 BLEND GATE：值得进一步做 stress 验证/正式训练，但当前仍不要改线上 V8。")
    else:
        print("REJECT V15 BLEND GATE：今天停止 robust 路线，继续使用当前官网 V8。")


if __name__ == "__main__":
    main()
