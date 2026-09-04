import time

import numpy as np

from config import MODEL_DIR
from find_best_weight import evaluate_multivariate
from src.dataset_builder import load_all_data, sample_weights, temporal_train_val_indices
from v14_robust_pca_diagnostic import (
    PERTURBATIONS,
    _fit_pair,
    _fit_pcas,
    _predict_with_pcas,
    _repeat_targets,
    _short,
    stress_features,
)


V15_PATH = MODEL_DIR / "v15_blend_v14_diagnostic.npz"
OUT_PATH = MODEL_DIR / "v16_stress_blend_diagnostic.npz"
EPS = 1e-12

# V16 是 V15 上线前最后一道 stress gate。
MAX_TEMP_STRESS_DEGRADATION = 0.005
MIN_LOSO_STRESS_IMPROVEMENT = 0.005
MAX_BOUNDARY_STRESS_DEGRADATION = 0.005
MAX_PROXY_DEGRADATION = 0.001


def _blend(clean, robust, alphas):
    a = np.asarray(alphas, dtype=np.float64).reshape(1, 1, -1)
    return (1.0 - a) * clean + a * robust


def _ratio(new, old):
    return float(new / max(old, EPS) - 1.0)


def main():
    print("=" * 124)
    print("V16 Final Stress Validation for V15 Blend")
    print("固定使用 V15 已选 alpha，不再调参；重新生成 temporal/LOSO/boundary stress，检查泛化与鲁棒性。")
    print("本脚本不修改线上 V8、API、callback 或 ensemble_config.pkl。")
    print("=" * 124)

    if not V15_PATH.exists():
        raise FileNotFoundError(f"缺少 {V15_PATH}，请先运行 bash run_v15_blend.sh")

    v15 = np.load(V15_PATH, allow_pickle=True)
    if not bool(v15["passed"].item()):
        raise RuntimeError("V15 local gate 未通过，不应继续跑 V16")
    alphas = v15["alphas"].astype(np.float64)
    print(f"locked alphas: {alphas.tolist()}")

    bundle = load_all_data()
    weights = sample_weights(bundle)
    train_idx, val_idx = temporal_train_val_indices(bundle)
    delta_all = (
        bundle.y_abs.astype(np.float64)
        - bundle.last_values.astype(np.float64)[:, None, :]
    )

    # ------------------------------------------------------------------
    # A. Temporal stress: alpha 完全锁死，不再搜索。
    # ------------------------------------------------------------------
    print("\n[A] Temporal stress")
    t_started = time.perf_counter()
    pcas_t = _fit_pcas(delta_all, train_idx)
    pair_t = _fit_pair(bundle, train_idx, pcas_t, weights, seed=14001)

    stress_X_t, _ = stress_features(bundle.X[val_idx], seed=14101)
    y_stress_t, last_stress_t = _repeat_targets(
        bundle.y_abs[val_idx], bundle.last_values[val_idx], len(PERTURBATIONS)
    )
    t_base = _predict_with_pcas(
        pair_t["clean"], pcas_t, stress_X_t, last_stress_t
    )
    t_rob = _predict_with_pcas(
        pair_t["robust"], pcas_t, stress_X_t, last_stress_t
    )
    t_blend = _blend(t_base, t_rob, alphas)

    t_base_r = evaluate_multivariate(y_stress_t, t_base, last_stress_t)
    t_blend_r = evaluate_multivariate(y_stress_t, t_blend, last_stress_t)
    print(f"  temporal stress reference: {_short(t_base_r)}")
    print(f"  temporal stress V15 blend: {_short(t_blend_r)}")
    print(f"  temporal seconds={time.perf_counter()-t_started:.1f}")

    # ------------------------------------------------------------------
    # B. LOSO stress + unseen boundary stress.
    # ------------------------------------------------------------------
    print("\n[B] LOSO + boundary stress")
    sequences = sorted(np.unique(bundle.sequence_names).tolist())
    loso_base_all = []
    loso_blend_all = []
    loso_y_all = []
    loso_last_all = []

    boundary_base_all = []
    boundary_blend_all = []
    boundary_y_all = []
    boundary_last_all = []

    started = time.perf_counter()
    for fold, heldout in enumerate(sequences):
        test_idx = np.where(bundle.sequence_names == heldout)[0]
        train_fold = np.where(bundle.sequence_names != heldout)[0]
        print("\n" + "-" * 124)
        print(
            f"[FOLD {fold+1}/{len(sequences)}] heldout={heldout} "
            f"train={len(train_fold)} test={len(test_idx)}"
        )
        fold_started = time.perf_counter()

        pcas = _fit_pcas(delta_all, train_fold)
        pair = _fit_pair(bundle, train_fold, pcas, weights, seed=14200 + fold)

        stress_X, _ = stress_features(bundle.X[test_idx], seed=14300 + fold)
        y_stress, last_stress = _repeat_targets(
            bundle.y_abs[test_idx], bundle.last_values[test_idx], len(PERTURBATIONS)
        )
        base = _predict_with_pcas(pair["clean"], pcas, stress_X, last_stress)
        robust = _predict_with_pcas(pair["robust"], pcas, stress_X, last_stress)
        blend = _blend(base, robust, alphas)

        loso_base_all.append(base)
        loso_blend_all.append(blend)
        loso_y_all.append(y_stress)
        loso_last_all.append(last_stress)

        bidx = test_idx[
            bundle.starts[test_idx] == np.max(bundle.starts[test_idx])
        ]
        bX, _ = stress_features(bundle.X[bidx], seed=14400 + fold)
        by, blast = _repeat_targets(
            bundle.y_abs[bidx], bundle.last_values[bidx], len(PERTURBATIONS)
        )
        bbase = _predict_with_pcas(pair["clean"], pcas, bX, blast)
        brob = _predict_with_pcas(pair["robust"], pcas, bX, blast)
        bblend = _blend(bbase, brob, alphas)

        boundary_base_all.append(bbase)
        boundary_blend_all.append(bblend)
        boundary_y_all.append(by)
        boundary_last_all.append(blast)

        fold_base_r = evaluate_multivariate(y_stress, base, last_stress)
        fold_blend_r = evaluate_multivariate(y_stress, blend, last_stress)
        print(f"  stress ref  : {_short(fold_base_r)}")
        print(f"  stress blend: {_short(fold_blend_r)}")
        print(f"  fold seconds={time.perf_counter()-fold_started:.1f}")

    loso_base = np.concatenate(loso_base_all, axis=0)
    loso_blend = np.concatenate(loso_blend_all, axis=0)
    loso_y = np.concatenate(loso_y_all, axis=0)
    loso_last = np.concatenate(loso_last_all, axis=0)

    b_base = np.concatenate(boundary_base_all, axis=0)
    b_blend = np.concatenate(boundary_blend_all, axis=0)
    b_y = np.concatenate(boundary_y_all, axis=0)
    b_last = np.concatenate(boundary_last_all, axis=0)

    l_base_r = evaluate_multivariate(loso_y, loso_base, loso_last)
    l_blend_r = evaluate_multivariate(loso_y, loso_blend, loso_last)
    b_base_r = evaluate_multivariate(b_y, b_base, b_last)
    b_blend_r = evaluate_multivariate(b_y, b_blend, b_last)

    print("\n" + "=" * 124)
    print("V16 final stress conclusion")
    print("=" * 124)
    print(f"alphas                    : {alphas.tolist()}")
    print(f"temporal stress reference : {_short(t_base_r)}")
    print(f"temporal stress candidate : {_short(t_blend_r)}")
    print(f"LOSO stress reference     : {_short(l_base_r)}")
    print(f"LOSO stress candidate     : {_short(l_blend_r)}")
    print(f"boundary stress reference : {_short(b_base_r)}")
    print(f"boundary stress candidate : {_short(b_blend_r)}")

    tr0 = t_base_r["mean"]["flat_rmse"]
    tr1 = t_blend_r["mean"]["flat_rmse"]
    lr0 = l_base_r["mean"]["flat_rmse"]
    lr1 = l_blend_r["mean"]["flat_rmse"]
    br0 = b_base_r["mean"]["flat_rmse"]
    br1 = b_blend_r["mean"]["flat_rmse"]

    temporal_ok = tr1 <= tr0 * (1.0 + MAX_TEMP_STRESS_DEGRADATION)
    loso_ok = lr1 <= lr0 * (1.0 - MIN_LOSO_STRESS_IMPROVEMENT)
    boundary_ok = br1 <= br0 * (1.0 + MAX_BOUNDARY_STRESS_DEGRADATION)
    proxy_ok = (
        t_blend_r["mean"]["proxy_loss"] <= t_base_r["mean"]["proxy_loss"] + MAX_PROXY_DEGRADATION
        and l_blend_r["mean"]["proxy_loss"] <= l_base_r["mean"]["proxy_loss"] + MAX_PROXY_DEGRADATION
        and b_blend_r["mean"]["proxy_loss"] <= b_base_r["mean"]["proxy_loss"] + MAX_PROXY_DEGRADATION
    )
    passed = bool(temporal_ok and loso_ok and boundary_ok and proxy_ok)

    print("\nGates:")
    print(f"  temporal stress gate : {temporal_ok} ({_ratio(tr1, tr0)*100:+.2f}%)")
    print(f"  LOSO stress gate     : {loso_ok} ({_ratio(lr1, lr0)*100:+.2f}%)")
    print(f"  boundary stress gate : {boundary_ok} ({_ratio(br1, br0)*100:+.2f}%)")
    print(f"  proxy gate           : {proxy_ok}")

    np.savez_compressed(
        OUT_PATH,
        alphas=alphas,
        temporal_stress_base=t_base.astype(np.float32),
        temporal_stress_candidate=t_blend.astype(np.float32),
        loso_stress_base=loso_base.astype(np.float32),
        loso_stress_candidate=loso_blend.astype(np.float32),
        boundary_stress_base=b_base.astype(np.float32),
        boundary_stress_candidate=b_blend.astype(np.float32),
        passed=np.asarray(passed),
    )
    print(f"artifact               : {OUT_PATH}")
    print(f"LOSO total seconds     : {time.perf_counter()-started:.1f}")

    if passed:
        print("PASS V16 FINAL STRESS GATE：V15 值得进入正式全量训练与线上接入。")
    else:
        print("REJECT V16 FINAL STRESS GATE：停止 robust 路线，继续提交当前官网 V8。")


if __name__ == "__main__":
    main()
