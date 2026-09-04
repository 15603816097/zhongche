import pickle

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, MODEL_DIR, TARGET_COLUMNS
from find_best_weight import evaluate_multivariate, print_report, variable_metrics
from src.data_cleaner import clean_sequence, clean_target_sequence
from src.dataset_builder import load_all_data, temporal_train_val_indices
from src.feature_engineer import extract_inference_features
from src.inference import load_models, predict_future
from src.legacy_shape_teacher import (
    build_fixed_future_shape_bank,
    classification_diagnostics,
    classifier_probabilities,
    confidence_gate,
    predict_fixed_future_shape,
    save_teacher,
    train_sequence_classifier,
)


V11_CONFIG_PATH = MODEL_DIR / "ensemble_config_candidate_v11.pkl"
V11_PRED_PATH = MODEL_DIR / "boundary_stress_candidate_v11.npz"
V11_TEACHER_PATH = MODEL_DIR / "legacy_shape_teacher_v11.pkl"

TEMPERATURE_GRID = (0.55, 0.75, 1.00, 1.35, 1.75)
GAIN_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)
THRESHOLD_GRID = (-1.0, 0.40, 0.55, 0.70, 0.80, 0.90)
POWER_GRID = (1.0, 2.0)
USE_MARGIN_GRID = (False, True)
EPS = 1e-12


def _trend_score(report):
    m = report["mean"]
    diff_score = float(np.clip((m["diff_corr"] + 1.0) / 2.0, 0.0, 1.0))
    return float(
        0.45 * diff_score
        + 0.25 * np.clip(m["peak_f1"], 0.0, 1.0)
        + 0.20 * np.clip(m["volatility_fit"], 0.0, 1.0)
        + 0.10 * np.clip(m["direction_accuracy"], 0.0, 1.0)
    )


def _temper(prob, temperature):
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-9, 1.0)
    z = np.log(p) / max(float(temperature), 1e-3)
    z -= np.max(z, axis=1, keepdims=True)
    out = np.exp(z)
    return out / np.maximum(np.sum(out, axis=1, keepdims=True), EPS)


def _stress_histories(history, seq_seed):
    """构造轻量输入扰动，目标 future 不变；用于模拟官方鲁棒性方向。"""
    base = history.copy()
    values = base[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    recent = values[-144:]
    scale = np.maximum(np.std(recent, axis=0), 1e-6)
    diff_scale = np.maximum(np.std(np.diff(recent, axis=0), axis=0), 1e-6)
    rng = np.random.default_rng(20260904 + int(seq_seed))

    cases = [("clean", base.copy())]

    noise = base.copy()
    noise_vals = values + rng.normal(0.0, 0.18 * diff_scale, size=values.shape)
    noise[TARGET_COLUMNS] = noise_vals
    cases.append(("noise", noise))

    missing = base.copy()
    mask = rng.random(values.shape) < 0.06
    miss_vals = values.copy()
    miss_vals[mask] = np.nan
    missing[TARGET_COLUMNS] = miss_vals
    cases.append(("missing_random", missing))

    block = base.copy()
    block_vals = values.copy()
    start = max(0, len(block_vals) - 96 + (int(seq_seed) * 7) % 48)
    block_vals[start:start + 12, :] = np.nan
    block[TARGET_COLUMNS] = block_vals
    cases.append(("missing_block", block))

    bias = base.copy()
    signs = np.where(np.arange(len(TARGET_COLUMNS)) % 2 == 0, 1.0, -1.0)
    bias[TARGET_COLUMNS] = values + signs[None, :] * (0.04 * scale)[None, :]
    cases.append(("bias", bias))

    drift = base.copy()
    ramp = np.linspace(0.0, 1.0, len(values), dtype=np.float64)[:, None]
    drift[TARGET_COLUMNS] = values + ramp * (0.06 * scale)[None, :]
    cases.append(("drift", drift))

    return cases


def _variable_objective(metrics_all, metrics_clean):
    rmse_all = metrics_all["rmse"] / max(metrics_all["persistence_rmse"], EPS)
    rmse_clean = metrics_clean["rmse"] / max(metrics_clean["persistence_rmse"], EPS)
    diff_loss = float(np.clip((1.0 - metrics_all["diff_corr"]) / 2.0, 0.0, 1.0))
    peak_loss = 1.0 - float(np.clip(metrics_all["peak_f1"], 0.0, 1.0))
    vol_loss = 1.0 - float(np.clip(metrics_all["volatility_fit"], 0.0, 1.0))
    return float(
        0.30 * rmse_all
        + 0.20 * rmse_clean
        + 0.20 * diff_loss
        + 0.15 * peak_loss
        + 0.15 * vol_loss
    )


def _build_boundary_stress(classes):
    features = []
    y_true = []
    anchors = []
    v8_pred = []
    seq_ids = []
    case_names = []

    class_to_id = {name: i for i, name in enumerate(classes)}

    for seq_i, seq in enumerate(classes):
        seq_dir = DATA_DIR / seq
        history = pd.read_csv(seq_dir / "history.csv")
        future = clean_target_sequence(pd.read_csv(seq_dir / "future.csv"))
        target = future.iloc[:HORIZON][TARGET_COLUMNS].to_numpy(dtype=np.float64)

        for case_name, perturbed in _stress_histories(history, seq_i + 1):
            cleaned = clean_sequence(perturbed)
            feat = extract_inference_features(cleaned)
            pred = predict_future(perturbed)
            last = cleaned.iloc[-1][TARGET_COLUMNS].to_numpy(dtype=np.float64)

            features.append(feat)
            y_true.append(target)
            anchors.append(last)
            v8_pred.append(pred)
            seq_ids.append(class_to_id[seq])
            case_names.append(case_name)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(y_true, dtype=np.float64),
        np.asarray(anchors, dtype=np.float64),
        np.asarray(v8_pred, dtype=np.float64),
        np.asarray(seq_ids, dtype=np.int32),
        np.asarray(case_names, dtype="<U24"),
    )


def main():
    _, _, _, _, online_config = load_models()
    if int(online_config.get("version", -1)) != 8:
        raise RuntimeError(
            f"V11 边界测试要求当前 online config 为 V8，实际={online_config.get('version')}"
        )

    bundle = load_all_data()
    train_idx, val_idx = temporal_train_val_indices(bundle)
    classifier, classes, y_ids = train_sequence_classifier(
        bundle.X,
        bundle.sequence_names,
        train_idx,
    )

    val_prob = classifier_probabilities(classifier, bundle.X[val_idx])
    val_diag = classification_diagnostics(val_prob, y_ids[val_idx])

    bank = build_fixed_future_shape_bank(classes)
    save_teacher(V11_TEACHER_PATH, classifier, bank)

    print("=" * 128)
    print("V11 Legacy Future-Shape Teacher")
    print("显式复刻最初 55.97 版本的核心：先识别 sequence，再调用该 sequence 固定 future.csv 的 endpoint-zero 形状。")
    print("只做边界 + 输入扰动离线诊断；不会覆盖 V8。")
    print("=" * 128)
    print(f"tail-val sequence accuracy : {val_diag['accuracy']:.4f}")
    print(f"tail-val mean top1         : {val_diag['mean_top1']:.4f}")
    print(f"tail-val mean margin       : {val_diag['mean_margin']:.4f}")

    Xb, yb, anchors, v8_pred, seq_ids, case_names = _build_boundary_stress(classes)
    base_prob = classifier_probabilities(classifier, Xb)
    boundary_diag = classification_diagnostics(base_prob, seq_ids)
    clean_mask = case_names == "clean"

    print(f"boundary+stress match acc  : {boundary_diag['accuracy']:.4f}")
    print(f"boundary+stress mean top1  : {boundary_diag['mean_top1']:.4f}")
    print(f"clean boundary match acc   : {float(np.mean(boundary_diag['pred'][clean_mask] == seq_ids[clean_mask])):.4f}")
    unique, counts = np.unique(case_names, return_counts=True)
    print("stress cases               :", dict(zip(unique.tolist(), counts.tolist())))

    ref_all = evaluate_multivariate(yb, v8_pred, anchors)
    ref_clean = evaluate_multivariate(yb[clean_mask], v8_pred[clean_mask], anchors[clean_mask])
    print_report("V8 boundary+stress baseline", ref_all)
    print_report("V8 clean-boundary baseline", ref_clean)

    best_global = None

    for temperature in TEMPERATURE_GRID:
        prob = _temper(base_prob, temperature)
        shape = predict_fixed_future_shape(prob, bank)
        params = []
        candidate = np.empty_like(v8_pred)

        for j, col in enumerate(TARGET_COLUMNS):
            yt_all = yb[:, :, j]
            an_all = anchors[:, j]
            ref_j = v8_pred[:, :, j]
            shape_j = shape[:, :, j]

            ref_m_all = variable_metrics(yt_all, ref_j, an_all)
            ref_m_clean = variable_metrics(
                yt_all[clean_mask],
                ref_j[clean_mask],
                an_all[clean_mask],
            )

            best = None
            for use_margin in USE_MARGIN_GRID:
                for threshold in THRESHOLD_GRID:
                    for power in POWER_GRID:
                        gate = confidence_gate(
                            prob,
                            threshold=threshold,
                            power=power,
                            use_margin=use_margin,
                        )
                        for gain in GAIN_GRID:
                            pred_j = ref_j + (
                                float(gain)
                                * gate[:, None]
                                * shape_j
                            )
                            m_all = variable_metrics(yt_all, pred_j, an_all)
                            m_clean = variable_metrics(
                                yt_all[clean_mask],
                                pred_j[clean_mask],
                                an_all[clean_mask],
                            )

                            if m_clean["rmse"] > ref_m_clean["rmse"] * 1.004 + EPS:
                                continue
                            if m_all["rmse"] > ref_m_all["rmse"] * 1.008 + EPS:
                                continue
                            if m_all["diff_corr"] < ref_m_all["diff_corr"] - 0.0010:
                                continue

                            key = (
                                _variable_objective(m_all, m_clean),
                                m_all["rmse"],
                                -m_all["peak_f1"],
                                -m_all["volatility_fit"],
                            )
                            if best is None or key < best[0]:
                                best = (
                                    key,
                                    float(gain),
                                    float(threshold),
                                    float(power),
                                    bool(use_margin),
                                    gate,
                                    m_all,
                                    m_clean,
                                )

            if best is None:
                raise RuntimeError(f"{col} 没找到候选")

            _, gain, threshold, power, use_margin, gate, m_all, m_clean = best
            candidate[:, :, j] = ref_j + gain * gate[:, None] * shape_j
            params.append((gain, threshold, power, use_margin, float(np.mean(gate))))

        report_all = evaluate_multivariate(yb, candidate, anchors)
        report_clean = evaluate_multivariate(
            yb[clean_mask], candidate[clean_mask], anchors[clean_mask]
        )
        trend_all = _trend_score(report_all)
        trend_clean = _trend_score(report_clean)
        ref_trend_all = _trend_score(ref_all)
        ref_trend_clean = _trend_score(ref_clean)

        all_rmse_ratio = report_all["mean"]["flat_rmse"] / ref_all["mean"]["flat_rmse"]
        clean_rmse_ratio = report_clean["mean"]["flat_rmse"] / ref_clean["mean"]["flat_rmse"]

        safe = bool(
            all_rmse_ratio <= 1.006
            and clean_rmse_ratio <= 1.003
            and report_all["mean"]["diff_corr"] >= ref_all["mean"]["diff_corr"] - 0.0003
        )
        gain_score = (
            (trend_all - ref_trend_all)
            + 0.5 * (trend_clean - ref_trend_clean)
            - 0.25 * max(0.0, all_rmse_ratio - 1.0)
        )

        print(
            f"TEMP={temperature:.2f} safe={safe} "
            f"allRMSE={report_all['mean']['flat_rmse']:.6f} "
            f"cleanRMSE={report_clean['mean']['flat_rmse']:.6f} "
            f"Diff={report_all['mean']['diff_corr']:.6f} "
            f"Peak={report_all['mean']['peak_f1']:.6f} "
            f"Vol={report_all['mean']['volatility_fit']:.6f} "
            f"TrendGain={(trend_all/ref_trend_all-1.0)*100:+.2f}%"
        )

        rank_key = (
            0 if safe else 1,
            -gain_score,
            report_all["mean"]["flat_rmse"],
        )
        if best_global is None or rank_key < best_global[0]:
            best_global = (
                rank_key,
                float(temperature),
                prob,
                shape,
                candidate,
                params,
                report_all,
                report_clean,
                safe,
            )

    (
        _, temperature, prob, shape, candidate, params,
        cand_all, cand_clean, safe,
    ) = best_global

    ref_trend_all = _trend_score(ref_all)
    cand_trend_all = _trend_score(cand_all)
    ref_trend_clean = _trend_score(ref_clean)
    cand_trend_clean = _trend_score(cand_clean)

    final_pass = bool(
        safe
        and cand_trend_all >= ref_trend_all * 1.02
        and cand_trend_clean >= ref_trend_clean * 1.01
        and cand_all["mean"]["peak_f1"] >= ref_all["mean"]["peak_f1"] + 0.005
        and cand_all["mean"]["volatility_fit"] >= ref_all["mean"]["volatility_fit"] + 0.008
    )

    print_report("V11 selected boundary+stress candidate", cand_all)
    print_report("V11 selected clean-boundary candidate", cand_clean)

    print("\n" + "=" * 128)
    print("V11 candidate conclusion")
    print("=" * 128)
    print(f"temperature                 : {temperature}")
    print(f"tail-val match accuracy     : {val_diag['accuracy']:.4f}")
    print(f"boundary+stress match acc   : {boundary_diag['accuracy']:.4f}")
    print(f"V8 all flat RMSE            : {ref_all['mean']['flat_rmse']:.6f}")
    print(f"V11 all flat RMSE           : {cand_all['mean']['flat_rmse']:.6f}")
    print(f"V8 all trendScore           : {ref_trend_all:.6f}")
    print(f"V11 all trendScore          : {cand_trend_all:.6f} ({(cand_trend_all/ref_trend_all-1)*100:+.2f}%)")
    print(f"V8 clean trendScore         : {ref_trend_clean:.6f}")
    print(f"V11 clean trendScore        : {cand_trend_clean:.6f} ({(cand_trend_clean/ref_trend_clean-1)*100:+.2f}%)")
    print(f"V11 all DiffCorr            : {cand_all['mean']['diff_corr']:.6f}")
    print(f"V11 all PeakF1              : {cand_all['mean']['peak_f1']:.6f}")
    print(f"V11 all VolFit              : {cand_all['mean']['volatility_fit']:.6f}")
    print("per-variable params:")
    for col, p in zip(TARGET_COLUMNS, params):
        gain, threshold, power, use_margin, mean_gate = p
        print(
            f"  {col:16s} gain={gain:.2f} thr={threshold:.2f} "
            f"pow={power:.0f} margin={use_margin} mean_gate={mean_gate:.3f}"
        )
    print("PASS V11 DIAGNOSTIC" if final_pass else "REJECT V11 DIAGNOSTIC")
    print("注意：V11 使用提供的 future.csv 作为 legacy shape teacher，仅用于复刻最初 55.97 机制并做离线诊断。")
    print("即使 PASS，也先不要改 API；先根据结果决定是否值得做正式在线版本。")

    np.savez_compressed(
        V11_PRED_PATH,
        y_true=yb.astype(np.float32),
        anchors=anchors.astype(np.float32),
        v8_pred=v8_pred.astype(np.float32),
        candidate_pred=candidate.astype(np.float32),
        probabilities=prob.astype(np.float32),
        sequence_ids=seq_ids,
        case_names=case_names,
        clean_mask=clean_mask,
    )

    config = {
        "version": 11,
        "candidate_only": True,
        "base_version": 8,
        "trajectory_model": "v8_plus_legacy_fixed_future_shape_teacher",
        "temperature": temperature,
        "classes": classes,
        "parameters": {
            col: {
                "gain": p[0],
                "threshold": p[1],
                "power": p[2],
                "use_margin": p[3],
                "mean_gate": p[4],
            }
            for col, p in zip(TARGET_COLUMNS, params)
        },
        "tail_val_match_accuracy": val_diag["accuracy"],
        "boundary_stress_match_accuracy": boundary_diag["accuracy"],
        "diagnostic_passed": final_pass,
        "all_flat_rmse": cand_all["mean"]["flat_rmse"],
        "all_diff_corr": cand_all["mean"]["diff_corr"],
        "all_peak_f1": cand_all["mean"]["peak_f1"],
        "all_volatility_fit": cand_all["mean"]["volatility_fit"],
        "all_trend_score": cand_trend_all,
        "clean_trend_score": cand_trend_clean,
    }
    with open(V11_CONFIG_PATH, "wb") as f:
        pickle.dump(config, f)

    print(f"candidate config: {V11_CONFIG_PATH}")
    print(f"teacher model   : {V11_TEACHER_PATH}")
    print(f"stress artifact : {V11_PRED_PATH}")
    print("models/ensemble_config.pkl 未修改。")


if __name__ == "__main__":
    main()
