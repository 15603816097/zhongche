from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from v9_analog_multiscale_diagnostic import (
    AnalogConfig,
    CONTEXTS,
    DESC_POINTS,
    KS,
    MODES,
    ROLLING_STRIDE,
    SELF_GUARDS,
    EPS,
    build_bank,
    evaluate_rows,
    multiscale_descriptor,
    robust_scale,
)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "artifacts" / "full_arch" / "expert_cache"
OUT_DIR = ROOT / "artifacts" / "full_arch" / "confidence_ood"
SEQUENCES = [f"sequence{i:04d}" for i in range(1, 6)]
N_TARGETS = len(TARGET_COLUMNS)


def load_history(name: str) -> np.ndarray:
    path = DATA_DIR / name / "history.csv"
    df = pd.read_csv(path)
    raw_missing = int(df[TARGET_COLUMNS].isna().sum().sum())
    clean = clean_sequence(df)
    arr = clean[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name}: non-finite values remain after cleaning")
    return arr, raw_missing


def load_truth_v8(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE_DIR / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage 2 cache: {path}")
    with np.load(path) as z:
        truth = np.asarray(z["truth"], dtype=np.float64)
        v8 = np.asarray(z["v8"], dtype=np.float64)
    return truth, v8


def _effective_k(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w * w))
    return float(1.0 / max(denom, EPS))


def _reference_neighbor_distance(
    descs: np.ndarray,
    src_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    """Cross-source reference distances for in-domain calibration.

    Each bank descriptor searches only descriptors coming from another sequence.
    This is stricter than same-sequence matching and therefore a useful OOD baseline.
    """
    n = len(descs)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        legal = src_ids != src_ids[i]
        idx = np.where(legal)[0]
        if len(idx) < k:
            continue
        diff = descs[idx] - descs[i : i + 1]
        dist = np.sqrt(np.mean(diff * diff, axis=1))
        nearest = np.partition(dist, k - 1)[:k]
        out[i] = float(np.median(nearest))
    return out[np.isfinite(out)]


def _calibration_stats(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"median": 1.0, "q75": 1.0, "q90": 1.0, "q95": 1.0}
    return {
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
    }


def _distance_ood(distance: float, cal: dict) -> float:
    med = float(cal["median"])
    q90 = float(cal["q90"])
    width = max(q90 - med, 1e-6)
    z = max(0.0, (float(distance) - med) / width)
    # 0=in-domain-like, asymptotically approaches 1 for distant samples.
    return float(1.0 - math.exp(-z))


def _query_analog(
    histories: list[np.ndarray],
    query_idx: int,
    target_idx: int,
    cfg: AnalogConfig,
    bank,
    calibration: dict,
) -> tuple[np.ndarray, dict]:
    descs, deltas, src_ids, future_ends = bank
    query_h = histories[query_idx]
    query_ctx = query_h[-cfg.context :]
    qdesc = multiscale_descriptor(query_ctx, target_idx, cfg.mode)

    legal = np.ones(len(descs), dtype=bool)
    self_mask = src_ids == query_idx
    legal[self_mask] = (
        future_ends[self_mask] <= len(query_h) - int(cfg.self_guard)
    )
    legal_idx = np.where(legal)[0]
    if len(legal_idx) < cfg.k:
        raise RuntimeError(
            f"not enough analogs target={TARGET_COLUMNS[target_idx]} cfg={cfg}"
        )

    diff = descs[legal_idx] - qdesc.reshape(1, -1)
    dist = np.sqrt(np.mean(diff * diff, axis=1))
    order = np.argsort(dist)[: cfg.k]
    chosen = legal_idx[order]
    chosen_dist = dist[order]

    denom = max(float(np.median(chosen_dist)), 1e-6)
    weights = np.exp(-chosen_dist / denom)
    if not np.isfinite(weights).all() or float(weights.sum()) <= EPS:
        weights = np.ones_like(chosen_dist)
    weights = weights / weights.sum()

    chosen_delta = deltas[chosen]
    dz = np.sum(weights[:, None] * chosen_delta, axis=0)
    anchor = float(query_ctx[-1, target_idx])
    qscale = robust_scale(query_ctx[:, target_idx])
    pred = anchor + dz * qscale
    pred = np.nan_to_num(pred, nan=anchor, posinf=anchor, neginf=anchor)

    median_distance = float(np.median(chosen_dist))
    mean_distance = float(np.mean(chosen_dist))
    nearest_distance = float(np.min(chosen_dist))
    diversity = float(
        len(np.unique(src_ids[chosen]))
        / max(1, min(cfg.k, len(histories)))
    )
    eff_k = _effective_k(weights)

    # Neighbour trajectory disagreement is measured in normalized-delta space.
    weighted_var = np.sum(
        weights[:, None] * (chosen_delta - dz.reshape(1, -1)) ** 2,
        axis=0,
    )
    trajectory_dispersion = float(np.sqrt(np.mean(weighted_var)))
    endpoint_signs = np.sign(chosen_delta[:, -1])
    endpoint_sign_agreement = float(abs(np.sum(weights * endpoint_signs)))

    distance_ood = _distance_ood(median_distance, calibration)
    distance_conf = 1.0 - distance_ood
    dispersion_conf = float(math.exp(-0.75 * max(0.0, trajectory_dispersion)))
    diversity_factor = 0.60 + 0.40 * diversity
    sign_factor = 0.75 + 0.25 * endpoint_sign_agreement
    confidence = float(
        np.clip(
            distance_conf
            * dispersion_conf
            * diversity_factor
            * sign_factor,
            0.0,
            1.0,
        )
    )

    info = {
        "nearest_distance": nearest_distance,
        "median_distance": median_distance,
        "mean_distance": mean_distance,
        "distance_ood": distance_ood,
        "trajectory_dispersion": trajectory_dispersion,
        "source_diversity": diversity,
        "effective_k": eff_k,
        "endpoint_sign_agreement": endpoint_sign_agreement,
        "confidence": confidence,
    }
    return pred, info


def _robust_z(value: float, reference: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return 0.0
    med = float(np.median(ref))
    q25, q75 = np.quantile(ref, [0.25, 0.75])
    sigma = max(float((q75 - q25) / 1.349), float(np.std(ref)) * 0.25, 1e-6)
    return float(abs(value - med) / sigma)


def _regime_ood_features(histories: list[np.ndarray], query_idx: int, target_idx: int) -> dict:
    q = histories[query_idx][:, target_idx]
    recent = q[-48:]
    q_level = float(np.median(recent))
    q_vol = float(np.std(np.diff(recent))) if len(recent) > 2 else 0.0
    q_slope = float((recent[-1] - recent[0]) / max(len(recent) - 1, 1))

    levels = []
    vols = []
    slopes = []
    for src_idx, h in enumerate(histories):
        x = h[:, target_idx]
        for end in range(48, len(x) - HORIZON + 1, ROLLING_STRIDE):
            seg = x[end - 48 : end]
            levels.append(float(np.median(seg)))
            vols.append(float(np.std(np.diff(seg))) if len(seg) > 2 else 0.0)
            slopes.append(float((seg[-1] - seg[0]) / max(len(seg) - 1, 1)))

    level_z = _robust_z(q_level, np.asarray(levels))
    vol_z = _robust_z(q_vol, np.asarray(vols))
    slope_z = _robust_z(q_slope, np.asarray(slopes))
    regime_ood = float(
        np.clip(
            0.40 * (1.0 - math.exp(-level_z / 3.0))
            + 0.35 * (1.0 - math.exp(-vol_z / 3.0))
            + 0.25 * (1.0 - math.exp(-slope_z / 3.0)),
            0.0,
            1.0,
        )
    )
    return {
        "level_z": level_z,
        "volatility_z": vol_z,
        "slope_z": slope_z,
        "regime_ood": regime_ood,
    }


def _eval_blend(
    truth: np.ndarray,
    v8: np.ndarray,
    analog: np.ndarray,
    anchor: float,
    alpha: float,
) -> dict:
    cand = (1.0 - alpha) * v8 + alpha * analog
    b = evaluate_rows(
        truth.reshape(1, -1),
        v8.reshape(1, -1),
        np.asarray([anchor], dtype=np.float64),
    )
    c = evaluate_rows(
        truth.reshape(1, -1),
        cand.reshape(1, -1),
        np.asarray([anchor], dtype=np.float64),
    )
    return {
        "rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
        "proxy_gain": float(
            (b["proxy_loss"] - c["proxy_loss"])
            / max(abs(b["proxy_loss"]), EPS)
        ),
        "trend_gain": float(c["trend_core"] - b["trend_core"]),
    }


def main() -> int:
    if not (CACHE_DIR / "manifest.json").is_file():
        raise FileNotFoundError("Stage 2 cache missing; run Stage 2 first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = []
    raw_missing = {}
    truths = []
    v8s = []
    anchors = []

    for name in SEQUENCES:
        h, miss = load_history(name)
        truth, v8 = load_truth_v8(name)
        histories.append(h)
        raw_missing[name] = miss
        truths.append(truth)
        v8s.append(v8)
        anchors.append(h[-1].copy())

    truth_arr = np.stack(truths, axis=0)
    v8_arr = np.stack(v8s, axis=0)
    anchor_arr = np.stack(anchors, axis=0)

    print("=" * 118)
    print("FULL ARCHITECTURE - STAGE 4 ANALOG CONFIDENCE + OOD")
    print("=" * 118)
    print("sequences       :", SEQUENCES)
    print("contexts        :", CONTEXTS)
    print("k values        :", KS)
    print("modes           :", MODES)
    print("self guards     :", SELF_GUARDS)
    print("descriptor pts  :", DESC_POINTS)
    print("raw missing     :", raw_missing)
    print("data policy     : analog bank uses cleaned history.csv only; future.csv is evaluation only")

    banks = {}
    calibrations = {}
    for context in CONTEXTS:
        for mode in MODES:
            for j, target in enumerate(TARGET_COLUMNS):
                bank = build_bank(histories, context, j, mode)
                banks[(context, mode, j)] = bank
                descs, _, src_ids, _ = bank
                for k in KS:
                    ref = _reference_neighbor_distance(descs, src_ids, k)
                    calibrations[(context, mode, j, k)] = _calibration_stats(ref)

    configs = []
    pred_list = []
    conf_list = []
    ood_list = []
    dispersion_list = []
    diversity_list = []
    effk_list = []
    sign_list = []
    distance_list = []
    diagnostic = {}

    total = len(CONTEXTS) * len(KS) * len(MODES) * len(SELF_GUARDS)
    done = 0
    for context in CONTEXTS:
        for k in KS:
            for mode in MODES:
                for guard in SELF_GUARDS:
                    cfg = AnalogConfig(context, k, mode, guard)
                    cfg_key = cfg.key()
                    configs.append(
                        {
                            "context": context,
                            "k": k,
                            "mode": mode,
                            "self_guard": guard,
                            "key": cfg_key,
                        }
                    )
                    pred = np.empty_like(truth_arr)
                    conf = np.zeros((len(SEQUENCES), N_TARGETS), dtype=np.float64)
                    ood = np.zeros_like(conf)
                    disp = np.zeros_like(conf)
                    diversity = np.zeros_like(conf)
                    effk = np.zeros_like(conf)
                    sign_agree = np.zeros_like(conf)
                    med_dist = np.zeros_like(conf)

                    for i in range(len(SEQUENCES)):
                        for j in range(N_TARGETS):
                            cal = calibrations[(context, mode, j, k)]
                            p, info = _query_analog(
                                histories,
                                i,
                                j,
                                cfg,
                                banks[(context, mode, j)],
                                cal,
                            )
                            pred[i, :, j] = p
                            conf[i, j] = info["confidence"]
                            ood[i, j] = info["distance_ood"]
                            disp[i, j] = info["trajectory_dispersion"]
                            diversity[i, j] = info["source_diversity"]
                            effk[i, j] = info["effective_k"]
                            sign_agree[i, j] = info["endpoint_sign_agreement"]
                            med_dist[i, j] = info["median_distance"]

                    # A tiny fixed blend is used only to test whether confidence aligns
                    # with actual usefulness. It is NOT a production choice.
                    rows = {}
                    for j, target in enumerate(TARGET_COLUMNS):
                        per_seq = []
                        for i, name in enumerate(SEQUENCES):
                            ev = _eval_blend(
                                truth_arr[i, :, j],
                                v8_arr[i, :, j],
                                pred[i, :, j],
                                float(anchor_arr[i, j]),
                                0.05,
                            )
                            ev.update(
                                {
                                    "sequence": name,
                                    "confidence": float(conf[i, j]),
                                    "distance_ood": float(ood[i, j]),
                                }
                            )
                            per_seq.append(ev)
                        rows[target] = per_seq
                    diagnostic[cfg_key] = rows

                    pred_list.append(pred)
                    conf_list.append(conf)
                    ood_list.append(ood)
                    dispersion_list.append(disp)
                    diversity_list.append(diversity)
                    effk_list.append(effk)
                    sign_list.append(sign_agree)
                    distance_list.append(med_dist)

                    done += 1
                    print(
                        f"analog confidence [{done:02d}/{total:02d}] {cfg_key} "
                        f"mean_conf={conf.mean():.3f} mean_ood={ood.mean():.3f}",
                        flush=True,
                    )

    analog_pred = np.stack(pred_list, axis=0)
    confidence = np.stack(conf_list, axis=0)
    distance_ood = np.stack(ood_list, axis=0)
    trajectory_dispersion = np.stack(dispersion_list, axis=0)
    source_diversity = np.stack(diversity_list, axis=0)
    effective_k = np.stack(effk_list, axis=0)
    endpoint_sign_agreement = np.stack(sign_list, axis=0)
    median_distance = np.stack(distance_list, axis=0)

    # Regime/OOD features do not depend on analog config.
    regime = np.zeros((len(SEQUENCES), N_TARGETS, 4), dtype=np.float64)
    for i in range(len(SEQUENCES)):
        for j in range(N_TARGETS):
            r = _regime_ood_features(histories, i, j)
            regime[i, j] = [
                r["level_z"],
                r["volatility_z"],
                r["slope_z"],
                r["regime_ood"],
            ]

    # Combined OOD = analog distance OOD + regime shift. We save it for every
    # analog config so Stage 5 can gate a chosen analog candidate directly.
    combined_ood = np.clip(
        0.65 * distance_ood
        + 0.35 * regime[None, :, :, 3],
        0.0,
        1.0,
    )

    npz_path = OUT_DIR / "analog_confidence_ood.npz"
    np.savez_compressed(
        npz_path,
        truth=truth_arr,
        v8=v8_arr,
        anchors=anchor_arr,
        analog_pred=analog_pred,
        confidence=confidence,
        distance_ood=distance_ood,
        combined_ood=combined_ood,
        trajectory_dispersion=trajectory_dispersion,
        source_diversity=source_diversity,
        effective_k=effective_k,
        endpoint_sign_agreement=endpoint_sign_agreement,
        median_distance=median_distance,
        regime_features=regime,
    )

    # Compact ranking: configurations where confidence is positively associated
    # with fixed-alpha proxy gain are useful gating candidates.
    ranking = []
    for cidx, cfg in enumerate(configs):
        key = cfg["key"]
        for j, target in enumerate(TARGET_COLUMNS):
            gains = np.asarray(
                [r["proxy_gain"] for r in diagnostic[key][target]], dtype=np.float64
            )
            confs = confidence[cidx, :, j]
            oods = combined_ood[cidx, :, j]
            if np.std(confs) > 1e-9 and np.std(gains) > 1e-9:
                corr = float(np.corrcoef(confs, gains)[0, 1])
            else:
                corr = 0.0
            ranking.append(
                {
                    "config_index": cidx,
                    "config": key,
                    "target": target,
                    "mean_confidence": float(np.mean(confs)),
                    "mean_combined_ood": float(np.mean(oods)),
                    "confidence_gain_corr": corr,
                    "positive_gain_005": int(np.sum(gains > 0.0)),
                    "mean_proxy_gain_005": float(np.mean(gains)),
                    "worst_proxy_gain_005": float(np.min(gains)),
                }
            )

    ranking.sort(
        key=lambda r: (
            r["positive_gain_005"],
            r["confidence_gain_corr"],
            r["mean_proxy_gain_005"],
            r["worst_proxy_gain_005"],
        ),
        reverse=True,
    )

    manifest = {
        "stage": 4,
        "sequences": SEQUENCES,
        "target_columns": list(TARGET_COLUMNS),
        "horizon": HORIZON,
        "configs": configs,
        "raw_missing_by_sequence": raw_missing,
        "regime_feature_order": [
            "level_z",
            "volatility_z",
            "slope_z",
            "regime_ood",
        ],
        "confidence_definition": (
            "cross-sequence normalized descriptor distance × neighbour trajectory "
            "agreement × source diversity × endpoint-sign agreement"
        ),
        "combined_ood_definition": (
            "0.65 * analog_distance_ood + 0.35 * regime_ood"
        ),
        "npz": str(npz_path.relative_to(ROOT)),
        "top_ranking": ranking[:40],
        "important_note": (
            "The 0.05 analog blend is diagnostic only. Stage 5 performs LOSO "
            "target×horizon gate search; no candidate is enabled here."
        ),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 118)
    print("TOP CONFIDENCE/GATING CANDIDATES (diagnostic alpha=0.05 only)")
    print("=" * 118)
    for r in ranking[:20]:
        print(
            f"{r['target']:15s} {r['config']:48s} "
            f"pos={r['positive_gain_005']}/5 "
            f"mean_gain={100*r['mean_proxy_gain_005']:+6.2f}% "
            f"corr={r['confidence_gain_corr']:+.3f} "
            f"conf={r['mean_confidence']:.3f} ood={r['mean_combined_ood']:.3f}"
        )

    print("\nnpz      :", npz_path)
    print("manifest :", manifest_path)
    print("STAGE 4 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
