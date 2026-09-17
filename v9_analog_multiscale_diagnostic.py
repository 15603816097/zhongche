from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, MODEL_DIR, TARGET_COLUMNS
from find_best_weight import competition_proxy_loss, variable_metrics
from src.inference import predict_future as predict_future_v8
from src.v8_runtime import v8_enabled


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = MODEL_DIR / "v9_analog_candidate.json"

CONTEXTS = (48, 96, 144)
KS = (3, 5, 8)
MODES = ("target", "multi")
SELF_GUARDS = (32, 64, 96)
ALPHAS = (0.025, 0.05, 0.075, 0.10)
ROLLING_STRIDE = 8
DESC_POINTS = 24

# Conservative gate. V8 remains exact fallback for every rejected target.
TRAIN_MIN_RMSE_GAIN = 0.0025
TRAIN_MIN_PROXY_GAIN = 0.005
MAX_HOLDOUT_RMSE_RATIO = 1.01
MIN_HOLDOUT_POSITIVE = 4
FIXED_MIN_POSITIVE_PROXY = 4
FIXED_MIN_NONNEG_TREND = 3
FIXED_MAX_RMSE_RATIO = 1.01
FIXED_MAX_POOLED_RMSE_RATIO = 0.995
FIXED_MIN_POOLED_PROXY_GAIN = 0.01
FINAL_MAX_FLAT_RMSE_RATIO = 1.0
FINAL_MIN_MEAN_PROXY_GAIN = 0.01

EPS = 1e-12


@dataclass(frozen=True)
class AnalogConfig:
    context: int
    k: int
    mode: str
    self_guard: int

    def key(self) -> str:
        return (
            f"ctx={self.context},k={self.k},mode={self.mode},"
            f"guard={self.self_guard}"
        )


@dataclass(frozen=True)
class BlendParam:
    analog: AnalogConfig
    alpha: float

    def key(self) -> str:
        return f"{self.analog.key()},alpha={self.alpha:.3f}"


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return 0.0
    a = a[ok]
    b = b[ok]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < EPS or sb < EPS:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def trend_core(metrics: dict) -> float:
    corr01 = 0.5 * (float(np.clip(metrics["diff_corr"], -1.0, 1.0)) + 1.0)
    return float(
        0.35 * corr01
        + 0.25 * float(np.clip(metrics["direction_accuracy"], 0.0, 1.0))
        + 0.20 * float(np.clip(metrics["peak_f1"], 0.0, 1.0))
        + 0.20 * float(np.clip(metrics["volatility_fit"], 0.0, 1.0))
    )


def evaluate_rows(y_true: np.ndarray, y_pred: np.ndarray, anchors: np.ndarray) -> dict:
    m = dict(variable_metrics(y_true, y_pred, anchors))
    m["proxy_loss"] = float(competition_proxy_loss(m))
    m["trend_core"] = trend_core(m)
    return m


def robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    std = float(np.std(x))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr_sigma = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    level_floor = 1e-4 * max(abs(float(np.median(x))), 1.0)
    return max(std, iqr_sigma, level_floor, 1e-6)


def resample_1d(x: np.ndarray, n: int = DESC_POINTS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == n:
        return x.copy()
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, x)


def channel_descriptor(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    med = float(np.median(x))
    scale = robust_scale(x)
    z = (x - med) / scale
    shape = resample_1d(z, DESC_POINTS)

    feats = []
    for w in (12, 24, 48, 96, 144):
        if len(x) >= 2:
            ww = min(w, len(x))
            seg = x[-ww:]
            denom = max(ww - 1, 1)
            slope = float((seg[-1] - seg[0]) / denom / scale)
            feats.extend([
                slope,
                float(np.mean((seg - med) / scale)),
                float(np.std(seg) / scale),
            ])
        else:
            feats.extend([0.0, 0.0, 0.0])

    diff = np.diff(x)
    if len(diff) >= 2:
        diff_std = float(np.std(diff) / scale)
        diff_mean = float(np.mean(diff) / scale)
    elif len(diff) == 1:
        diff_std = 0.0
        diff_mean = float(diff[0] / scale)
    else:
        diff_std = 0.0
        diff_mean = 0.0

    ac1 = safe_corr(x[:-1], x[1:]) if len(x) >= 3 else 0.0
    ac4 = safe_corr(x[:-4], x[4:]) if len(x) >= 9 else 0.0
    feats.extend([diff_mean, diff_std, ac1, ac4])

    out = np.concatenate([shape, np.asarray(feats, dtype=np.float64)])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def multiscale_descriptor(window: np.ndarray, target_idx: int, mode: str) -> np.ndarray:
    window = np.asarray(window, dtype=np.float64)
    if mode == "target":
        return channel_descriptor(window[:, target_idx])
    if mode != "multi":
        raise ValueError(mode)

    parts = []
    for j in range(window.shape[1]):
        weight = 1.0 if j == target_idx else 0.35
        parts.append(weight * channel_descriptor(window[:, j]))
    return np.concatenate(parts)


def rolling_ends(length: int, context: int, stride: int) -> List[int]:
    last = length - HORIZON
    if last < context:
        return []
    values = list(range(context, last + 1, stride))
    if values and values[-1] != last:
        values.append(last)
    return values


def load_official_sequences() -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    names: List[str] = []
    histories: List[np.ndarray] = []
    futures: List[np.ndarray] = []

    for seq_dir in sorted(DATA_DIR.glob("sequence*")):
        hp = seq_dir / "history.csv"
        fp = seq_dir / "future.csv"
        if not hp.is_file() or not fp.is_file():
            continue
        hdf = pd.read_csv(hp)
        fdf = pd.read_csv(fp)
        missing_h = [c for c in TARGET_COLUMNS if c not in hdf.columns]
        missing_f = [c for c in TARGET_COLUMNS if c not in fdf.columns]
        if missing_h or missing_f:
            raise RuntimeError(
                f"{seq_dir.name}: missing history={missing_h}, future={missing_f}"
            )
        h = hdf[TARGET_COLUMNS].to_numpy(dtype=np.float64)
        f = fdf[TARGET_COLUMNS].to_numpy(dtype=np.float64)[:HORIZON]
        if len(f) != HORIZON:
            raise RuntimeError(f"{seq_dir.name}: future length={len(f)}")
        if len(h) < max(CONTEXTS) + HORIZON + max(SELF_GUARDS):
            raise RuntimeError(
                f"{seq_dir.name}: history too short for V9 diagnostic: {len(h)}"
            )
        names.append(seq_dir.name)
        histories.append(h)
        futures.append(f)

    if len(names) < 5:
        raise RuntimeError(f"expected >=5 official sequences, got {len(names)}")
    return names, histories, futures


def build_bank(
    histories: Sequence[np.ndarray],
    context: int,
    target_idx: int,
    mode: str,
):
    descs = []
    deltas = []
    src_ids = []
    future_ends = []

    for src_idx, h in enumerate(histories):
        for end in rolling_ends(len(h), context, ROLLING_STRIDE):
            ctx = h[end - context:end]
            fut = h[end:end + HORIZON, target_idx]
            if len(fut) != HORIZON:
                continue
            descs.append(multiscale_descriptor(ctx, target_idx, mode))
            anchor = float(ctx[-1, target_idx])
            scale = robust_scale(ctx[:, target_idx])
            dz = (fut - anchor) / scale
            dz = np.clip(dz, -8.0, 8.0)
            deltas.append(dz)
            src_ids.append(src_idx)
            future_ends.append(end + HORIZON)

    return (
        np.asarray(descs, dtype=np.float64),
        np.asarray(deltas, dtype=np.float64),
        np.asarray(src_ids, dtype=np.int32),
        np.asarray(future_ends, dtype=np.int32),
    )


def analog_predict_one(
    histories: Sequence[np.ndarray],
    query_idx: int,
    target_idx: int,
    cfg: AnalogConfig,
    bank,
) -> np.ndarray:
    descs, deltas, src_ids, future_ends = bank
    query_h = histories[query_idx]
    query_ctx = query_h[-cfg.context:]
    qdesc = multiscale_descriptor(query_ctx, target_idx, cfg.mode)

    legal = np.ones(len(descs), dtype=bool)
    self_mask = src_ids == query_idx
    legal[self_mask] = (
        future_ends[self_mask] <= len(query_h) - int(cfg.self_guard)
    )

    legal_idx = np.where(legal)[0]
    if len(legal_idx) < cfg.k:
        raise RuntimeError(
            f"not enough analogs for {cfg.key()} target={TARGET_COLUMNS[target_idx]}"
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

    dz = np.sum(weights[:, None] * deltas[chosen], axis=0)
    anchor = float(query_ctx[-1, target_idx])
    qscale = robust_scale(query_ctx[:, target_idx])
    pred = anchor + dz * qscale
    return np.nan_to_num(pred, nan=anchor, posinf=anchor, neginf=anchor)


def all_analog_predictions(
    histories: Sequence[np.ndarray],
) -> Dict[AnalogConfig, np.ndarray]:
    n = len(histories)
    result: Dict[AnalogConfig, np.ndarray] = {}

    banks = {}
    for context in CONTEXTS:
        for mode in MODES:
            for j in range(len(TARGET_COLUMNS)):
                banks[(context, mode, j)] = build_bank(
                    histories, context, j, mode
                )

    total = len(CONTEXTS) * len(KS) * len(MODES) * len(SELF_GUARDS)
    done = 0
    for context in CONTEXTS:
        for k in KS:
            for mode in MODES:
                for guard in SELF_GUARDS:
                    cfg = AnalogConfig(context, k, mode, guard)
                    pred = np.empty((n, HORIZON, len(TARGET_COLUMNS)), dtype=np.float64)
                    for i in range(n):
                        for j in range(len(TARGET_COLUMNS)):
                            pred[i, :, j] = analog_predict_one(
                                histories,
                                i,
                                j,
                                cfg,
                                banks[(context, mode, j)],
                            )
                    result[cfg] = pred
                    done += 1
                    print(f"analog [{done:02d}/{total:02d}] {cfg.key()}", flush=True)
    return result


def blend(v8: np.ndarray, analog: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - float(alpha)) * v8 + float(alpha) * analog


def proxy_gain(base: dict, cand: dict) -> float:
    return float(
        (base["proxy_loss"] - cand["proxy_loss"])
        / max(abs(base["proxy_loss"]), EPS)
    )


def choose_on_train(
    truth: np.ndarray,
    v8: np.ndarray,
    analog_map: Dict[AnalogConfig, np.ndarray],
    anchors: np.ndarray,
    train_idx: np.ndarray,
    target_idx: int,
) -> BlendParam | None:
    base = evaluate_rows(
        truth[train_idx, :, target_idx],
        v8[train_idx, :, target_idx],
        anchors[train_idx, target_idx],
    )
    best_param = None
    best_score = float("inf")

    for cfg, ap in analog_map.items():
        for alpha in ALPHAS:
            p = blend(v8[:, :, target_idx], ap[:, :, target_idx], alpha)
            m = evaluate_rows(
                truth[train_idx, :, target_idx],
                p[train_idx],
                anchors[train_idx, target_idx],
            )
            rmse_gain = 1.0 - m["rmse"] / max(base["rmse"], EPS)
            pg = proxy_gain(base, m)
            if rmse_gain < TRAIN_MIN_RMSE_GAIN:
                continue
            if pg < TRAIN_MIN_PROXY_GAIN:
                continue
            if m["trend_core"] + 1e-12 < base["trend_core"]:
                continue
            score = m["proxy_loss"] + 0.05 * (
                m["rmse"] / max(base["rmse"], EPS)
            )
            if score < best_score:
                best_score = score
                best_param = BlendParam(cfg, float(alpha))

    return best_param


def evaluate_fixed_param(
    truth: np.ndarray,
    v8: np.ndarray,
    analog_map: Dict[AnalogConfig, np.ndarray],
    anchors: np.ndarray,
    target_idx: int,
    param: BlendParam,
):
    cand = blend(
        v8[:, :, target_idx],
        analog_map[param.analog][:, :, target_idx],
        param.alpha,
    )
    base_pool = evaluate_rows(
        truth[:, :, target_idx], v8[:, :, target_idx], anchors[:, target_idx]
    )
    cand_pool = evaluate_rows(
        truth[:, :, target_idx], cand, anchors[:, target_idx]
    )

    seq_rows = []
    for i in range(len(truth)):
        b = evaluate_rows(
            truth[i:i + 1, :, target_idx],
            v8[i:i + 1, :, target_idx],
            anchors[i:i + 1, target_idx],
        )
        c = evaluate_rows(
            truth[i:i + 1, :, target_idx],
            cand[i:i + 1],
            anchors[i:i + 1, target_idx],
        )
        seq_rows.append({
            "proxy_gain": proxy_gain(b, c),
            "trend_gain": c["trend_core"] - b["trend_core"],
            "rmse_ratio": c["rmse"] / max(b["rmse"], EPS),
        })

    positive_proxy = int(sum(r["proxy_gain"] > 0.0 for r in seq_rows))
    nonneg_trend = int(sum(r["trend_gain"] >= -1e-12 for r in seq_rows))
    max_rmse_ratio = float(max(r["rmse_ratio"] for r in seq_rows))
    pooled_rmse_ratio = float(cand_pool["rmse"] / max(base_pool["rmse"], EPS))
    pooled_pg = proxy_gain(base_pool, cand_pool)

    eligible = bool(
        positive_proxy >= FIXED_MIN_POSITIVE_PROXY
        and nonneg_trend >= FIXED_MIN_NONNEG_TREND
        and max_rmse_ratio <= FIXED_MAX_RMSE_RATIO
        and pooled_rmse_ratio <= FIXED_MAX_POOLED_RMSE_RATIO
        and pooled_pg >= FIXED_MIN_POOLED_PROXY_GAIN
        and cand_pool["trend_core"] >= base_pool["trend_core"] - 1e-12
    )

    return {
        "eligible": eligible,
        "prediction": cand,
        "base_pool": base_pool,
        "cand_pool": cand_pool,
        "positive_proxy": positive_proxy,
        "nonneg_trend": nonneg_trend,
        "max_rmse_ratio": max_rmse_ratio,
        "pooled_rmse_ratio": pooled_rmse_ratio,
        "pooled_proxy_gain": pooled_pg,
        "seq_rows": seq_rows,
    }


def find_fixed_candidate(
    truth: np.ndarray,
    v8: np.ndarray,
    analog_map: Dict[AnalogConfig, np.ndarray],
    anchors: np.ndarray,
    target_idx: int,
):
    best = None
    best_score = None
    for cfg in analog_map:
        for alpha in ALPHAS:
            param = BlendParam(cfg, float(alpha))
            ev = evaluate_fixed_param(
                truth, v8, analog_map, anchors, target_idx, param
            )
            if not ev["eligible"]:
                continue
            worst_pg = min(r["proxy_gain"] for r in ev["seq_rows"])
            score = (
                worst_pg,
                ev["pooled_proxy_gain"],
                1.0 - ev["pooled_rmse_ratio"],
            )
            if best is None or score > best_score:
                best = (param, ev)
                best_score = score
    return best


def main() -> int:
    required_models = [
        MODEL_DIR / "model_lgb.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "model_xgb.pkl",
        MODEL_DIR / "scaler_xgb.pkl",
        MODEL_DIR / "ensemble_config.pkl",
        MODEL_DIR / "model_pca_xgb.pkl",
        MODEL_DIR / "preprocess_pca_xgb.pkl",
    ]
    missing = [str(p) for p in required_models if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing V8 models:\n" + "\n".join(missing))

    with open(MODEL_DIR / "ensemble_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    if not v8_enabled(cfg):
        raise RuntimeError(
            f"ensemble_config is not V8: version={cfg.get('version')} "
            f"trajectory_model={cfg.get('trajectory_model')}"
        )

    names, histories, futures = load_official_sequences()
    truth = np.stack(futures, axis=0)
    anchors = np.stack([h[-1] for h in histories], axis=0)

    print("=" * 118)
    print("V9 FINAL OFFLINE DIAGNOSTIC: V8 + OFFICIAL-ONLY MULTISCALE ANALOG RETRIEVAL")
    print("=" * 118)
    print(f"sequences           : {names}")
    print(f"history lengths     : {[len(x) for x in histories]}")
    print(f"V8 version          : {cfg.get('version')}")
    print(f"trajectory model    : {cfg.get('trajectory_model')}")
    print(f"contexts            : {CONTEXTS}")
    print(f"k values            : {KS}")
    print(f"modes               : {MODES}")
    print(f"self guards         : {SELF_GUARDS}")
    print(f"blend alphas        : {ALPHAS}")
    print("data rule           : analog bank uses history.csv only; future.csv is evaluation only")
    print("safety              : rejected targets stay EXACT V8; production API untouched")

    v8 = np.empty_like(truth)
    for i, name in enumerate(names):
        print(f"V8 predict [{i+1}/{len(names)}] {name}", flush=True)
        hdf = pd.DataFrame(histories[i], columns=TARGET_COLUMNS)
        v8[i] = np.asarray(predict_future_v8(hdf), dtype=np.float64)

    print("\nBuilding rolling analog banks and predictions ...")
    analog_map = all_analog_predictions(histories)

    final = v8.copy()
    target_results = {}
    enabled = []

    print("\n" + "=" * 118)
    print("LOSO + FIXED WORST-CASE SAFETY GATE")
    print("=" * 118)

    for j, name in enumerate(TARGET_COLUMNS):
        print(f"\n{name}")
        holdout_rows = []
        selected_count = 0
        positive_count = 0

        for holdout in range(len(names)):
            train_idx = np.asarray(
                [i for i in range(len(names)) if i != holdout], dtype=np.int64
            )
            param = choose_on_train(
                truth, v8, analog_map, anchors, train_idx, j
            )
            if param is None:
                holdout_rows.append({
                    "sequence": names[holdout],
                    "selected": None,
                    "proxy_gain": 0.0,
                    "trend_gain": 0.0,
                    "rmse_ratio": 1.0,
                })
                print(f"  {names[holdout]:14s}: V8 (no safe train candidate)")
                continue

            selected_count += 1
            pred_h = blend(
                v8[holdout:holdout + 1, :, j],
                analog_map[param.analog][holdout:holdout + 1, :, j],
                param.alpha,
            )
            b = evaluate_rows(
                truth[holdout:holdout + 1, :, j],
                v8[holdout:holdout + 1, :, j],
                anchors[holdout:holdout + 1, j],
            )
            c = evaluate_rows(
                truth[holdout:holdout + 1, :, j],
                pred_h,
                anchors[holdout:holdout + 1, j],
            )
            pg = proxy_gain(b, c)
            tg = c["trend_core"] - b["trend_core"]
            rr = c["rmse"] / max(b["rmse"], EPS)
            good = bool(pg > 0.0 and rr <= MAX_HOLDOUT_RMSE_RATIO)
            positive_count += int(good)
            holdout_rows.append({
                "sequence": names[holdout],
                "selected": param.key(),
                "proxy_gain": pg,
                "trend_gain": tg,
                "rmse_ratio": rr,
            })
            print(
                f"  {names[holdout]:14s}: {param.key():55s} "
                f"proxy={100*pg:+6.2f}% trend={tg:+.4f} rmse_ratio={rr:.4f}"
            )

        fixed = find_fixed_candidate(truth, v8, analog_map, anchors, j)
        loso_ok = bool(
            selected_count >= MIN_HOLDOUT_POSITIVE
            and positive_count >= MIN_HOLDOUT_POSITIVE
        )

        if fixed is None:
            print(
                f"  FIXED GATE: REJECT | loso selected={selected_count}/5 "
                f"positive={positive_count}/5 | no 4/5 worst-case candidate"
            )
            target_results[name] = {
                "enabled": False,
                "loso_selected": selected_count,
                "loso_positive": positive_count,
                "holdouts": holdout_rows,
                "reason": "no fixed safe candidate",
            }
            continue

        param, ev = fixed
        fixed_ok = bool(ev["eligible"] and loso_ok)
        print(
            f"  FIXED: {param.key()} | positive_proxy={ev['positive_proxy']}/5 "
            f"nonneg_trend={ev['nonneg_trend']}/5 max_rmse_ratio={ev['max_rmse_ratio']:.4f} "
            f"pooled_rmse_ratio={ev['pooled_rmse_ratio']:.4f} "
            f"pooled_proxy={100*ev['pooled_proxy_gain']:+.2f}%"
        )
        print(f"  TARGET GATE: {'PASS' if fixed_ok else 'REJECT'}")

        target_results[name] = {
            "enabled": fixed_ok,
            "loso_selected": selected_count,
            "loso_positive": positive_count,
            "holdouts": holdout_rows,
            "fixed_param": param.key(),
            "fixed_positive_proxy": ev["positive_proxy"],
            "fixed_nonneg_trend": ev["nonneg_trend"],
            "fixed_max_rmse_ratio": ev["max_rmse_ratio"],
            "fixed_pooled_rmse_ratio": ev["pooled_rmse_ratio"],
            "fixed_pooled_proxy_gain_pct": 100.0 * ev["pooled_proxy_gain"],
        }

        if fixed_ok:
            final[:, :, j] = ev["prediction"]
            enabled.append(name)

    base_flat = float(np.sqrt(np.mean((truth - v8) ** 2)))
    final_flat = float(np.sqrt(np.mean((truth - final) ** 2)))
    flat_ratio = final_flat / max(base_flat, EPS)

    base_proxy = []
    final_proxy = []
    for j in range(len(TARGET_COLUMNS)):
        b = evaluate_rows(truth[:, :, j], v8[:, :, j], anchors[:, j])
        c = evaluate_rows(truth[:, :, j], final[:, :, j], anchors[:, j])
        base_proxy.append(b["proxy_loss"])
        final_proxy.append(c["proxy_loss"])
    mean_base_proxy = float(np.mean(base_proxy))
    mean_final_proxy = float(np.mean(final_proxy))
    mean_proxy_gain = float(
        (mean_base_proxy - mean_final_proxy) / max(abs(mean_base_proxy), EPS)
    )

    final_gate = bool(
        len(enabled) > 0
        and flat_ratio <= FINAL_MAX_FLAT_RMSE_RATIO
        and mean_proxy_gain >= FINAL_MIN_MEAN_PROXY_GAIN
    )

    print("\n" + "=" * 118)
    print("V9 FINAL SUMMARY")
    print("=" * 118)
    print(f"enabled targets     : {enabled}")
    print(f"flat RMSE V8        : {base_flat:.6f}")
    print(f"flat RMSE V9        : {final_flat:.6f}")
    print(f"flat RMSE ratio     : {flat_ratio:.6f}")
    print(f"mean proxy V8       : {mean_base_proxy:.6f}")
    print(f"mean proxy V9       : {mean_final_proxy:.6f}")
    print(f"mean proxy gain     : {100*mean_proxy_gain:+.2f}%")
    print(f"V9 OFFLINE GATE     : {'PASS' if final_gate else 'REJECT'}")

    output = {
        "candidate": "v9_official_only_multiscale_analog",
        "base_version": int(cfg.get("version", -1)),
        "base_trajectory_model": cfg.get("trajectory_model"),
        "data_policy": "official history.csv only for analog bank; future.csv evaluation only",
        "contexts": list(CONTEXTS),
        "k_values": list(KS),
        "modes": list(MODES),
        "self_guards": list(SELF_GUARDS),
        "alphas": list(ALPHAS),
        "rolling_stride": ROLLING_STRIDE,
        "enabled_targets": enabled,
        "targets": target_results,
        "flat_rmse_v8": base_flat,
        "flat_rmse_v9": final_flat,
        "flat_rmse_ratio": flat_ratio,
        "mean_proxy_v8": mean_base_proxy,
        "mean_proxy_v9": mean_final_proxy,
        "mean_proxy_gain_pct": 100.0 * mean_proxy_gain,
        "offline_gate_pass": final_gate,
        "important_note": (
            "This is an offline diagnostic only. It does not modify app.py, "
            "ensemble_config.pkl, or production V8. Local proxy is not the official score."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"metrics             : {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
