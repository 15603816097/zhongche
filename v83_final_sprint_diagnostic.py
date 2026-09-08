from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from find_best_weight import competition_proxy_loss, variable_metrics
from src.feature_engineer import robust_trend_forecast
from src.inference import predict_future as predict_future_v8
from src.trajectory_fusion import moving_average
from src.trend_pattern import adaptive_pattern_forecast
from src.v8_runtime import v8_enabled


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
OUTPUT_PATH = ROOT / "models" / "v83_final_candidate.json"
SEGMENTS = ((0, 32), (32, 64), (64, 96))

PATTERN_WEIGHTS = (0.0, 0.05, 0.10, 0.15)
TREND_WEIGHTS = (0.0, 0.05, 0.10)
HP_GAINS = (0.80, 1.00, 1.20, 1.40)
AMP_GAINS = (0.95, 1.00, 1.05)
HP_WINDOWS = (7, 11, 15)

# Last-submission safety rules.  We only permit tiny accuracy movement in exchange
# for a repeatable trend improvement.  Failed targets stay EXACT V8.
MAX_RMSE_RATIO_TRAIN = 1.010
MAX_MAE_RATIO_TRAIN = 1.015
MIN_TREND_GAIN_TRAIN = 0.005
MIN_PROXY_GAIN_TRAIN = 0.003
MIN_HOLDOUT_POSITIVE = 4
MIN_FIXED_PROXY_GAIN_PCT = 0.50
MIN_FIXED_TREND_GAIN = 0.004
MAX_FIXED_RMSE_RATIO = 1.010


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return 0.0
    a = a[ok]
    b = b[ok]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
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


def evaluate_rows(true_rows: np.ndarray, pred_rows: np.ndarray, anchors: np.ndarray) -> dict:
    m = variable_metrics(true_rows, pred_rows, anchors)
    m = dict(m)
    m["proxy_loss"] = float(competition_proxy_loss(m))
    m["trend_core"] = trend_core(m)
    return m


def apply_params(
    v8: np.ndarray,
    pattern: np.ndarray,
    trend: np.ndarray,
    anchor: float,
    params: tuple[float, float, float, float, int],
) -> np.ndarray:
    pw, tw, hp_gain, amp_gain, hp_window = params
    if pw + tw > 0.30 + 1e-12:
        raise ValueError("unsafe blend sum")
    mixed = (1.0 - pw - tw) * v8 + pw * pattern + tw * trend
    smooth = moving_average(mixed.reshape(1, -1), int(hp_window))[0]
    shaped = smooth + float(hp_gain) * (mixed - smooth)
    out = float(anchor) + float(amp_gain) * (shaped - float(anchor))
    return np.nan_to_num(out, nan=float(anchor), posinf=float(anchor), neginf=float(anchor))


def candidate_grid():
    for pw in PATTERN_WEIGHTS:
        for tw in TREND_WEIGHTS:
            if pw + tw > 0.20 + 1e-12:
                continue
            for hg in HP_GAINS:
                for ag in AMP_GAINS:
                    for win in HP_WINDOWS:
                        yield (float(pw), float(tw), float(hg), float(ag), int(win))


def params_key(p: tuple[float, float, float, float, int]) -> str:
    return f"pw={p[0]:.2f},tw={p[1]:.2f},hp={p[2]:.2f},amp={p[3]:.2f},win={p[4]}"


def choose_on_train(
    true_rows: np.ndarray,
    v8_rows: np.ndarray,
    pattern_rows: np.ndarray,
    trend_rows: np.ndarray,
    anchors: np.ndarray,
) -> tuple[tuple[float, float, float, float, int] | None, dict, dict]:
    base = evaluate_rows(true_rows, v8_rows, anchors)
    best_params = None
    best = base

    for params in candidate_grid():
        cand = np.vstack([
            apply_params(v8_rows[i], pattern_rows[i], trend_rows[i], anchors[i], params)
            for i in range(len(true_rows))
        ])
        m = evaluate_rows(true_rows, cand, anchors)
        if m["rmse"] > base["rmse"] * MAX_RMSE_RATIO_TRAIN:
            continue
        if m["mae"] > base["mae"] * MAX_MAE_RATIO_TRAIN:
            continue
        if m["trend_core"] < base["trend_core"] + MIN_TREND_GAIN_TRAIN:
            continue
        if m["proxy_loss"] > base["proxy_loss"] - MIN_PROXY_GAIN_TRAIN:
            continue
        if best_params is None or m["proxy_loss"] < best["proxy_loss"] - 1e-12:
            best_params = params
            best = m

    return best_params, base, best


def segment_diagnostic(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for start, end in SEGMENTS:
        yt = y[:, start:end]
        pt = p[:, start:end]
        rmse = float(np.sqrt(np.mean((yt - pt) ** 2)))
        mae = float(np.mean(np.abs(yt - pt)))
        if end - start >= 3:
            corr = safe_corr(np.diff(pt, axis=1), np.diff(yt, axis=1))
            dp = np.diff(pt, axis=1).reshape(-1)
            dy = np.diff(yt, axis=1).reshape(-1)
            scale = float(np.std(dy))
            dead = max(1e-12, 1e-4 * scale)
            sp = np.where(np.abs(dp) <= dead, 0.0, np.sign(dp))
            sy = np.where(np.abs(dy) <= dead, 0.0, np.sign(dy))
            direction = float(np.mean(sp == sy))
        else:
            corr = 0.0
            direction = 0.0
        rows.append({
            "start": start,
            "end": end,
            "rmse": rmse,
            "mae": mae,
            "diff_corr": corr,
            "direction_accuracy": direction,
        })
    return rows


def robust_consensus(selected: list[tuple[float, float, float, float, int] | None]):
    nonzero = [p for p in selected if p is not None]
    if len(nonzero) < 3:
        return None
    counts = Counter(nonzero)
    mode, count = counts.most_common(1)[0]
    if count >= 3:
        return mode

    arr = np.asarray([[p[0], p[1], p[2], p[3], p[4]] for p in nonzero], dtype=np.float64)
    med = np.median(arr, axis=0)
    # Snap robust medians back to the tested grid so production behavior is deterministic.
    def nearest(value, grid):
        return min(grid, key=lambda x: abs(float(x) - float(value)))
    return (
        float(nearest(med[0], PATTERN_WEIGHTS)),
        float(nearest(med[1], TREND_WEIGHTS)),
        float(nearest(med[2], HP_GAINS)),
        float(nearest(med[3], AMP_GAINS)),
        int(nearest(med[4], HP_WINDOWS)),
    )


def main() -> int:
    required = [
        CORPUS_PATH,
        MODEL_DIR / "model_lgb.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "model_xgb.pkl",
        MODEL_DIR / "scaler_xgb.pkl",
        MODEL_DIR / "ensemble_config.pkl",
        MODEL_DIR / "model_pca_xgb.pkl",
        MODEL_DIR / "preprocess_pca_xgb.pkl",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))

    with open(MODEL_DIR / "ensemble_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    if not v8_enabled(cfg):
        raise RuntimeError(f"current ensemble is not V8: version={cfg.get('version')}")

    data = np.load(CORPUS_PATH, allow_pickle=False)
    X = data["X"].astype(np.float64, copy=False)
    Yz = data["Y"].astype(np.float64, copy=False)
    center = data["center"].astype(np.float64, copy=False)
    scale = data["scale"].astype(np.float64, copy=False)
    group_id = data["group_id"].astype(str)
    targets = data["targets"].astype(str).tolist()
    if targets != list(TARGET_COLUMNS):
        raise RuntimeError(f"target mismatch: {targets}")

    center3 = center[:, None, :]
    scale3 = scale[:, None, :]
    history = X * scale3 + center3
    truth = Yz * scale3 + center3
    n = len(group_id)
    anchors = history[:, -1, :]

    v8 = np.empty_like(truth)
    pattern = np.empty_like(truth)
    trend = np.empty_like(truth)

    print("=" * 118)
    print("V8.3 FINAL SPRINT - ALL-TARGET DIAGNOSTIC + CONSERVATIVE LOSO POST-PROCESS SEARCH")
    print("=" * 118)
    print(f"V8 version          : {cfg.get('version')}")
    print(f"trajectory model    : {cfg.get('trajectory_model')}")
    print(f"sequences           : {list(group_id)}")
    print("search              : exact V8 + small pattern/trend/high-pass/amplitude corrections")
    print("safety              : failed target stays exact V8; app.py/callback/config untouched")

    for i, gid in enumerate(group_id):
        hdf = pd.DataFrame(history[i], columns=TARGET_COLUMNS)
        print(f"build [{i+1}/{n}] {gid}: V8 + pattern + robust trend ...", flush=True)
        v8[i] = np.asarray(predict_future_v8(hdf), dtype=np.float64)
        pattern[i] = np.asarray(adaptive_pattern_forecast(hdf, HORIZON), dtype=np.float64)
        trend[i] = np.asarray(robust_trend_forecast(hdf, HORIZON), dtype=np.float64)

    print("\n" + "=" * 118)
    print("BASE V8 DIAGNOSTIC BY TARGET")
    print("=" * 118)
    base_rows = {}
    for j, name in enumerate(TARGET_COLUMNS):
        m = evaluate_rows(truth[:, :, j], v8[:, :, j], anchors[:, j])
        base_rows[name] = m
        print(
            f"{name:16s} rmse={m['rmse']:.6f} mae={m['mae']:.6f} "
            f"corr={m['diff_corr']:+.4f} dir={m['direction_accuracy']:.4f} "
            f"peak={m['peak_f1']:.4f} vol={m['volatility_fit']:.4f} "
            f"trend={m['trend_core']:.4f} proxy={m['proxy_loss']:.5f}"
        )
        for seg in segment_diagnostic(truth[:, :, j], v8[:, :, j]):
            print(
                f"  [{seg['start']:02d}:{seg['end']:02d}] rmse={seg['rmse']:.6f} "
                f"corr={seg['diff_corr']:+.4f} dir={seg['direction_accuracy']:.4f}"
            )

    final_pred = v8.copy()
    target_results = {}
    enabled_targets = []

    print("\n" + "=" * 118)
    print("TRUE LOSO SEARCH")
    print("=" * 118)
    for j, name in enumerate(TARGET_COLUMNS):
        selected = []
        holdout_proxy_gain = []
        holdout_trend_gain = []
        holdout_rmse_ratio = []
        fold_rows = {}
        print(f"\n{name}")
        for holdout in range(n):
            train_idx = np.asarray([k for k in range(n) if k != holdout], dtype=np.int64)
            p, train_base, train_best = choose_on_train(
                truth[train_idx, :, j],
                v8[train_idx, :, j],
                pattern[train_idx, :, j],
                trend[train_idx, :, j],
                anchors[train_idx, j],
            )
            selected.append(p)
            base_h = evaluate_rows(
                truth[holdout:holdout+1, :, j],
                v8[holdout:holdout+1, :, j],
                anchors[holdout:holdout+1, j],
            )
            if p is None:
                cand_h = base_h
            else:
                pred_h = apply_params(
                    v8[holdout, :, j], pattern[holdout, :, j], trend[holdout, :, j],
                    anchors[holdout, j], p,
                )[None, :]
                cand_h = evaluate_rows(
                    truth[holdout:holdout+1, :, j], pred_h, anchors[holdout:holdout+1, j]
                )
            pg = 100.0 * (base_h["proxy_loss"] - cand_h["proxy_loss"]) / max(abs(base_h["proxy_loss"]), 1e-12)
            tg = cand_h["trend_core"] - base_h["trend_core"]
            rr = cand_h["rmse"] / max(base_h["rmse"], 1e-12)
            holdout_proxy_gain.append(pg)
            holdout_trend_gain.append(tg)
            holdout_rmse_ratio.append(rr)
            fold_rows[str(group_id[holdout])] = {
                "selected": None if p is None else params_key(p),
                "proxy_gain_pct": pg,
                "trend_gain": tg,
                "rmse_ratio": rr,
            }
            print(
                f"  {group_id[holdout]:14s}: {('V8' if p is None else params_key(p)):42s} "
                f"proxy={pg:+6.2f}% trend={tg:+.4f} rmse_ratio={rr:.4f}"
            )

        consensus = robust_consensus(selected)
        positive = int(np.sum(np.asarray(holdout_proxy_gain) > 0.0))
        fixed = base_rows[name]
        fixed_proxy_gain_pct = 0.0
        fixed_trend_gain = 0.0
        fixed_rmse_ratio = 1.0
        enabled = False

        if consensus is not None:
            fixed_rows = np.vstack([
                apply_params(v8[i, :, j], pattern[i, :, j], trend[i, :, j], anchors[i, j], consensus)
                for i in range(n)
            ])
            fixed = evaluate_rows(truth[:, :, j], fixed_rows, anchors[:, j])
            fixed_proxy_gain_pct = 100.0 * (
                base_rows[name]["proxy_loss"] - fixed["proxy_loss"]
            ) / max(abs(base_rows[name]["proxy_loss"]), 1e-12)
            fixed_trend_gain = fixed["trend_core"] - base_rows[name]["trend_core"]
            fixed_rmse_ratio = fixed["rmse"] / max(base_rows[name]["rmse"], 1e-12)
            enabled = bool(
                positive >= MIN_HOLDOUT_POSITIVE
                and fixed_proxy_gain_pct >= MIN_FIXED_PROXY_GAIN_PCT
                and fixed_trend_gain >= MIN_FIXED_TREND_GAIN
                and fixed_rmse_ratio <= MAX_FIXED_RMSE_RATIO
            )
            if enabled:
                final_pred[:, :, j] = fixed_rows
                enabled_targets.append(name)

        print(
            f"  CONSENSUS: {('none' if consensus is None else params_key(consensus))} | "
            f"holdout_positive={positive}/{n} fixed_proxy={fixed_proxy_gain_pct:+.2f}% "
            f"fixed_trend={fixed_trend_gain:+.4f} fixed_rmse_ratio={fixed_rmse_ratio:.4f} "
            f"=> {'ENABLE' if enabled else 'KEEP V8'}"
        )
        target_results[name] = {
            "base": base_rows[name],
            "folds": fold_rows,
            "selected_fold_params": [None if p is None else params_key(p) for p in selected],
            "holdout_positive_proxy": positive,
            "consensus": None if consensus is None else {
                "pattern_weight": consensus[0],
                "trend_weight": consensus[1],
                "highpass_gain": consensus[2],
                "amplitude_gain": consensus[3],
                "highpass_window": consensus[4],
            },
            "fixed_proxy_gain_pct": fixed_proxy_gain_pct,
            "fixed_trend_gain": fixed_trend_gain,
            "fixed_rmse_ratio": fixed_rmse_ratio,
            "enabled": enabled,
        }

    # Global safety check: average per-target metrics, plus true flat RMSE across all physical values.
    base_flat_rmse = float(np.sqrt(np.mean((v8 - truth) ** 2)))
    final_flat_rmse = float(np.sqrt(np.mean((final_pred - truth) ** 2)))
    global_rmse_ratio = final_flat_rmse / max(base_flat_rmse, 1e-12)
    base_proxy_mean = float(np.mean([base_rows[n]["proxy_loss"] for n in TARGET_COLUMNS]))
    final_metrics = {
        name: evaluate_rows(truth[:, :, j], final_pred[:, :, j], anchors[:, j])
        for j, name in enumerate(TARGET_COLUMNS)
    }
    final_proxy_mean = float(np.mean([final_metrics[n]["proxy_loss"] for n in TARGET_COLUMNS]))
    global_proxy_gain_pct = 100.0 * (base_proxy_mean - final_proxy_mean) / max(abs(base_proxy_mean), 1e-12)

    gate = bool(
        len(enabled_targets) >= 1
        and global_rmse_ratio <= 1.005
        and global_proxy_gain_pct >= 0.30
    )

    print("\n" + "=" * 118)
    print("V8.3 FINAL DECISION")
    print("=" * 118)
    print(f"enabled targets      : {enabled_targets}")
    print(f"flat RMSE V8->V8.3   : {base_flat_rmse:.6f} -> {final_flat_rmse:.6f} ratio={global_rmse_ratio:.5f}")
    print(f"mean proxy gain      : {global_proxy_gain_pct:+.2f}%")
    print(f"V8.3 FINAL GATE      : {'PASS' if gate else 'REJECT'}")
    print("IMPORTANT: this script did not modify app.py, callback schema, ensemble_config.pkl, or the running service.")

    result = {
        "candidate": "V8.3 final-sprint conservative trend postprocessor",
        "base_version": int(cfg.get("version", 0)),
        "enabled_targets": enabled_targets,
        "targets": target_results,
        "base_flat_rmse": base_flat_rmse,
        "final_flat_rmse": final_flat_rmse,
        "global_rmse_ratio": global_rmse_ratio,
        "base_proxy_mean": base_proxy_mean,
        "final_proxy_mean": final_proxy_mean,
        "global_proxy_gain_pct": global_proxy_gain_pct,
        "offline_gate_pass": gate,
        "note": "Only targets passing conservative true-LOSO gates are changed; all others remain exact V8.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics              : {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
