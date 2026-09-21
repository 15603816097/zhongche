from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.full_arch_runtime import (
    preload_full_arch_runtime,
    predict_future as predict_future_full_arch,
)
from src.trajectory_fusion import endpoint_zero_highpass
from v9_analog_multiscale_diagnostic import evaluate_rows, proxy_gain


ROOT = Path(__file__).resolve().parent
OUT_JSON = MODEL_DIR / "fa52_temp_msf_lite_candidate.json"
OUT_NPZ = MODEL_DIR / "fa52_temp_msf_lite_candidate.npz"

TEMP_NAME = "temperature_c"
TEMP_IDX = TARGET_COLUMNS.index(TEMP_NAME)

# Full Architecture V1 frozen offline reference. This is used only as an
# integrity check that the baseline path is the intended 52.895 architecture.
EXPECTED_FA_FLAT_RMSE = 7.801844
BASELINE_RMSE_TOL = 5e-3

# MSF-Lite deliberately DOES NOT touch the short scale.
# It only makes small medium/long corrections in the middle/far horizon.
WINDOWS = (5, 13, 33)
SEGMENTS = ((32, 64), (64, 96))
GAIN_GRID = (-0.05, -0.025, 0.0, 0.025, 0.05)

# Hard safety gates. Trend is protected more strictly than before because
# the previous hidden evaluation showed Accuracy can improve while Trend falls.
TRAIN_MAX_RMSE_RATIO = 0.999
TRAIN_MIN_PROXY_GAIN = 0.002
TRAIN_MIN_TREND_GAIN = 0.0

FIXED_MIN_POSITIVE_PROXY = 4
FIXED_MIN_NONNEG_TREND = 5
FIXED_MAX_SEQ_RMSE_RATIO = 1.005
FIXED_MAX_SEQ_TREND_DROP = 1e-12
FIXED_MAX_POOLED_RMSE_RATIO = 0.997
FIXED_MIN_POOLED_PROXY_GAIN = 0.005
FIXED_MIN_POOLED_TREND_GAIN = 0.0

LOSO_MIN_POSITIVE_PROXY = 4
LOSO_MIN_NONNEG_TREND = 5
LOSO_MAX_RMSE_RATIO = 1.01
LOSO_MAX_TREND_DROP = 1e-12

GLOBAL_MAX_FLAT_RMSE_RATIO = 1.0
GLOBAL_MIN_PROXY_GAIN = 0.0
GLOBAL_MIN_TREND_GAIN = 0.0

EPS = 1e-12


def load_sequences():
    names = []
    histories_df = []
    truth = []
    anchors = []

    for seq_dir in sorted(Path(DATA_DIR).glob("sequence*")):
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
                f"{seq_dir.name}: missing history={missing_h} future={missing_f}"
            )

        clean = clean_sequence(hdf[TARGET_COLUMNS])
        future = fdf[TARGET_COLUMNS].iloc[:HORIZON].to_numpy(dtype=np.float64)
        if future.shape != (HORIZON, len(TARGET_COLUMNS)):
            raise RuntimeError(f"{seq_dir.name}: bad future shape={future.shape}")
        if not np.isfinite(future).all():
            raise RuntimeError(f"{seq_dir.name}: future contains NaN/Inf")

        names.append(seq_dir.name)
        histories_df.append(hdf[TARGET_COLUMNS].copy())
        truth.append(future)
        anchors.append(
            clean[TARGET_COLUMNS].iloc[-1].to_numpy(dtype=np.float64)
        )

    if len(names) < 5:
        raise RuntimeError(f"expected >=5 official sequences, got {len(names)}")

    return (
        names,
        histories_df,
        np.stack(truth),
        np.stack(anchors),
    )


def predict_full_arch(histories_df):
    preload_full_arch_runtime()
    out = np.empty(
        (len(histories_df), HORIZON, len(TARGET_COLUMNS)),
        dtype=np.float64,
    )
    timings = []
    for i, hdf in enumerate(histories_df):
        print(f"Full-Arch predict [{i+1}/{len(histories_df)}]", flush=True)
        pred, t = predict_future_full_arch(hdf, return_timings=True)
        out[i] = np.asarray(pred, dtype=np.float64)
        timings.append(float(t.get("total", np.nan)))
    return out, timings


def make_bands(x):
    x = np.asarray(x, dtype=np.float64)
    hp5 = endpoint_zero_highpass(x, WINDOWS[0])
    hp13 = endpoint_zero_highpass(x, WINDOWS[1])
    hp33 = endpoint_zero_highpass(x, WINDOWS[2])
    medium = hp13 - hp5
    long_local = hp33 - hp13
    return medium, long_local


def apply_candidate_one(base_1d, params):
    """
    params = (mid_32_64, long_32_64, mid_64_96, long_64_96)

    First 32 steps remain EXACT Full Architecture.
    Short-scale band is always zero-weight.
    """
    base_1d = np.asarray(base_1d, dtype=np.float64)
    medium, long_local = make_bands(base_1d)
    out = base_1d.copy()

    for seg_idx, (start, end) in enumerate(SEGMENTS):
        gm = float(params[2 * seg_idx])
        gl = float(params[2 * seg_idx + 1])
        out[start:end] = (
            base_1d[start:end]
            + gm * medium[start:end]
            + gl * long_local[start:end]
        )
    return out


def candidate_cube(base, params):
    out = np.asarray(base, dtype=np.float64).copy()
    for i in range(len(out)):
        out[i, :, TEMP_IDX] = apply_candidate_one(
            base[i, :, TEMP_IDX],
            params,
        )
    return out


def temp_metrics(truth, pred, anchors, idx):
    return evaluate_rows(
        truth[idx, :, TEMP_IDX],
        pred[idx, :, TEMP_IDX],
        anchors[idx, TEMP_IDX],
    )


def per_sequence_rows(truth, base, cand, anchors):
    rows = []
    for i in range(len(truth)):
        b = evaluate_rows(
            truth[i:i+1, :, TEMP_IDX],
            base[i:i+1, :, TEMP_IDX],
            anchors[i:i+1, TEMP_IDX],
        )
        c = evaluate_rows(
            truth[i:i+1, :, TEMP_IDX],
            cand[i:i+1, :, TEMP_IDX],
            anchors[i:i+1, TEMP_IDX],
        )
        rows.append(
            {
                "proxy_gain": float(proxy_gain(b, c)),
                "trend_gain": float(c["trend_core"] - b["trend_core"]),
                "rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
            }
        )
    return rows


def all_params():
    return itertools.product(GAIN_GRID, repeat=4)


def choose_on_train(truth, base, anchors, train_idx):
    b = temp_metrics(truth, base, anchors, train_idx)
    best = None

    for raw in all_params():
        params = tuple(float(x) for x in raw)
        if not any(abs(x) > 1e-12 for x in params):
            continue

        cand = candidate_cube(base, params)
        c = temp_metrics(truth, cand, anchors, train_idx)

        rmse_ratio = float(c["rmse"] / max(b["rmse"], EPS))
        pg = float(proxy_gain(b, c))
        tg = float(c["trend_core"] - b["trend_core"])

        if rmse_ratio > TRAIN_MAX_RMSE_RATIO:
            continue
        if pg < TRAIN_MIN_PROXY_GAIN:
            continue
        if tg < TRAIN_MIN_TREND_GAIN - 1e-12:
            continue

        # Score-aware but safety-first: favor worst-case style conservative
        # parameters by preferring smaller total correction after proxy/RMSE.
        complexity = float(sum(abs(x) for x in params))
        score = (pg, 1.0 - rmse_ratio, tg, -complexity)

        if best is None or score > best[0]:
            best = (score, params)

    return None if best is None else best[1]


def evaluate_loso(names, truth, base, anchors):
    rows = []

    for holdout in range(len(truth)):
        train_idx = np.asarray(
            [i for i in range(len(truth)) if i != holdout],
            dtype=np.int64,
        )
        params = choose_on_train(truth, base, anchors, train_idx)

        if params is None:
            rows.append(
                {
                    "holdout": holdout,
                    "params": None,
                    "proxy_gain": 0.0,
                    "trend_gain": 0.0,
                    "rmse_ratio": 1.0,
                }
            )
            continue

        cand = candidate_cube(base[holdout:holdout+1], params)
        b = evaluate_rows(
            truth[holdout:holdout+1, :, TEMP_IDX],
            base[holdout:holdout+1, :, TEMP_IDX],
            anchors[holdout:holdout+1, TEMP_IDX],
        )
        c = evaluate_rows(
            truth[holdout:holdout+1, :, TEMP_IDX],
            cand[:, :, TEMP_IDX],
            anchors[holdout:holdout+1, TEMP_IDX],
        )
        rows.append(
            {
                "holdout": holdout,
                "params": params,
                "proxy_gain": float(proxy_gain(b, c)),
                "trend_gain": float(c["trend_core"] - b["trend_core"]),
                "rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
            }
        )

    positive_proxy = sum(r["proxy_gain"] > 0.0 for r in rows)
    nonneg_trend = sum(r["trend_gain"] >= -LOSO_MAX_TREND_DROP for r in rows)
    max_rmse_ratio = max(r["rmse_ratio"] for r in rows)
    worst_trend_gain = min(r["trend_gain"] for r in rows)

    passed = bool(
        positive_proxy >= LOSO_MIN_POSITIVE_PROXY
        and nonneg_trend >= LOSO_MIN_NONNEG_TREND
        and max_rmse_ratio <= LOSO_MAX_RMSE_RATIO
        and worst_trend_gain >= -LOSO_MAX_TREND_DROP
    )

    print("\nLOSO TEMPERATURE SAFETY")
    for r in rows:
        print(
            f"{names[r['holdout']]} params={r['params']} "
            f"proxy_gain={100*r['proxy_gain']:+.2f}% "
            f"trend_gain={r['trend_gain']:+.6f} "
            f"rmse_ratio={r['rmse_ratio']:.6f}"
        )
    print(
        f"LOSO summary: positive_proxy={positive_proxy}/{len(rows)} "
        f"nonneg_trend={nonneg_trend}/{len(rows)} "
        f"max_rmse_ratio={max_rmse_ratio:.6f} "
        f"worst_trend_gain={worst_trend_gain:+.6f} "
        f"PASS={passed}"
    )
    return rows, passed


def choose_fixed(truth, base, anchors):
    idx = np.arange(len(truth), dtype=np.int64)
    b_pool = temp_metrics(truth, base, anchors, idx)
    best = None

    for raw in all_params():
        params = tuple(float(x) for x in raw)
        if not any(abs(x) > 1e-12 for x in params):
            continue

        cand = candidate_cube(base, params)
        c_pool = temp_metrics(truth, cand, anchors, idx)
        seq = per_sequence_rows(truth, base, cand, anchors)

        positive = sum(r["proxy_gain"] > 0.0 for r in seq)
        nonneg_trend = sum(
            r["trend_gain"] >= -FIXED_MAX_SEQ_TREND_DROP for r in seq
        )
        max_seq_rmse = max(r["rmse_ratio"] for r in seq)
        worst_trend = min(r["trend_gain"] for r in seq)
        rmse_ratio = float(c_pool["rmse"] / max(b_pool["rmse"], EPS))
        pg = float(proxy_gain(b_pool, c_pool))
        tg = float(c_pool["trend_core"] - b_pool["trend_core"])

        eligible = bool(
            positive >= FIXED_MIN_POSITIVE_PROXY
            and nonneg_trend >= FIXED_MIN_NONNEG_TREND
            and max_seq_rmse <= FIXED_MAX_SEQ_RMSE_RATIO
            and worst_trend >= -FIXED_MAX_SEQ_TREND_DROP
            and rmse_ratio <= FIXED_MAX_POOLED_RMSE_RATIO
            and pg >= FIXED_MIN_POOLED_PROXY_GAIN
            and tg >= FIXED_MIN_POOLED_TREND_GAIN - 1e-12
        )
        if not eligible:
            continue

        worst_proxy = min(r["proxy_gain"] for r in seq)
        complexity = float(sum(abs(x) for x in params))
        score = (
            worst_proxy,
            pg,
            1.0 - rmse_ratio,
            tg,
            -complexity,
        )

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "params": params,
                "candidate": cand,
                "seq": seq,
                "positive_proxy": int(positive),
                "nonnegative_trend": int(nonneg_trend),
                "max_seq_rmse_ratio": float(max_seq_rmse),
                "worst_trend_gain": float(worst_trend),
                "pooled_rmse_ratio": rmse_ratio,
                "pooled_proxy_gain": pg,
                "pooled_trend_gain": tg,
            }

    return best


def flat_rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def mean_proxy_and_trend(truth, pred, anchors):
    proxies = []
    trends = []
    for j in range(len(TARGET_COLUMNS)):
        m = evaluate_rows(
            truth[:, :, j],
            pred[:, :, j],
            anchors[:, j],
        )
        proxies.append(float(m["proxy_loss"]))
        trends.append(float(m["trend_core"]))
    return float(np.mean(proxies)), float(np.mean(trends))


def main() -> int:
    print("=" * 118)
    print("FA52.895 + TEMPERATURE-ONLY MSF-LITE OFFLINE DIAGNOSTIC")
    print("=" * 118)
    print("baseline         : frozen Full Architecture V1")
    print("target           : temperature_c only")
    print("short scale      : LOCKED at 0")
    print(f"segments         : {SEGMENTS}")
    print(f"mid/long grid    : {GAIN_GRID}")
    print("first 32 steps   : EXACT Full Architecture")
    print("production 8800  : untouched")

    names, histories_df, truth, anchors = load_sequences()
    base, timings = predict_full_arch(histories_df)

    base_flat = flat_rmse(truth, base)
    print(f"\nFull-Arch local flat RMSE: {base_flat:.6f}")
    print(f"Frozen reference         : {EXPECTED_FA_FLAT_RMSE:.6f}")
    print(f"mean predict time        : {np.nanmean(timings):.3f}s")

    if abs(base_flat - EXPECTED_FA_FLAT_RMSE) > BASELINE_RMSE_TOL:
        raise RuntimeError(
            "Full Architecture baseline integrity check failed: "
            f"got={base_flat:.6f} expected={EXPECTED_FA_FLAT_RMSE:.6f}"
        )
    print("BASELINE INTEGRITY: PASS")

    loso_rows, loso_pass = evaluate_loso(
        names, truth, base, anchors
    )
    fixed = choose_fixed(truth, base, anchors)

    if fixed is None:
        print("\nFIXED CANDIDATE: NONE")
        print("FA52 MSF-LITE GATE: REJECT")
        payload = {
            "version": "fa52-temp-msf-lite-v1",
            "baseline": "full_arch_dynamic_gate_v1",
            "baseline_flat_rmse": base_flat,
            "loso_pass": bool(loso_pass),
            "fixed_candidate": None,
            "global_gate_pass": False,
        }
        OUT_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 2

    cand = fixed["candidate"]
    cand_flat = flat_rmse(truth, cand)
    base_proxy, base_trend = mean_proxy_and_trend(truth, base, anchors)
    cand_proxy, cand_trend = mean_proxy_and_trend(truth, cand, anchors)

    global_rmse_ratio = cand_flat / max(base_flat, EPS)
    global_proxy_gain = (
        (base_proxy - cand_proxy) / max(abs(base_proxy), EPS)
    )
    global_trend_gain = cand_trend - base_trend

    global_pass = bool(
        loso_pass
        and global_rmse_ratio <= GLOBAL_MAX_FLAT_RMSE_RATIO
        and global_proxy_gain >= GLOBAL_MIN_PROXY_GAIN
        and global_trend_gain >= GLOBAL_MIN_TREND_GAIN - 1e-12
    )

    print("\n" + "=" * 118)
    print("FIXED TEMPERATURE CANDIDATE")
    print("=" * 118)
    print(f"params                  : {fixed['params']}")
    print(
        "interpretation          : "
        "(mid32-64, long32-64, mid64-96, long64-96)"
    )
    print(
        f"positive proxy          : "
        f"{fixed['positive_proxy']}/{len(names)}"
    )
    print(
        f"nonnegative trend       : "
        f"{fixed['nonnegative_trend']}/{len(names)}"
    )
    print(
        f"max seq RMSE ratio      : "
        f"{fixed['max_seq_rmse_ratio']:.6f}"
    )
    print(
        f"worst seq trend gain    : "
        f"{fixed['worst_trend_gain']:+.6f}"
    )
    print(
        f"temperature RMSE ratio  : "
        f"{fixed['pooled_rmse_ratio']:.6f}"
    )
    print(
        f"temperature proxy gain  : "
        f"{100*fixed['pooled_proxy_gain']:+.2f}%"
    )
    print(
        f"temperature trend gain  : "
        f"{fixed['pooled_trend_gain']:+.6f}"
    )

    print("\n" + "=" * 118)
    print("GLOBAL SAFETY GATE")
    print("=" * 118)
    print(f"FA52 flat RMSE          : {base_flat:.6f}")
    print(f"candidate flat RMSE     : {cand_flat:.6f}")
    print(f"global RMSE ratio       : {global_rmse_ratio:.6f}")
    print(f"global proxy gain       : {100*global_proxy_gain:+.2f}%")
    print(f"global trend gain       : {global_trend_gain:+.6f}")
    print(f"LOSO gate               : {'PASS' if loso_pass else 'REJECT'}")
    print(f"FA52 MSF-LITE GATE      : {'PASS' if global_pass else 'REJECT'}")

    payload = {
        "version": "fa52-temp-msf-lite-v1",
        "baseline": "full_arch_dynamic_gate_v1",
        "target": TEMP_NAME,
        "windows": list(WINDOWS),
        "segments": [list(x) for x in SEGMENTS],
        "short_scale_gain": 0.0,
        "gain_grid": list(GAIN_GRID),
        "params": list(fixed["params"]),
        "fixed": {
            "positive_proxy": fixed["positive_proxy"],
            "nonnegative_trend": fixed["nonnegative_trend"],
            "max_seq_rmse_ratio": fixed["max_seq_rmse_ratio"],
            "worst_trend_gain": fixed["worst_trend_gain"],
            "temperature_rmse_ratio": fixed["pooled_rmse_ratio"],
            "temperature_proxy_gain": fixed["pooled_proxy_gain"],
            "temperature_trend_gain": fixed["pooled_trend_gain"],
        },
        "loso_pass": bool(loso_pass),
        "loso_rows": loso_rows,
        "global": {
            "baseline_flat_rmse": base_flat,
            "candidate_flat_rmse": cand_flat,
            "flat_rmse_ratio": global_rmse_ratio,
            "proxy_gain": global_proxy_gain,
            "trend_gain": global_trend_gain,
        },
        "global_gate_pass": global_pass,
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT_NPZ,
        truth=truth.astype(np.float32),
        baseline=base.astype(np.float32),
        candidate=cand.astype(np.float32),
        anchors=anchors.astype(np.float32),
    )

    print(f"candidate config        : {OUT_JSON}")
    print(f"candidate predictions   : {OUT_NPZ}")
    print("production 8800         : UNTOUCHED")
    return 0 if global_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
