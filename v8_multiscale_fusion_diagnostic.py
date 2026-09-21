from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.inference import predict_future
from src.trajectory_fusion import endpoint_zero_highpass
from v9_analog_multiscale_diagnostic import evaluate_rows, proxy_gain


OUTPUT_JSON = MODEL_DIR / "v8_multiscale_fusion_candidate.json"
OUTPUT_NPZ = MODEL_DIR / "v8_multiscale_fusion_candidate.npz"

# Frequency decomposition of the FINAL V8 trajectory.
# hp(5)                    -> short/high-frequency detail
# hp(13) - hp(5)           -> medium-scale detail
# hp(33) - hp(13)          -> long/local-shape detail
WINDOWS = (5, 13, 33)

# Conservative gains around exact V8. Zero is always an available fallback.
GAIN_GRID = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)

# Per-target fixed-candidate safety gates.
MIN_POSITIVE_PROXY_SEQS = 4
MIN_NONNEG_TREND_SEQS = 3
MAX_SEQ_RMSE_RATIO = 1.01
MAX_POOLED_RMSE_RATIO = 0.998
MIN_POOLED_PROXY_GAIN = 0.005

# LOSO safety gate: parameter is selected on four sequences and checked on the fifth.
MIN_LOSO_POSITIVE_PROXY = 4
MIN_LOSO_NONNEG_TREND = 3
MAX_LOSO_RMSE_RATIO = 1.01

# Final combined-candidate gate.
MAX_GLOBAL_FLAT_RMSE_RATIO = 1.0
MIN_GLOBAL_PROXY_GAIN = 0.002
MAX_GLOBAL_TREND_DROP = 1e-12

EPS = 1e-12


def load_official_sequences():
    names = []
    histories = []
    futures = []

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
                f"{seq_dir.name}: missing history={missing_h} future={missing_f}"
            )

        clean = clean_sequence(hdf[TARGET_COLUMNS])
        h = clean[TARGET_COLUMNS].to_numpy(dtype=np.float64)
        f = fdf[TARGET_COLUMNS].iloc[:HORIZON].to_numpy(dtype=np.float64)

        if f.shape != (HORIZON, len(TARGET_COLUMNS)):
            raise RuntimeError(f"{seq_dir.name}: future shape={f.shape}")
        if not np.isfinite(f).all():
            raise RuntimeError(f"{seq_dir.name}: future contains NaN/Inf")

        names.append(seq_dir.name)
        histories.append(h)
        futures.append(f)

    if len(names) < 5:
        raise RuntimeError(f"expected >=5 sequences, got {len(names)}")
    return names, histories, np.stack(futures)


def compute_v8(histories):
    out = np.empty(
        (len(histories), HORIZON, len(TARGET_COLUMNS)),
        dtype=np.float64,
    )
    for i, h in enumerate(histories):
        print(f"V8 predict [{i+1}/{len(histories)}]", flush=True)
        hdf = pd.DataFrame(h, columns=TARGET_COLUMNS)
        out[i] = np.asarray(predict_future(hdf), dtype=np.float64)
    return out


def make_bands(pred_1d):
    x = np.asarray(pred_1d, dtype=np.float64)
    hp5 = endpoint_zero_highpass(x, WINDOWS[0])
    hp13 = endpoint_zero_highpass(x, WINDOWS[1])
    hp33 = endpoint_zero_highpass(x, WINDOWS[2])

    short = hp5
    medium = hp13 - hp5
    long_local = hp33 - hp13
    return short, medium, long_local


def apply_gains(base, gains):
    short, medium, long_local = make_bands(base)
    gs, gm, gl = (float(x) for x in gains)
    return base + gs * short + gm * medium + gl * long_local


def pooled_metrics(truth, pred, anchors, idx, target_idx):
    return evaluate_rows(
        truth[idx, :, target_idx],
        pred[idx, :, target_idx],
        anchors[idx, target_idx],
    )


def seq_metrics(truth, base, cand, anchors, target_idx):
    rows = []
    for i in range(len(truth)):
        b = evaluate_rows(
            truth[i:i+1, :, target_idx],
            base[i:i+1, :, target_idx],
            anchors[i:i+1, target_idx],
        )
        c = evaluate_rows(
            truth[i:i+1, :, target_idx],
            cand[i:i+1, :, target_idx],
            anchors[i:i+1, target_idx],
        )
        rows.append(
            {
                "proxy_gain": proxy_gain(b, c),
                "trend_gain": float(c["trend_core"] - b["trend_core"]),
                "rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
            }
        )
    return rows


def candidate_cube(base_target, gains):
    out = np.empty_like(base_target, dtype=np.float64)
    for i in range(len(base_target)):
        out[i] = apply_gains(base_target[i], gains)
    return out


def train_select(truth, v8, anchors, train_idx, target_idx):
    base_train = pooled_metrics(truth, v8, anchors, train_idx, target_idx)
    best = None

    for gains in itertools.product(GAIN_GRID, repeat=3):
        if gains == (0.0, 0.0, 0.0):
            continue

        cand_target = candidate_cube(v8[:, :, target_idx], gains)
        cand_train = evaluate_rows(
            truth[train_idx, :, target_idx],
            cand_target[train_idx],
            anchors[train_idx, target_idx],
        )

        rmse_ratio = cand_train["rmse"] / max(base_train["rmse"], EPS)
        pg = proxy_gain(base_train, cand_train)
        trend_gain = cand_train["trend_core"] - base_train["trend_core"]

        if rmse_ratio > 0.9995:
            continue
        if pg < 0.0025:
            continue
        if trend_gain < -1e-12:
            continue

        # Prefer proxy gain, then RMSE, then smaller correction magnitude.
        complexity = sum(abs(float(x)) for x in gains)
        score = (
            pg,
            1.0 - rmse_ratio,
            trend_gain,
            -complexity,
        )
        if best is None or score > best[0]:
            best = (score, tuple(float(x) for x in gains))

    return None if best is None else best[1]


def evaluate_loso(truth, v8, anchors, target_idx):
    rows = []

    for holdout in range(len(truth)):
        train_idx = np.asarray(
            [i for i in range(len(truth)) if i != holdout],
            dtype=np.int64,
        )
        gains = train_select(truth, v8, anchors, train_idx, target_idx)
        if gains is None:
            rows.append(
                {
                    "holdout": holdout,
                    "selected": None,
                    "proxy_gain": 0.0,
                    "trend_gain": 0.0,
                    "rmse_ratio": 1.0,
                }
            )
            continue

        base = v8[holdout:holdout+1, :, target_idx]
        cand = candidate_cube(base, gains)
        b = evaluate_rows(
            truth[holdout:holdout+1, :, target_idx],
            base,
            anchors[holdout:holdout+1, target_idx],
        )
        c = evaluate_rows(
            truth[holdout:holdout+1, :, target_idx],
            cand,
            anchors[holdout:holdout+1, target_idx],
        )
        rows.append(
            {
                "holdout": holdout,
                "selected": gains,
                "proxy_gain": proxy_gain(b, c),
                "trend_gain": float(c["trend_core"] - b["trend_core"]),
                "rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
            }
        )

    positive = sum(r["proxy_gain"] > 0.0 for r in rows)
    nonneg_trend = sum(r["trend_gain"] >= -1e-12 for r in rows)
    max_rmse_ratio = max(r["rmse_ratio"] for r in rows)

    passed = bool(
        positive >= MIN_LOSO_POSITIVE_PROXY
        and nonneg_trend >= MIN_LOSO_NONNEG_TREND
        and max_rmse_ratio <= MAX_LOSO_RMSE_RATIO
    )
    return rows, passed


def choose_fixed(truth, v8, anchors, target_idx):
    all_idx = np.arange(len(truth), dtype=np.int64)
    base_pool = pooled_metrics(truth, v8, anchors, all_idx, target_idx)

    best = None
    for gains in itertools.product(GAIN_GRID, repeat=3):
        if gains == (0.0, 0.0, 0.0):
            continue

        cand_target = candidate_cube(v8[:, :, target_idx], gains)
        cand_pool = evaluate_rows(
            truth[:, :, target_idx],
            cand_target,
            anchors[:, target_idx],
        )
        rows = seq_metrics(
            truth,
            v8[:, :, target_idx],
            cand_target,
            anchors,
            target_idx,
        )

        positive = sum(r["proxy_gain"] > 0.0 for r in rows)
        nonneg_trend = sum(r["trend_gain"] >= -1e-12 for r in rows)
        max_seq_ratio = max(r["rmse_ratio"] for r in rows)
        pooled_ratio = cand_pool["rmse"] / max(base_pool["rmse"], EPS)
        pooled_pg = proxy_gain(base_pool, cand_pool)
        pooled_trend_gain = cand_pool["trend_core"] - base_pool["trend_core"]

        eligible = bool(
            positive >= MIN_POSITIVE_PROXY_SEQS
            and nonneg_trend >= MIN_NONNEG_TREND_SEQS
            and max_seq_ratio <= MAX_SEQ_RMSE_RATIO
            and pooled_ratio <= MAX_POOLED_RMSE_RATIO
            and pooled_pg >= MIN_POOLED_PROXY_GAIN
            and pooled_trend_gain >= -1e-12
        )
        if not eligible:
            continue

        worst_pg = min(r["proxy_gain"] for r in rows)
        complexity = sum(abs(float(x)) for x in gains)
        score = (
            worst_pg,
            pooled_pg,
            1.0 - pooled_ratio,
            pooled_trend_gain,
            -complexity,
        )

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "gains": tuple(float(x) for x in gains),
                "prediction": cand_target,
                "base_pool": base_pool,
                "cand_pool": cand_pool,
                "rows": rows,
                "positive": int(positive),
                "nonneg_trend": int(nonneg_trend),
                "max_seq_rmse_ratio": float(max_seq_ratio),
                "pooled_rmse_ratio": float(pooled_ratio),
                "pooled_proxy_gain": float(pooled_pg),
                "pooled_trend_gain": float(pooled_trend_gain),
            }

    return best


def combined_flat_rmse(truth, pred):
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mean_proxy(truth, pred, anchors):
    values = []
    for j in range(len(TARGET_COLUMNS)):
        values.append(
            evaluate_rows(
                truth[:, :, j],
                pred[:, :, j],
                anchors[:, j],
            )["proxy_loss"]
        )
    return float(np.mean(values))


def mean_trend_core(truth, pred, anchors):
    values = []
    for j in range(len(TARGET_COLUMNS)):
        values.append(
            evaluate_rows(
                truth[:, :, j],
                pred[:, :, j],
                anchors[:, j],
            )["trend_core"]
        )
    return float(np.mean(values))


def main() -> int:
    print("=" * 116)
    print("V8 MULTI-SCALE FUSION DIAGNOSTIC")
    print("=" * 116)
    print(f"windows      : {WINDOWS}")
    print(f"gain grid    : {GAIN_GRID}")
    print("decomposition: short=HP5, medium=HP13-HP5, long=HP33-HP13")
    print("safety       : exact V8 fallback for every rejected target")
    print("production   : untouched")

    names, histories, truth = load_official_sequences()
    anchors = np.stack([h[-1] for h in histories], axis=0)
    print(f"sequences    : {names}")

    v8 = compute_v8(histories)
    final = v8.copy()
    selected = {}
    enabled = []

    for j, name in enumerate(TARGET_COLUMNS):
        print("\n" + "-" * 116)
        print(name)
        print("-" * 116)

        loso_rows, loso_pass = evaluate_loso(
            truth, v8, anchors, j
        )
        for row in loso_rows:
            print(
                f"LOSO holdout={names[row['holdout']]} "
                f"gains={row['selected']} "
                f"proxy_gain={100*row['proxy_gain']:+.2f}% "
                f"trend_gain={row['trend_gain']:+.5f} "
                f"rmse_ratio={row['rmse_ratio']:.6f}"
            )

        fixed = choose_fixed(truth, v8, anchors, j)

        if fixed is None:
            print("FIXED: no candidate passed conservative gate -> EXACT V8")
            selected[name] = {
                "enabled": False,
                "reason": "no_fixed_candidate",
            }
            continue

        print(
            "FIXED "
            f"gains={fixed['gains']} "
            f"positive={fixed['positive']}/{len(names)} "
            f"nonneg_trend={fixed['nonneg_trend']}/{len(names)} "
            f"max_seq_rmse_ratio={fixed['max_seq_rmse_ratio']:.6f} "
            f"pooled_rmse_ratio={fixed['pooled_rmse_ratio']:.6f} "
            f"pooled_proxy_gain={100*fixed['pooled_proxy_gain']:+.2f}% "
            f"pooled_trend_gain={fixed['pooled_trend_gain']:+.5f} "
            f"LOSO_PASS={loso_pass}"
        )

        enabled_target = bool(loso_pass)
        selected[name] = {
            "enabled": enabled_target,
            "gains": list(fixed["gains"]),
            "positive_sequences": fixed["positive"],
            "nonnegative_trend_sequences": fixed["nonneg_trend"],
            "max_sequence_rmse_ratio": fixed["max_seq_rmse_ratio"],
            "pooled_rmse_ratio": fixed["pooled_rmse_ratio"],
            "pooled_proxy_gain": fixed["pooled_proxy_gain"],
            "pooled_trend_gain": fixed["pooled_trend_gain"],
            "loso_pass": bool(loso_pass),
            "loso_rows": loso_rows,
        }

        if enabled_target:
            final[:, :, j] = fixed["prediction"]
            enabled.append(name)
            print("DECISION: ENABLE")
        else:
            print("DECISION: REJECT -> EXACT V8")

    base_rmse = combined_flat_rmse(truth, v8)
    cand_rmse = combined_flat_rmse(truth, final)
    base_proxy = mean_proxy(truth, v8, anchors)
    cand_proxy = mean_proxy(truth, final, anchors)
    base_trend = mean_trend_core(truth, v8, anchors)
    cand_trend = mean_trend_core(truth, final, anchors)

    flat_ratio = cand_rmse / max(base_rmse, EPS)
    global_proxy_gain = (
        (base_proxy - cand_proxy) / max(abs(base_proxy), EPS)
    )
    global_trend_gain = cand_trend - base_trend

    global_pass = bool(
        len(enabled) > 0
        and flat_ratio <= MAX_GLOBAL_FLAT_RMSE_RATIO
        and global_proxy_gain >= MIN_GLOBAL_PROXY_GAIN
        and global_trend_gain >= -MAX_GLOBAL_TREND_DROP
    )

    print("\n" + "=" * 116)
    print("FINAL MULTI-SCALE FUSION GATE")
    print("=" * 116)
    print(f"enabled targets       : {enabled}")
    print(f"V8 flat RMSE          : {base_rmse:.6f}")
    print(f"MSF flat RMSE         : {cand_rmse:.6f}")
    print(f"flat RMSE ratio       : {flat_ratio:.6f}")
    print(f"V8 mean proxy         : {base_proxy:.6f}")
    print(f"MSF mean proxy        : {cand_proxy:.6f}")
    print(f"global proxy gain     : {100*global_proxy_gain:+.2f}%")
    print(f"V8 mean trend_core    : {base_trend:.6f}")
    print(f"MSF mean trend_core   : {cand_trend:.6f}")
    print(f"global trend gain     : {global_trend_gain:+.6f}")
    print(f"OFFLINE GATE          : {'PASS' if global_pass else 'REJECT'}")

    payload = {
        "version": "v8-msf-v1",
        "base": "V8 exact output",
        "windows": list(WINDOWS),
        "decomposition": [
            "HP5",
            "HP13-HP5",
            "HP33-HP13",
        ],
        "gain_grid": list(GAIN_GRID),
        "enabled_targets": enabled if global_pass else [],
        "selected": selected,
        "global": {
            "flat_rmse_v8": base_rmse,
            "flat_rmse_candidate": cand_rmse,
            "flat_rmse_ratio": flat_ratio,
            "mean_proxy_v8": base_proxy,
            "mean_proxy_candidate": cand_proxy,
            "global_proxy_gain": global_proxy_gain,
            "mean_trend_v8": base_trend,
            "mean_trend_candidate": cand_trend,
            "global_trend_gain": global_trend_gain,
        },
        "offline_gate_pass": global_pass,
    }

    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        OUTPUT_NPZ,
        truth=truth.astype(np.float32),
        v8=v8.astype(np.float32),
        candidate=final.astype(np.float32),
        anchors=anchors.astype(np.float32),
    )

    print(f"candidate config      : {OUTPUT_JSON}")
    print(f"candidate predictions : {OUTPUT_NPZ}")
    print("models/ensemble_config.pkl was NOT modified.")
    print("production 8800 was NOT modified.")
    return 0 if global_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
