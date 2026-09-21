from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.inference import predict_future as predict_v8
import src.full_arch_runtime as fa
from v9_analog_multiscale_diagnostic import evaluate_rows

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "full_arch_frozen_gate_v1.json"
OUT_DIR = ROOT / "artifacts" / "fa52_gate_shrink"
OUT_JSON = OUT_DIR / "selected_gate_shrink.json"
OUT_NPZ = OUT_DIR / "gate_shrink_predictions.npz"

ENABLED = ["speed_rpm", "acoustic_db", "pressure_kpa"]
SCALE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

EXPECTED_V8_RMSE = 8.072473
EXPECTED_FA_RMSE = 7.801844
RMSE_TOL = 5e-3

# We deliberately permit only a very small local loss versus frozen FA52.895
# while searching for a simpler gate. Hidden official feedback showed that the
# full gate gave only a tiny accuracy gain but reduced trend/robustness, so the
# purpose here is controlled shrinkage, not a new expert.
MAX_GLOBAL_RMSE_RATIO_TO_FA = 1.005
MAX_GLOBAL_PROXY_RATIO_TO_FA = 1.005
MAX_SEQ_RMSE_RATIO_TO_FA = 1.015

EPS = 1e-12


def load_sequences():
    names, hdfs, truth, anchors = [], [], [], []
    for seq_dir in sorted(Path(DATA_DIR).glob("sequence*")):
        hp, fp = seq_dir / "history.csv", seq_dir / "future.csv"
        if not hp.is_file() or not fp.is_file():
            continue
        hdf = pd.read_csv(hp)[TARGET_COLUMNS]
        fdf = pd.read_csv(fp)[TARGET_COLUMNS].iloc[:HORIZON]
        clean = clean_sequence(hdf)
        y = fdf.to_numpy(dtype=np.float64)
        if y.shape != (HORIZON, len(TARGET_COLUMNS)):
            raise RuntimeError(f"{seq_dir.name}: future shape={y.shape}")
        names.append(seq_dir.name)
        hdfs.append(hdf)
        truth.append(y)
        anchors.append(clean.iloc[-1].to_numpy(dtype=np.float64))
    if len(names) < 5:
        raise RuntimeError(f"expected >=5 sequences, got {len(names)}")
    return names, hdfs, np.stack(truth), np.stack(anchors)


def predict_bases(hdfs):
    fa.preload_full_arch_runtime()
    v8, full = [], []
    for i, hdf in enumerate(hdfs):
        print(f"predict [{i+1}/{len(hdfs)}]", flush=True)
        pv8 = np.asarray(predict_v8(hdf, return_timings=False), dtype=np.float64)
        pfa = np.asarray(fa.predict_future(hdf, return_timings=False), dtype=np.float64)
        v8.append(pv8)
        full.append(pfa)
    return np.stack(v8), np.stack(full)


def flat_rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mean_metrics(truth, pred, anchors):
    proxy = []
    trend = []
    rmse = []
    for j in range(len(TARGET_COLUMNS)):
        m = evaluate_rows(
            truth[:, :, j],
            pred[:, :, j],
            anchors[:, j],
        )
        proxy.append(float(m["proxy_loss"]))
        trend.append(float(m["trend_core"]))
        rmse.append(float(m["rmse"]))
    return {
        "proxy": float(np.mean(proxy)),
        "trend": float(np.mean(trend)),
        "target_rmse": rmse,
    }


def seq_flat_rmse(truth, pred):
    return np.sqrt(np.mean((truth - pred) ** 2, axis=(1, 2)))


def candidate_from_scales(v8, full, scales):
    out = v8.copy()
    for name, scale in zip(ENABLED, scales):
        j = TARGET_COLUMNS.index(name)
        out[:, :, j] = (
            v8[:, :, j]
            + float(scale) * (full[:, :, j] - v8[:, :, j])
        )
    return out


def scale_config(scales):
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    mapping = dict(zip(ENABLED, scales))
    enabled_targets = []
    for target in ENABLED:
        scale = float(mapping[target])
        tcfg = cfg["targets"][target]
        for seg in tcfg.get("segments", []):
            seg["alpha"] = float(seg["alpha"]) * scale
            seg["shrink_scale"] = scale
        if scale > 1e-12:
            enabled_targets.append(target)
            tcfg["enabled"] = True
        else:
            tcfg["enabled"] = False
            tcfg["reason"] = "fa52-safe target shrink scale=0"
    cfg["version"] = "full_arch_gate_v1_shrunk"
    cfg["candidate_name"] = "fa52_gate_shrink_v1"
    cfg["enabled_targets"] = enabled_targets
    cfg["shrink_scales"] = mapping
    return cfg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("FA52.895 EXISTING ANALOG-GATE SHRINK ABLATION")
    print("=" * 118)
    print("baseline  : frozen Full Architecture V1")
    print("targets   :", ENABLED)
    print("grid      :", SCALE_GRID)
    print("meaning   : 0=exact V8 for target, 1=frozen FA52.895 gate")
    print("production: untouched")

    names, hdfs, truth, anchors = load_sequences()
    v8, full = predict_bases(hdfs)

    v8_rmse = flat_rmse(truth, v8)
    fa_rmse = flat_rmse(truth, full)
    print(f"\nV8 flat RMSE : {v8_rmse:.6f}")
    print(f"FA flat RMSE : {fa_rmse:.6f}")
    if abs(v8_rmse - EXPECTED_V8_RMSE) > RMSE_TOL:
        raise RuntimeError(
            f"V8 integrity failed {v8_rmse:.6f} vs {EXPECTED_V8_RMSE:.6f}"
        )
    if abs(fa_rmse - EXPECTED_FA_RMSE) > RMSE_TOL:
        raise RuntimeError(
            f"FA integrity failed {fa_rmse:.6f} vs {EXPECTED_FA_RMSE:.6f}"
        )
    print("BASELINE INTEGRITY: PASS")

    v8_m = mean_metrics(truth, v8, anchors)
    fa_m = mean_metrics(truth, full, anchors)
    fa_seq_rmse = seq_flat_rmse(truth, full)

    rows = []
    for scales in itertools.product(SCALE_GRID, repeat=len(ENABLED)):
        cand = candidate_from_scales(v8, full, scales)
        rmse = flat_rmse(truth, cand)
        mm = mean_metrics(truth, cand, anchors)
        seq_rmse = seq_flat_rmse(truth, cand)

        row = {
            "scales": tuple(float(x) for x in scales),
            "rmse": rmse,
            "rmse_ratio_fa": rmse / max(fa_rmse, EPS),
            "rmse_ratio_v8": rmse / max(v8_rmse, EPS),
            "proxy": mm["proxy"],
            "proxy_ratio_fa": mm["proxy"] / max(fa_m["proxy"], EPS),
            "trend": mm["trend"],
            "trend_vs_fa": mm["trend"] - fa_m["trend"],
            "trend_vs_v8": mm["trend"] - v8_m["trend"],
            "max_seq_rmse_ratio_fa": float(
                np.max(seq_rmse / np.maximum(fa_seq_rmse, EPS))
            ),
            "scale_sum": float(sum(scales)),
            "active_targets": int(sum(float(x) > 1e-12 for x in scales)),
        }
        rows.append(row)

    # Key interpretable ablations.
    key = {
        (1.0, 1.0, 1.0): "FULL_FA",
        (0.0, 0.0, 0.0): "EXACT_V8",
        (0.0, 0.0, 1.0): "PRESSURE_ONLY",
        (1.0, 0.0, 1.0): "SPEED+PRESSURE",
        (0.0, 1.0, 1.0): "ACOUSTIC+PRESSURE",
        (0.5, 0.5, 1.0): "HALF_SPEED_ACOUSTIC+PRESSURE",
        (0.25, 0.25, 1.0): "QUARTER_SPEED_ACOUSTIC+PRESSURE",
        (0.5, 0.5, 0.75): "BALANCED_SHRINK",
    }

    print("\n" + "-" * 118)
    print("KEY ABLATIONS")
    print("-" * 118)
    by_scales = {r["scales"]: r for r in rows}
    for scales, label in key.items():
        r = by_scales[scales]
        print(
            f"{label:34s} scales={scales} "
            f"RMSE={r['rmse']:.6f} "
            f"vsFA={r['rmse_ratio_fa']:.6f} "
            f"proxy_vsFA={r['proxy_ratio_fa']:.6f} "
            f"trend_vsFA={r['trend_vs_fa']:+.6f} "
            f"maxSeqVsFA={r['max_seq_rmse_ratio_fa']:.6f}"
        )

    eligible = [
        r for r in rows
        if r["scales"] != (1.0, 1.0, 1.0)
        and r["rmse_ratio_fa"] <= MAX_GLOBAL_RMSE_RATIO_TO_FA
        and r["proxy_ratio_fa"] <= MAX_GLOBAL_PROXY_RATIO_TO_FA
        and r["max_seq_rmse_ratio_fa"] <= MAX_SEQ_RMSE_RATIO_TO_FA
    ]

    # Maximal safe shrink first, then lower RMSE/proxy. The hidden official
    # result is the reason for preferring a simpler gate once local loss is
    # tightly bounded.
    eligible.sort(
        key=lambda r: (
            r["scale_sum"],
            r["active_targets"],
            r["rmse_ratio_fa"],
            r["proxy_ratio_fa"],
            -r["trend_vs_fa"],
        )
    )

    print("\n" + "-" * 118)
    print("TOP SAFE-SHRINK CANDIDATES")
    print("-" * 118)
    for r in eligible[:15]:
        print(
            f"scales={r['scales']} sum={r['scale_sum']:.2f} "
            f"active={r['active_targets']} "
            f"RMSE={r['rmse']:.6f} "
            f"vsFA={r['rmse_ratio_fa']:.6f} "
            f"proxy_vsFA={r['proxy_ratio_fa']:.6f} "
            f"trend_vsFA={r['trend_vs_fa']:+.6f} "
            f"maxSeqVsFA={r['max_seq_rmse_ratio_fa']:.6f}"
        )

    if not eligible:
        print("\nFA52 GATE SHRINK: NO SAFE CANDIDATE")
        return 2

    best = eligible[0]
    cfg = scale_config(best["scales"])
    cfg["offline_ablation"] = {
        "v8_flat_rmse": v8_rmse,
        "fa_flat_rmse": fa_rmse,
        "candidate_flat_rmse": best["rmse"],
        "rmse_ratio_to_fa": best["rmse_ratio_fa"],
        "proxy_ratio_to_fa": best["proxy_ratio_fa"],
        "trend_vs_fa": best["trend_vs_fa"],
        "max_sequence_rmse_ratio_to_fa": best["max_seq_rmse_ratio_fa"],
    }
    OUT_JSON.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best_pred = candidate_from_scales(v8, full, best["scales"])
    np.savez_compressed(
        OUT_NPZ,
        truth=truth.astype(np.float32),
        v8=v8.astype(np.float32),
        full=full.astype(np.float32),
        candidate=best_pred.astype(np.float32),
        anchors=anchors.astype(np.float32),
    )

    print("\n" + "=" * 118)
    print("SELECTED SAFE SHRINK")
    print("=" * 118)
    print("scales                    :", best["scales"])
    print("order                     :", ENABLED)
    print(f"scale sum                 : {best['scale_sum']:.2f}")
    print(f"candidate flat RMSE       : {best['rmse']:.6f}")
    print(f"RMSE ratio to FA          : {best['rmse_ratio_fa']:.6f}")
    print(f"proxy ratio to FA         : {best['proxy_ratio_fa']:.6f}")
    print(f"trend vs FA               : {best['trend_vs_fa']:+.6f}")
    print(f"max seq RMSE ratio to FA  : {best['max_seq_rmse_ratio_fa']:.6f}")
    print("candidate config          :", OUT_JSON)
    print("candidate predictions     :", OUT_NPZ)
    print("production 8800           : UNTOUCHED")
    print("FA52 GATE SHRINK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
