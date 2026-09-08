from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.feature_engineer import robust_trend_forecast
from src.inference import predict_future as predict_future_v8
from src.trend_pattern import adaptive_pattern_forecast
from v83_final_sprint_diagnostic import apply_params, evaluate_rows


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
SOURCE_PATH = ROOT / "models" / "v83_final_candidate.json"
OUTPUT_PATH = ROOT / "models" / "v83_final_safe_candidate.json"
SHRINK_GRID = (0.25, 0.50, 0.75, 1.00)

# Final-submission rule: a changed target must be positive on every provided sequence.
MAX_HOLDOUT_RMSE_RATIO = 1.020
MIN_POOLED_PROXY_GAIN_PCT = 0.30
MIN_POOLED_TREND_GAIN = 0.002


def _gain_pct(base: float, cand: float) -> float:
    return 100.0 * (base - cand) / max(abs(base), 1e-12)


def main() -> int:
    if not CORPUS_PATH.is_file() or not SOURCE_PATH.is_file():
        raise FileNotFoundError("run bash run_v83_final_sprint.sh first")

    source_cfg = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    if not bool(source_cfg.get("offline_gate_pass", False)):
        raise RuntimeError("source V8.3 candidate is not PASS")

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
    anchors = history[:, -1, :]
    n = len(group_id)

    v8 = np.empty_like(truth)
    full = np.empty_like(truth)
    for i, gid in enumerate(group_id):
        print(f"build [{i+1}/{n}] {gid} ...", flush=True)
        hdf = pd.DataFrame(history[i], columns=TARGET_COLUMNS)
        v8[i] = np.asarray(predict_future_v8(hdf), dtype=np.float64)
        pattern = np.asarray(adaptive_pattern_forecast(hdf, HORIZON), dtype=np.float64)
        trend = np.asarray(robust_trend_forecast(hdf, HORIZON), dtype=np.float64)
        full[i] = v8[i]
        for name in source_cfg.get("enabled_targets", []):
            j = TARGET_COLUMNS.index(name)
            c = source_cfg["targets"][name]["consensus"]
            params = (
                float(c["pattern_weight"]),
                float(c["trend_weight"]),
                float(c["highpass_gain"]),
                float(c["amplitude_gain"]),
                int(c["highpass_window"]),
            )
            full[i, :, j] = apply_params(
                v8[i, :, j], pattern[:, j], trend[:, j], float(anchors[i, j]), params
            )

    final = v8.copy()
    target_rows = {}
    enabled = []

    print("\n" + "=" * 112)
    print("V8.3 FINAL SAFETY SHRINK - REQUIRE 5/5 POSITIVE HOLDOUTS")
    print("=" * 112)

    for j, name in enumerate(TARGET_COLUMNS):
        base_all = evaluate_rows(truth[:, :, j], v8[:, :, j], anchors[:, j])
        source_enabled = name in source_cfg.get("enabled_targets", [])
        best = None

        if source_enabled:
            for alpha in SHRINK_GRID:
                cand = v8[:, :, j] + float(alpha) * (full[:, :, j] - v8[:, :, j])
                all_m = evaluate_rows(truth[:, :, j], cand, anchors[:, j])
                pooled_proxy = _gain_pct(base_all["proxy_loss"], all_m["proxy_loss"])
                pooled_trend = all_m["trend_core"] - base_all["trend_core"]

                folds = []
                for i, gid in enumerate(group_id):
                    bm = evaluate_rows(
                        truth[i:i+1, :, j], v8[i:i+1, :, j], anchors[i:i+1, j]
                    )
                    cm = evaluate_rows(
                        truth[i:i+1, :, j], cand[i:i+1, :], anchors[i:i+1, j]
                    )
                    pg = _gain_pct(bm["proxy_loss"], cm["proxy_loss"])
                    rr = cm["rmse"] / max(bm["rmse"], 1e-12)
                    tg = cm["trend_core"] - bm["trend_core"]
                    folds.append({
                        "group": str(gid),
                        "proxy_gain_pct": pg,
                        "rmse_ratio": rr,
                        "trend_gain": tg,
                    })

                positive = sum(x["proxy_gain_pct"] > 0.0 for x in folds)
                max_rr = max(x["rmse_ratio"] for x in folds)
                eligible = bool(
                    positive == n
                    and max_rr <= MAX_HOLDOUT_RMSE_RATIO
                    and pooled_proxy >= MIN_POOLED_PROXY_GAIN_PCT
                    and pooled_trend >= MIN_POOLED_TREND_GAIN
                )
                print(
                    f"{name:16s} alpha={alpha:.2f} positive={positive}/{n} "
                    f"max_rmse_ratio={max_rr:.4f} pooled_proxy={pooled_proxy:+.2f}% "
                    f"pooled_trend={pooled_trend:+.4f} {'ELIGIBLE' if eligible else ''}"
                )
                if eligible and (best is None or pooled_proxy > best["pooled_proxy_gain_pct"]):
                    best = {
                        "runtime_scale": float(alpha),
                        "pooled_proxy_gain_pct": float(pooled_proxy),
                        "pooled_trend_gain": float(pooled_trend),
                        "max_holdout_rmse_ratio": float(max_rr),
                        "folds": folds,
                    }

        if best is not None:
            alpha = best["runtime_scale"]
            final[:, :, j] = v8[:, :, j] + alpha * (full[:, :, j] - v8[:, :, j])
            enabled.append(name)
            consensus = source_cfg["targets"][name]["consensus"]
        else:
            consensus = None

        target_rows[name] = {
            "enabled": best is not None,
            "runtime_scale": 0.0 if best is None else best["runtime_scale"],
            "consensus": consensus,
            "safety": best,
        }
        print(
            f"  => {name}: {'ENABLE alpha='+format(best['runtime_scale'], '.2f') if best else 'KEEP EXACT V8'}"
        )

    base_metrics = [
        evaluate_rows(truth[:, :, j], v8[:, :, j], anchors[:, j])
        for j in range(len(TARGET_COLUMNS))
    ]
    final_metrics = [
        evaluate_rows(truth[:, :, j], final[:, :, j], anchors[:, j])
        for j in range(len(TARGET_COLUMNS))
    ]
    base_proxy = float(np.mean([m["proxy_loss"] for m in base_metrics]))
    final_proxy = float(np.mean([m["proxy_loss"] for m in final_metrics]))
    proxy_gain = _gain_pct(base_proxy, final_proxy)
    base_flat = float(np.sqrt(np.mean((v8 - truth) ** 2)))
    final_flat = float(np.sqrt(np.mean((final - truth) ** 2)))
    rmse_ratio = final_flat / max(base_flat, 1e-12)

    gate = bool(len(enabled) >= 1 and proxy_gain >= 0.30 and np.isfinite(final).all())
    print("\n" + "=" * 112)
    print("V8.3 SAFE FINAL DECISION")
    print("=" * 112)
    print(f"enabled targets      : {enabled}")
    print(f"flat RMSE ratio      : {rmse_ratio:.6f}")
    print(f"mean proxy gain      : {proxy_gain:+.2f}%")
    print(f"V8.3 SAFE GATE       : {'PASS' if gate else 'REJECT'}")

    out = {
        "candidate": "V8.3 safe final with per-target shrink",
        "base_version": int(source_cfg.get("base_version", 0)),
        "enabled_targets": enabled,
        "targets": target_rows,
        "global_rmse_ratio": rmse_ratio,
        "global_proxy_gain_pct": proxy_gain,
        "offline_gate_pass": gate,
        "source_candidate": str(SOURCE_PATH.name),
        "rule": "each changed target requires 5/5 positive provided-sequence proxy gain and <=2% worst-holdout RMSE degradation",
    }
    OUTPUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics              : {OUTPUT_PATH}")
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
