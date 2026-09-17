from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import HORIZON, MODEL_DIR, TARGET_COLUMNS
from src.inference import predict_future as predict_future_v8
from src.v8_runtime import v8_enabled
from v9_analog_multiscale_diagnostic import (
    AnalogConfig,
    EPS,
    analog_predict_one,
    blend,
    build_bank,
    evaluate_rows,
    load_official_sequences,
    proxy_gain,
)


OUTPUT_PATH = MODEL_DIR / "v9_safe_candidate.json"

TEMP_NAME = "temperature_c"
PRESSURE_NAME = "pressure_kpa"

TEMP_CFG = AnalogConfig(context=96, k=8, mode="target", self_guard=96)
PRESSURE_CFG = AnalogConfig(context=144, k=8, mode="multi", self_guard=32)

TEMP_ALPHAS = (0.075, 0.10)
PRESSURE_ALPHAS = (0.05, 0.075)

# Conservative gates. Four untouched targets must remain EXACT V8.
TEMP_MIN_POSITIVE = 5
TEMP_MIN_NONNEG_TREND = 4
TEMP_MAX_RMSE_RATIO = 1.01
TEMP_MAX_POOLED_RMSE_RATIO = 0.99
TEMP_MIN_POOLED_PROXY_GAIN = 0.01

PRESSURE_MIN_POSITIVE = 4
PRESSURE_MIN_NONNEG_TREND = 4
PRESSURE_MAX_RMSE_RATIO = 1.03
PRESSURE_MAX_POOLED_RMSE_RATIO = 0.99
PRESSURE_MIN_POOLED_PROXY_GAIN = 0.01

FINAL_MAX_FLAT_RMSE_RATIO = 1.0
FINAL_MIN_MEAN_PROXY_GAIN = 0.01


def make_analog_prediction(
    histories: Sequence[np.ndarray],
    target_idx: int,
    cfg: AnalogConfig,
) -> np.ndarray:
    bank = build_bank(histories, cfg.context, target_idx, cfg.mode)
    out = np.empty((len(histories), HORIZON), dtype=np.float64)
    for i in range(len(histories)):
        out[i] = analog_predict_one(
            histories=histories,
            query_idx=i,
            target_idx=target_idx,
            cfg=cfg,
            bank=bank,
        )
    return out


def evaluate_target(
    truth: np.ndarray,
    base: np.ndarray,
    cand: np.ndarray,
    anchors: np.ndarray,
    target_idx: int,
) -> Dict[str, object]:
    base_pool = evaluate_rows(
        truth[:, :, target_idx],
        base[:, :, target_idx],
        anchors[:, target_idx],
    )
    cand_pool = evaluate_rows(
        truth[:, :, target_idx],
        cand[:, :, target_idx],
        anchors[:, target_idx],
    )

    rows: List[Dict[str, float]] = []
    for i in range(len(truth)):
        b = evaluate_rows(
            truth[i:i + 1, :, target_idx],
            base[i:i + 1, :, target_idx],
            anchors[i:i + 1, target_idx],
        )
        c = evaluate_rows(
            truth[i:i + 1, :, target_idx],
            cand[i:i + 1, :, target_idx],
            anchors[i:i + 1, target_idx],
        )
        rows.append(
            {
                "proxy_gain": proxy_gain(b, c),
                "trend_gain": float(c["trend_core"] - b["trend_core"]),
                "rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
            }
        )

    positive_proxy = int(sum(r["proxy_gain"] > 0.0 for r in rows))
    nonneg_trend = int(sum(r["trend_gain"] >= -1e-12 for r in rows))
    max_rmse_ratio = float(max(r["rmse_ratio"] for r in rows))
    pooled_rmse_ratio = float(cand_pool["rmse"] / max(base_pool["rmse"], EPS))
    pooled_proxy_gain = float(proxy_gain(base_pool, cand_pool))

    return {
        "rows": rows,
        "positive_proxy": positive_proxy,
        "nonneg_trend": nonneg_trend,
        "max_rmse_ratio": max_rmse_ratio,
        "pooled_rmse_ratio": pooled_rmse_ratio,
        "pooled_proxy_gain": pooled_proxy_gain,
    }


def target_gate(name: str, ev: Dict[str, object]) -> bool:
    if name == TEMP_NAME:
        return bool(
            int(ev["positive_proxy"]) >= TEMP_MIN_POSITIVE
            and int(ev["nonneg_trend"]) >= TEMP_MIN_NONNEG_TREND
            and float(ev["max_rmse_ratio"]) <= TEMP_MAX_RMSE_RATIO
            and float(ev["pooled_rmse_ratio"]) <= TEMP_MAX_POOLED_RMSE_RATIO
            and float(ev["pooled_proxy_gain"]) >= TEMP_MIN_POOLED_PROXY_GAIN
        )
    if name == PRESSURE_NAME:
        return bool(
            int(ev["positive_proxy"]) >= PRESSURE_MIN_POSITIVE
            and int(ev["nonneg_trend"]) >= PRESSURE_MIN_NONNEG_TREND
            and float(ev["max_rmse_ratio"]) <= PRESSURE_MAX_RMSE_RATIO
            and float(ev["pooled_rmse_ratio"]) <= PRESSURE_MAX_POOLED_RMSE_RATIO
            and float(ev["pooled_proxy_gain"]) >= PRESSURE_MIN_POOLED_PROXY_GAIN
        )
    raise ValueError(name)


def global_metrics(
    truth: np.ndarray,
    base: np.ndarray,
    cand: np.ndarray,
    anchors: np.ndarray,
) -> Dict[str, float]:
    base_flat = float(np.sqrt(np.mean((truth - base) ** 2)))
    cand_flat = float(np.sqrt(np.mean((truth - cand) ** 2)))
    flat_ratio = float(cand_flat / max(base_flat, EPS))

    base_proxy = []
    cand_proxy = []
    for j in range(len(TARGET_COLUMNS)):
        b = evaluate_rows(truth[:, :, j], base[:, :, j], anchors[:, j])
        c = evaluate_rows(truth[:, :, j], cand[:, :, j], anchors[:, j])
        base_proxy.append(float(b["proxy_loss"]))
        cand_proxy.append(float(c["proxy_loss"]))

    mean_base_proxy = float(np.mean(base_proxy))
    mean_cand_proxy = float(np.mean(cand_proxy))
    mean_proxy_gain = float(
        (mean_base_proxy - mean_cand_proxy)
        / max(abs(mean_base_proxy), EPS)
    )

    return {
        "flat_rmse_v8": base_flat,
        "flat_rmse_candidate": cand_flat,
        "flat_rmse_ratio": flat_ratio,
        "mean_proxy_v8": mean_base_proxy,
        "mean_proxy_candidate": mean_cand_proxy,
        "mean_proxy_gain": mean_proxy_gain,
    }


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
    print("V9-SAFE DIAGNOSTIC: EXACT V8 + TEMPERATURE/PRESSURE ANALOG ONLY")
    print("=" * 118)
    print(f"sequences           : {names}")
    print(f"V8 version          : {cfg.get('version')}")
    print(f"temperature config  : {TEMP_CFG.key()}")
    print(f"temperature alphas  : {TEMP_ALPHAS}")
    print(f"pressure config     : {PRESSURE_CFG.key()}")
    print(f"pressure alphas     : {PRESSURE_ALPHAS}")
    print("untouched targets   : vibration_rms/current_a/speed_rpm/acoustic_db = EXACT V8")
    print("selection rule      : choose the smallest-alpha passing pair, not the largest gain")

    v8 = np.empty_like(truth)
    for i, name in enumerate(names):
        print(f"V8 predict [{i + 1}/{len(names)}] {name}", flush=True)
        hdf = pd.DataFrame(histories[i], columns=TARGET_COLUMNS)
        v8[i] = np.asarray(predict_future_v8(hdf), dtype=np.float64)

    temp_idx = TARGET_COLUMNS.index(TEMP_NAME)
    pressure_idx = TARGET_COLUMNS.index(PRESSURE_NAME)

    print("\nBuilding frozen analog predictions ...", flush=True)
    temp_analog = make_analog_prediction(histories, temp_idx, TEMP_CFG)
    pressure_analog = make_analog_prediction(histories, pressure_idx, PRESSURE_CFG)

    candidates = []

    for temp_alpha in TEMP_ALPHAS:
        for pressure_alpha in PRESSURE_ALPHAS:
            cand = v8.copy()
            cand[:, :, temp_idx] = blend(
                v8[:, :, temp_idx], temp_analog, temp_alpha
            )
            cand[:, :, pressure_idx] = blend(
                v8[:, :, pressure_idx], pressure_analog, pressure_alpha
            )

            temp_ev = evaluate_target(
                truth, v8, cand, anchors, temp_idx
            )
            pressure_ev = evaluate_target(
                truth, v8, cand, anchors, pressure_idx
            )

            untouched = [
                j for j, name in enumerate(TARGET_COLUMNS)
                if name not in (TEMP_NAME, PRESSURE_NAME)
            ]
            max_untouched_diff = float(
                np.max(np.abs(cand[:, :, untouched] - v8[:, :, untouched]))
            )
            exact_untouched = bool(max_untouched_diff == 0.0)

            g = global_metrics(truth, v8, cand, anchors)
            temp_ok = target_gate(TEMP_NAME, temp_ev)
            pressure_ok = target_gate(PRESSURE_NAME, pressure_ev)
            final_ok = bool(
                temp_ok
                and pressure_ok
                and exact_untouched
                and g["flat_rmse_ratio"] < FINAL_MAX_FLAT_RMSE_RATIO
                and g["mean_proxy_gain"] >= FINAL_MIN_MEAN_PROXY_GAIN
            )

            result = {
                "temperature_alpha": float(temp_alpha),
                "pressure_alpha": float(pressure_alpha),
                "temperature": temp_ev,
                "pressure": pressure_ev,
                "max_untouched_abs_diff": max_untouched_diff,
                "untouched_exact_v8": exact_untouched,
                **g,
                "gate_pass": final_ok,
            }
            candidates.append(result)

            print("\n" + "-" * 118)
            print(
                f"TEMP alpha={temp_alpha:.3f} | PRESSURE alpha={pressure_alpha:.3f}"
            )
            print(
                f"temperature: positive={temp_ev['positive_proxy']}/5 "
                f"nonneg_trend={temp_ev['nonneg_trend']}/5 "
                f"max_rmse_ratio={temp_ev['max_rmse_ratio']:.4f} "
                f"pooled_rmse_ratio={temp_ev['pooled_rmse_ratio']:.4f} "
                f"proxy_gain={100 * temp_ev['pooled_proxy_gain']:+.2f}% "
                f"gate={'PASS' if temp_ok else 'REJECT'}"
            )
            print(
                f"pressure   : positive={pressure_ev['positive_proxy']}/5 "
                f"nonneg_trend={pressure_ev['nonneg_trend']}/5 "
                f"max_rmse_ratio={pressure_ev['max_rmse_ratio']:.4f} "
                f"pooled_rmse_ratio={pressure_ev['pooled_rmse_ratio']:.4f} "
                f"proxy_gain={100 * pressure_ev['pooled_proxy_gain']:+.2f}% "
                f"gate={'PASS' if pressure_ok else 'REJECT'}"
            )
            print(
                f"global     : flat_rmse_ratio={g['flat_rmse_ratio']:.6f} "
                f"mean_proxy_gain={100 * g['mean_proxy_gain']:+.2f}% "
                f"untouched_exact={exact_untouched}"
            )
            print(f"V9-SAFE COMBO GATE: {'PASS' if final_ok else 'REJECT'}")

            for label, ev in (("temperature", temp_ev), ("pressure", pressure_ev)):
                print(f"  {label} per-sequence:")
                for seq_name, row in zip(names, ev["rows"]):
                    print(
                        f"    {seq_name:14s} "
                        f"proxy={100 * row['proxy_gain']:+6.2f}% "
                        f"trend={row['trend_gain']:+.4f} "
                        f"rmse_ratio={row['rmse_ratio']:.4f}"
                    )

    passing = [c for c in candidates if c["gate_pass"]]
    selected = None
    if passing:
        # Conservative rule: prefer the least analog exposure first.
        passing.sort(
            key=lambda x: (
                x["temperature_alpha"] + x["pressure_alpha"],
                x["temperature_alpha"],
                x["pressure_alpha"],
                -x["mean_proxy_gain"],
            )
        )
        selected = passing[0]

    print("\n" + "=" * 118)
    print("V9-SAFE FINAL SUMMARY")
    print("=" * 118)
    print(f"passing combinations : {len(passing)}/{len(candidates)}")
    if selected is None:
        print("V9-SAFE FINAL GATE    : REJECT")
        print("action                : keep exact V8; do not build runtime")
    else:
        print("V9-SAFE FINAL GATE    : PASS")
        print(
            f"selected              : temperature alpha={selected['temperature_alpha']:.3f}, "
            f"pressure alpha={selected['pressure_alpha']:.3f}"
        )
        print(f"flat RMSE V8          : {selected['flat_rmse_v8']:.6f}")
        print(f"flat RMSE V9-Safe     : {selected['flat_rmse_candidate']:.6f}")
        print(f"flat RMSE ratio       : {selected['flat_rmse_ratio']:.6f}")
        print(f"mean proxy V8         : {selected['mean_proxy_v8']:.6f}")
        print(f"mean proxy V9-Safe    : {selected['mean_proxy_candidate']:.6f}")
        print(f"mean proxy gain       : {100 * selected['mean_proxy_gain']:+.2f}%")
        print("next                  : build isolated V9-Safe runtime on port 8801")

    output = {
        "candidate": "v9_safe_temperature_pressure_analog",
        "base_version": int(cfg.get("version", -1)),
        "base_trajectory_model": cfg.get("trajectory_model"),
        "data_policy": "official history.csv only for analog bank; future.csv evaluation only",
        "temperature_config": TEMP_CFG.key(),
        "pressure_config": PRESSURE_CFG.key(),
        "temperature_alphas": list(TEMP_ALPHAS),
        "pressure_alphas": list(PRESSURE_ALPHAS),
        "untouched_targets": [
            x for x in TARGET_COLUMNS if x not in (TEMP_NAME, PRESSURE_NAME)
        ],
        "selection_rule": "smallest-alpha passing combination",
        "candidates": candidates,
        "selected": selected,
        "offline_gate_pass": selected is not None,
        "important_note": (
            "This is an offline sensitivity/safety diagnostic only. "
            "Production V8 on port 8800 is untouched. "
            "Local official-sequence results do not guarantee hidden-test improvement."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"metrics               : {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
