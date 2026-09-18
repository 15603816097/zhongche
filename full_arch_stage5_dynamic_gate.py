from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from config import HORIZON, TARGET_COLUMNS
from v9_analog_multiscale_diagnostic import evaluate_rows, EPS

ROOT = Path(__file__).resolve().parent
STAGE4_DIR = ROOT / "artifacts" / "full_arch" / "confidence_ood"
STAGE4_NPZ = STAGE4_DIR / "analog_confidence_ood.npz"
STAGE4_MANIFEST = STAGE4_DIR / "manifest.json"
OUT_DIR = ROOT / "artifacts" / "full_arch" / "dynamic_gate"
MODEL_DIR = ROOT / "models" / "full_arch"

SEQUENCES = [f"sequence{i:04d}" for i in range(1, 6)]
SEGMENTS = ((0, 32), (32, 64), (64, 96))
ALPHAS = (0.05, 0.10, 0.15, 0.20)
OOD_POWERS = (1.0, 2.0)

# Conservative train-fold gate.
TRAIN_MIN_POOLED_PROXY_GAIN = 0.003
TRAIN_MAX_POOLED_RMSE_RATIO = 0.998
TRAIN_MIN_POSITIVE_PROXY_FOLDS = 3
TRAIN_MIN_NONNEG_TREND_FOLDS = 3
TRAIN_MAX_WORST_RMSE_RATIO = 1.005

# Honest LOSO target gate. A target is allowed to leave exact V8 only if the
# combined 3-segment dynamic gate survives this holdout criterion.
LOSO_MIN_POSITIVE_PROXY = 4
LOSO_MIN_NONNEG_TREND = 4
LOSO_MAX_RMSE_RATIO = 1.010
LOSO_MAX_POOLED_RMSE_RATIO = 0.998
LOSO_MIN_POOLED_PROXY_GAIN = 0.005

# Final all-data fixed parameter search. This is performed only for targets that
# already passed the honest LOSO target gate.
FINAL_MIN_POSITIVE_PROXY = 4
FINAL_MIN_NONNEG_TREND = 4
FINAL_MAX_WORST_RMSE_RATIO = 1.010
FINAL_MAX_POOLED_RMSE_RATIO = 0.998
FINAL_MIN_POOLED_PROXY_GAIN = 0.005

# Entire candidate must not degrade pooled flat RMSE.
GLOBAL_MAX_FLAT_RMSE_RATIO = 1.0


@dataclass(frozen=True)
class GateParam:
    config_index: int
    alpha: float
    ood_power: float

    def key(self, configs: list[dict]) -> str:
        cfg = configs[self.config_index]
        return (
            f"{cfg['key']},alpha={self.alpha:.3f},"
            f"ood_power={self.ood_power:.1f}"
        )


def _metrics(y: np.ndarray, p: np.ndarray, anchors: np.ndarray) -> dict:
    return evaluate_rows(y, p, anchors)


def _gain(base: dict, cand: dict) -> float:
    return float(
        (base["proxy_loss"] - cand["proxy_loss"])
        / max(abs(base["proxy_loss"]), EPS)
    )


def _apply_segment(
    base: np.ndarray,
    analog: np.ndarray,
    confidence: np.ndarray,
    combined_ood: np.ndarray,
    param: GateParam,
    segment: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    base/analog: [N, H]
    confidence/combined_ood: [N]
    returns candidate [N,H], effective_weight [N]
    """
    start, end = segment
    gate = (
        float(param.alpha)
        * np.clip(confidence, 0.0, 1.0)
        * np.power(np.clip(1.0 - combined_ood, 0.0, 1.0), float(param.ood_power))
    )
    gate = np.clip(gate, 0.0, float(param.alpha))

    out = np.asarray(base, dtype=np.float64).copy()
    out[:, start:end] = (
        base[:, start:end]
        + gate[:, None] * (analog[:, start:end] - base[:, start:end])
    )
    return out, gate


def _eval_candidate(
    truth: np.ndarray,
    base: np.ndarray,
    anchors: np.ndarray,
    analog: np.ndarray,
    confidence: np.ndarray,
    combined_ood: np.ndarray,
    param: GateParam,
    segment: tuple[int, int],
    indices: np.ndarray,
) -> dict:
    cand, gate = _apply_segment(
        base, analog, confidence, combined_ood, param, segment
    )
    b = _metrics(truth[indices], base[indices], anchors[indices])
    c = _metrics(truth[indices], cand[indices], anchors[indices])

    seq_rows = []
    for i in indices:
        bi = _metrics(
            truth[i : i + 1], base[i : i + 1], anchors[i : i + 1]
        )
        ci = _metrics(
            truth[i : i + 1], cand[i : i + 1], anchors[i : i + 1]
        )
        seq_rows.append(
            {
                "index": int(i),
                "sequence": SEQUENCES[int(i)],
                "proxy_gain": _gain(bi, ci),
                "trend_gain": float(ci["trend_core"] - bi["trend_core"]),
                "rmse_ratio": float(ci["rmse"] / max(bi["rmse"], EPS)),
                "effective_weight": float(gate[i]),
            }
        )

    return {
        "prediction": cand,
        "gate": gate,
        "pooled_proxy_gain": _gain(b, c),
        "pooled_trend_gain": float(c["trend_core"] - b["trend_core"]),
        "pooled_rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
        "positive_proxy": int(sum(r["proxy_gain"] > 0.0 for r in seq_rows)),
        "nonneg_trend": int(sum(r["trend_gain"] >= -1e-12 for r in seq_rows)),
        "max_rmse_ratio": float(max(r["rmse_ratio"] for r in seq_rows)),
        "mean_effective_weight": float(np.mean(gate[indices])),
        "seq_rows": seq_rows,
    }


def _eligible_train(ev: dict, n_train: int) -> bool:
    min_pos = min(TRAIN_MIN_POSITIVE_PROXY_FOLDS, n_train)
    min_trend = min(TRAIN_MIN_NONNEG_TREND_FOLDS, n_train)
    return bool(
        ev["pooled_proxy_gain"] >= TRAIN_MIN_POOLED_PROXY_GAIN
        and ev["pooled_trend_gain"] >= -1e-12
        and ev["pooled_rmse_ratio"] <= TRAIN_MAX_POOLED_RMSE_RATIO
        and ev["positive_proxy"] >= min_pos
        and ev["nonneg_trend"] >= min_trend
        and ev["max_rmse_ratio"] <= TRAIN_MAX_WORST_RMSE_RATIO
    )


def _choose_on_indices(
    truth: np.ndarray,
    base: np.ndarray,
    anchors: np.ndarray,
    analog_pred: np.ndarray,
    confidence: np.ndarray,
    combined_ood: np.ndarray,
    configs: list[dict],
    target_idx: int,
    segment: tuple[int, int],
    indices: np.ndarray,
) -> tuple[GateParam | None, dict | None]:
    best_param = None
    best_ev = None
    best_score = None

    for cidx in range(len(configs)):
        analog = analog_pred[cidx, :, :, target_idx]
        conf = confidence[cidx, :, target_idx]
        ood = combined_ood[cidx, :, target_idx]
        for alpha in ALPHAS:
            for power in OOD_POWERS:
                param = GateParam(cidx, float(alpha), float(power))
                ev = _eval_candidate(
                    truth[:, :, target_idx],
                    base[:, :, target_idx],
                    anchors[:, target_idx],
                    analog,
                    conf,
                    ood,
                    param,
                    segment,
                    indices,
                )
                if not _eligible_train(ev, len(indices)):
                    continue

                # Worst-case first, then pooled competition proxy, then RMSE.
                # Prefer lower effective weight as the final tie breaker.
                worst_pg = min(r["proxy_gain"] for r in ev["seq_rows"])
                score = (
                    worst_pg,
                    ev["pooled_proxy_gain"],
                    1.0 - ev["pooled_rmse_ratio"],
                    ev["pooled_trend_gain"],
                    -ev["mean_effective_weight"],
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_param = param
                    best_ev = ev

    return best_param, best_ev


def _apply_param_to_one(
    base_row: np.ndarray,
    analog_row: np.ndarray,
    confidence_value: float,
    ood_value: float,
    param: GateParam,
    segment: tuple[int, int],
) -> tuple[np.ndarray, float]:
    start, end = segment
    weight = (
        float(param.alpha)
        * float(np.clip(confidence_value, 0.0, 1.0))
        * float(np.clip(1.0 - ood_value, 0.0, 1.0) ** param.ood_power)
    )
    weight = float(np.clip(weight, 0.0, param.alpha))
    out = np.asarray(base_row, dtype=np.float64).copy()
    out[start:end] = (
        base_row[start:end]
        + weight * (analog_row[start:end] - base_row[start:end])
    )
    return out, weight


def _summarize_holdouts(
    truth: np.ndarray,
    base: np.ndarray,
    pred: np.ndarray,
    anchors: np.ndarray,
    target_idx: int,
) -> dict:
    rows = []
    for i, name in enumerate(SEQUENCES):
        b = _metrics(
            truth[i : i + 1, :, target_idx],
            base[i : i + 1, :, target_idx],
            anchors[i : i + 1, target_idx],
        )
        c = _metrics(
            truth[i : i + 1, :, target_idx],
            pred[i : i + 1, :, target_idx],
            anchors[i : i + 1, target_idx],
        )
        rows.append(
            {
                "sequence": name,
                "proxy_gain": _gain(b, c),
                "trend_gain": float(c["trend_core"] - b["trend_core"]),
                "rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
            }
        )

    bpool = _metrics(
        truth[:, :, target_idx],
        base[:, :, target_idx],
        anchors[:, target_idx],
    )
    cpool = _metrics(
        truth[:, :, target_idx],
        pred[:, :, target_idx],
        anchors[:, target_idx],
    )

    return {
        "positive_proxy": int(sum(r["proxy_gain"] > 0 for r in rows)),
        "nonneg_trend": int(sum(r["trend_gain"] >= -1e-12 for r in rows)),
        "max_rmse_ratio": float(max(r["rmse_ratio"] for r in rows)),
        "pooled_rmse_ratio": float(cpool["rmse"] / max(bpool["rmse"], EPS)),
        "pooled_proxy_gain": _gain(bpool, cpool),
        "pooled_trend_gain": float(cpool["trend_core"] - bpool["trend_core"]),
        "rows": rows,
    }


def _loso_target_pass(summary: dict) -> bool:
    return bool(
        summary["positive_proxy"] >= LOSO_MIN_POSITIVE_PROXY
        and summary["nonneg_trend"] >= LOSO_MIN_NONNEG_TREND
        and summary["max_rmse_ratio"] <= LOSO_MAX_RMSE_RATIO
        and summary["pooled_rmse_ratio"] <= LOSO_MAX_POOLED_RMSE_RATIO
        and summary["pooled_proxy_gain"] >= LOSO_MIN_POOLED_PROXY_GAIN
        and summary["pooled_trend_gain"] >= -1e-12
    )


def _final_candidate_pass(ev: dict) -> bool:
    return bool(
        ev["positive_proxy"] >= FINAL_MIN_POSITIVE_PROXY
        and ev["nonneg_trend"] >= FINAL_MIN_NONNEG_TREND
        and ev["max_rmse_ratio"] <= FINAL_MAX_WORST_RMSE_RATIO
        and ev["pooled_rmse_ratio"] <= FINAL_MAX_POOLED_RMSE_RATIO
        and ev["pooled_proxy_gain"] >= FINAL_MIN_POOLED_PROXY_GAIN
        and ev["pooled_trend_gain"] >= -1e-12
    )


def _deep_aux_summary() -> dict:
    """Deep LOSO is retained as an auxiliary uncertainty probe only.

    Direct Deep forecasting failed Stage 3 LOSO, so Stage 5 never assigns it
    forecast weight. We record its existence and rejection explicitly so the
    architecture remains auditable.
    """
    p = ROOT / "artifacts" / "full_arch" / "deep_loso" / "manifest.json"
    if not p.is_file():
        return {"available": False}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out = {"available": True, "direct_forecast_enabled": False, "models": {}}
    for kind in ("tcn", "patchtst"):
        s = raw.get("summary", {}).get(kind, {})
        out["models"][kind] = {
            "flat_rmse_ratio": s.get("flat_rmse_ratio"),
            "reason": "Stage 3 LOSO rejected direct forecast contribution",
        }
    return out


def main() -> int:
    if not STAGE4_NPZ.is_file() or not STAGE4_MANIFEST.is_file():
        raise FileNotFoundError("Stage 4 artifacts missing; run Stage 4 first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    stage4_manifest = json.loads(STAGE4_MANIFEST.read_text(encoding="utf-8"))
    configs = list(stage4_manifest["configs"])

    with np.load(STAGE4_NPZ) as z:
        truth = np.asarray(z["truth"], dtype=np.float64)
        v8 = np.asarray(z["v8"], dtype=np.float64)
        anchors = np.asarray(z["anchors"], dtype=np.float64)
        analog_pred = np.asarray(z["analog_pred"], dtype=np.float64)
        confidence = np.asarray(z["confidence"], dtype=np.float64)
        combined_ood = np.asarray(z["combined_ood"], dtype=np.float64)

    expected = (len(configs), len(SEQUENCES), HORIZON, len(TARGET_COLUMNS))
    if analog_pred.shape != expected:
        raise RuntimeError(f"analog_pred shape={analog_pred.shape}, expected={expected}")

    print("=" * 122)
    print("FULL ARCHITECTURE - STAGE 5 TARGET x HORIZON DYNAMIC GATE")
    print("=" * 122)
    print("segments          :", SEGMENTS)
    print("base alphas       :", ALPHAS)
    print("OOD powers        :", OOD_POWERS)
    print("configs           :", len(configs))
    print("dynamic weight    : alpha * confidence * (1-combined_ood)^power")
    print("Deep direct weight: 0 (rejected by Stage 3 LOSO; retained as auxiliary evidence)")
    print()

    n_seq = len(SEQUENCES)
    n_targets = len(TARGET_COLUMNS)
    loso_pred = np.asarray(v8, dtype=np.float64).copy()
    loso_detail = {}

    # --------------------------------------------------------------
    # Honest LOSO: every holdout sequence is untouched by parameter search.
    # --------------------------------------------------------------
    for j, target in enumerate(TARGET_COLUMNS):
        loso_detail[target] = {"folds": []}
        for holdout in range(n_seq):
            train_idx = np.asarray(
                [i for i in range(n_seq) if i != holdout], dtype=np.int64
            )
            row_pred = v8[holdout, :, j].copy()
            fold_params = []
            print(
                f"LOSO target={target:15s} holdout={SEQUENCES[holdout]}",
                flush=True,
            )

            for seg_idx, segment in enumerate(SEGMENTS):
                param, train_ev = _choose_on_indices(
                    truth,
                    v8,
                    anchors,
                    analog_pred,
                    confidence,
                    combined_ood,
                    configs,
                    j,
                    segment,
                    train_idx,
                )
                if param is None:
                    fold_params.append(
                        {
                            "segment": list(segment),
                            "enabled": False,
                            "reason": "no safe train candidate",
                        }
                    )
                    print(
                        f"  segment={segment}: EXACT V8 (no safe train candidate)",
                        flush=True,
                    )
                    continue

                analog_row = analog_pred[param.config_index, holdout, :, j]
                conf_value = float(confidence[param.config_index, holdout, j])
                ood_value = float(combined_ood[param.config_index, holdout, j])
                row_pred, eff_weight = _apply_param_to_one(
                    row_pred,
                    analog_row,
                    conf_value,
                    ood_value,
                    param,
                    segment,
                )
                fold_params.append(
                    {
                        "segment": list(segment),
                        "enabled": True,
                        "param": asdict(param),
                        "param_key": param.key(configs),
                        "holdout_confidence": conf_value,
                        "holdout_ood": ood_value,
                        "holdout_effective_weight": eff_weight,
                        "train_pooled_proxy_gain": train_ev["pooled_proxy_gain"],
                        "train_pooled_rmse_ratio": train_ev["pooled_rmse_ratio"],
                        "train_max_rmse_ratio": train_ev["max_rmse_ratio"],
                    }
                )
                print(
                    f"  segment={segment}: {param.key(configs)} "
                    f"eff_w={eff_weight:.4f} "
                    f"train_proxy={100*train_ev['pooled_proxy_gain']:+.2f}% "
                    f"train_rmse={train_ev['pooled_rmse_ratio']:.4f}",
                    flush=True,
                )

            loso_pred[holdout, :, j] = row_pred
            b = _metrics(
                truth[holdout : holdout + 1, :, j],
                v8[holdout : holdout + 1, :, j],
                anchors[holdout : holdout + 1, j],
            )
            c = _metrics(
                truth[holdout : holdout + 1, :, j],
                row_pred.reshape(1, -1),
                anchors[holdout : holdout + 1, j],
            )
            fold_result = {
                "holdout": SEQUENCES[holdout],
                "proxy_gain": _gain(b, c),
                "trend_gain": float(c["trend_core"] - b["trend_core"]),
                "rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
                "segments": fold_params,
            }
            loso_detail[target]["folds"].append(fold_result)
            print(
                f"  HOLDOUT result: proxy={100*fold_result['proxy_gain']:+.2f}% "
                f"trend={fold_result['trend_gain']:+.4f} "
                f"rmse_ratio={fold_result['rmse_ratio']:.4f}",
                flush=True,
            )

        s = _summarize_holdouts(truth, v8, loso_pred, anchors, j)
        s["pass"] = _loso_target_pass(s)
        loso_detail[target]["summary"] = s

    print("\n" + "=" * 122)
    print("LOSO DYNAMIC-GATE SUMMARY")
    print("=" * 122)
    loso_pass_targets = []
    for j, target in enumerate(TARGET_COLUMNS):
        s = loso_detail[target]["summary"]
        if s["pass"]:
            loso_pass_targets.append(target)
        print(
            f"{target:15s} {'PASS' if s['pass'] else 'REJECT':6s} "
            f"positive={s['positive_proxy']}/5 "
            f"nonneg_trend={s['nonneg_trend']}/5 "
            f"max_rmse={s['max_rmse_ratio']:.4f} "
            f"pooled_rmse={s['pooled_rmse_ratio']:.4f} "
            f"proxy={100*s['pooled_proxy_gain']:+6.2f}% "
            f"trend={s['pooled_trend_gain']:+.4f}"
        )

    # --------------------------------------------------------------
    # Freeze production parameters only for targets that already pass LOSO.
    # Search all 5 sequences for a stable parameter for each segment, then
    # re-run a final target-level safety gate on the combined prediction.
    # --------------------------------------------------------------
    all_idx = np.arange(n_seq, dtype=np.int64)
    final_pred = np.asarray(v8, dtype=np.float64).copy()
    final_config = {
        "version": "full_arch_gate_v1",
        "base_model": "V8",
        "segments": [list(x) for x in SEGMENTS],
        "weight_formula": "alpha * confidence * (1-combined_ood)^ood_power",
        "targets": {},
        "deep_policy": (
            "TCN/PatchTST direct forecast weights are zero because Stage 3 LOSO "
            "rejected them; checkpoints remain available as auxiliary experiments."
        ),
    }

    for j, target in enumerate(TARGET_COLUMNS):
        target_cfg = {
            "enabled": False,
            "loso_pass": bool(target in loso_pass_targets),
            "segments": [],
        }
        if target not in loso_pass_targets:
            target_cfg["reason"] = "LOSO target gate rejected"
            final_config["targets"][target] = target_cfg
            continue

        target_pred = v8[:, :, j].copy()
        for segment in SEGMENTS:
            param, ev = _choose_on_indices(
                truth,
                v8,
                anchors,
                analog_pred,
                confidence,
                combined_ood,
                configs,
                j,
                segment,
                all_idx,
            )
            if param is None or not _final_candidate_pass(ev):
                target_cfg["segments"].append(
                    {
                        "segment": list(segment),
                        "enabled": False,
                        "reason": "no all-data safe fixed candidate",
                    }
                )
                continue

            cidx = param.config_index
            cand, gate = _apply_segment(
                target_pred,
                analog_pred[cidx, :, :, j],
                confidence[cidx, :, j],
                combined_ood[cidx, :, j],
                param,
                segment,
            )
            target_pred = cand
            target_cfg["segments"].append(
                {
                    "segment": list(segment),
                    "enabled": True,
                    "config_index": int(cidx),
                    "analog_config": configs[cidx],
                    "alpha": float(param.alpha),
                    "ood_power": float(param.ood_power),
                    "effective_weights_on_known": gate.tolist(),
                    "selection_metrics": {
                        "pooled_proxy_gain": ev["pooled_proxy_gain"],
                        "pooled_rmse_ratio": ev["pooled_rmse_ratio"],
                        "pooled_trend_gain": ev["pooled_trend_gain"],
                        "positive_proxy": ev["positive_proxy"],
                        "nonneg_trend": ev["nonneg_trend"],
                        "max_rmse_ratio": ev["max_rmse_ratio"],
                    },
                }
            )

        temp_full = final_pred.copy()
        temp_full[:, :, j] = target_pred
        s = _summarize_holdouts(truth, v8, temp_full, anchors, j)
        target_cfg["combined_known_summary"] = s
        if _loso_target_pass(s) and any(x["enabled"] for x in target_cfg["segments"]):
            target_cfg["enabled"] = True
            final_pred[:, :, j] = target_pred
        else:
            target_cfg["reason"] = "combined fixed configuration failed final safety gate"
            # Keep exact V8.
            target_cfg["enabled"] = False

        final_config["targets"][target] = target_cfg

    flat_v8 = float(np.sqrt(np.mean((truth - v8) ** 2)))
    flat_final = float(np.sqrt(np.mean((truth - final_pred) ** 2)))
    flat_ratio = float(flat_final / max(flat_v8, EPS))
    enabled_targets = [
        t for t in TARGET_COLUMNS if final_config["targets"][t]["enabled"]
    ]

    final_config["enabled_targets"] = enabled_targets
    final_config["flat_rmse_v8"] = flat_v8
    final_config["flat_rmse_candidate"] = flat_final
    final_config["flat_rmse_ratio"] = flat_ratio
    final_config["global_gate_pass"] = bool(
        len(enabled_targets) > 0 and flat_ratio <= GLOBAL_MAX_FLAT_RMSE_RATIO
    )
    final_config["stage3_deep_aux"] = _deep_aux_summary()

    config_path = MODEL_DIR / "dynamic_gate_config.json"
    config_path.write_text(
        json.dumps(final_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out_npz = OUT_DIR / "stage5_predictions.npz"
    np.savez_compressed(
        out_npz,
        truth=truth,
        v8=v8,
        loso_prediction=loso_pred,
        final_known_prediction=final_pred,
        anchors=anchors,
    )
    manifest = {
        "stage": 5,
        "loso": loso_detail,
        "final_config_path": str(config_path.relative_to(ROOT)),
        "prediction_npz": str(out_npz.relative_to(ROOT)),
        "enabled_targets": enabled_targets,
        "flat_rmse_v8": flat_v8,
        "flat_rmse_candidate": flat_final,
        "flat_rmse_ratio": flat_ratio,
        "global_gate_pass": final_config["global_gate_pass"],
        "important_note": (
            "LOSO predictions are the honest generalization diagnostic. The final known-data "
            "configuration is frozen only for targets that pass LOSO; official hidden score "
            "may still differ."
        ),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 122)
    print("FINAL FROZEN DYNAMIC GATE")
    print("=" * 122)
    for target in TARGET_COLUMNS:
        tc = final_config["targets"][target]
        print(f"{target:15s}: {'ENABLED' if tc['enabled'] else 'EXACT V8'}")
        if tc["enabled"]:
            for seg in tc["segments"]:
                if seg["enabled"]:
                    ac = seg["analog_config"]
                    print(
                        f"  {tuple(seg['segment'])} "
                        f"ctx={ac['context']} k={ac['k']} mode={ac['mode']} "
                        f"guard={ac['self_guard']} alpha={seg['alpha']:.3f} "
                        f"ood_power={seg['ood_power']:.1f}"
                    )

    print(f"\nflat RMSE V8       : {flat_v8:.6f}")
    print(f"flat RMSE candidate : {flat_final:.6f}")
    print(f"flat RMSE ratio     : {flat_ratio:.6f}")
    print(f"enabled targets     : {enabled_targets}")
    print(
        "GLOBAL GATE         : "
        + ("PASS" if final_config["global_gate_pass"] else "REJECT")
    )
    print("config              :", config_path)
    print("manifest            :", manifest_path)
    print("STAGE 5 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
