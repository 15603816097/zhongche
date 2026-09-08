from __future__ import annotations

import json
import math
import pickle
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, TARGET_COLUMNS
from evaluate_patchtst_vs_v8_official import gain, rmse, safe_corr
from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster, PatchTSTConfig
from src.inference import predict_future as predict_future_v8
from src.v8_runtime import v8_enabled


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
V1_CHECKPOINT = ROOT / "models" / "deep" / "patchtst_v1_pretrain.pt"
V82_CHECKPOINT = ROOT / "models" / "deep" / "patchtst_v82_trend.pt"
OUTPUT_PATH = ROOT / "models" / "deep" / "patchtst_v82_temperature_candidate.json"
TEMP_NAME = "temperature_c"
TEMP_IDX = TARGET_COLUMNS.index(TEMP_NAME)
WEIGHT_GRID = np.asarray([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=np.float64)


def config_from_checkpoint(raw: dict) -> PatchTSTConfig:
    cfg = dict(raw.get("config", {}))
    allowed = {f.name for f in fields(PatchTSTConfig)}
    return PatchTSTConfig(**{k: v for k, v in cfg.items() if k in allowed})


def load_patch_predictions(path: Path, X: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    raw = torch.load(path, map_location=device, weights_only=False)
    model = MaskedPatchTSTForecaster(config_from_checkpoint(raw)).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()
    with torch.no_grad():
        out = model(
            torch.from_numpy(X).to(device=device, dtype=torch.float32),
            torch.from_numpy(mask).to(device=device, dtype=torch.float32),
        )
    return out.cpu().numpy().astype(np.float64)


def direction_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    dp = np.diff(np.asarray(pred, dtype=np.float64), axis=-1).reshape(-1)
    dt = np.diff(np.asarray(true, dtype=np.float64), axis=-1).reshape(-1)
    ok = np.isfinite(dp) & np.isfinite(dt)
    if int(ok.sum()) < 3:
        return float("nan")
    dp = dp[ok]
    dt = dt[ok]
    threshold = max(1e-9, 0.10 * float(np.std(dt)))
    meaningful = np.abs(dt) >= threshold
    if int(meaningful.sum()) < 3:
        return float("nan")
    return float(np.mean(np.sign(dp[meaningful]) == np.sign(dt[meaningful])))


def trend_proxy(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    dp = np.diff(np.asarray(pred, dtype=np.float64), axis=-1).reshape(-1)
    dt = np.diff(np.asarray(true, dtype=np.float64), axis=-1).reshape(-1)
    corr = safe_corr(dp, dt)
    dacc = direction_accuracy(pred, true)
    corr01 = 0.5 if not math.isfinite(corr) else 0.5 * (float(np.clip(corr, -1.0, 1.0)) + 1.0)
    dacc01 = 0.5 if not math.isfinite(dacc) else float(np.clip(dacc, 0.0, 1.0))
    proxy = 0.5 * corr01 + 0.5 * dacc01
    return proxy, corr, dacc


def subset_metrics(pred_z: np.ndarray, true_z: np.ndarray, indices: np.ndarray) -> dict:
    p = pred_z[indices, :, TEMP_IDX]
    y = true_z[indices, :, TEMP_IDX]
    r = rmse(p, y)
    proxy, corr, dacc = trend_proxy(p, y)
    endpoint = rmse(p[:, -1], y[:, -1])
    return {
        "rmse_z": r,
        "trend_proxy": proxy,
        "diff_corr": corr,
        "direction_accuracy": dacc,
        "endpoint_rmse_z": endpoint,
    }


def blend(base_z: np.ndarray, patch_z: np.ndarray, weight: float) -> np.ndarray:
    out = np.asarray(base_z, dtype=np.float64).copy()
    out[:, :, TEMP_IDX] = (
        (1.0 - float(weight)) * base_z[:, :, TEMP_IDX]
        + float(weight) * patch_z[:, :, TEMP_IDX]
    )
    return out


def choose_weight(
    base_z: np.ndarray,
    patch_z: np.ndarray,
    true_z: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[float, dict, list[dict]]:
    base_metrics = subset_metrics(base_z, true_z, train_idx)
    rows: list[dict] = []
    best_weight = 0.0
    best_score = 0.0
    best_metrics = dict(base_metrics)

    for w in WEIGHT_GRID:
        candidate = blend(base_z, patch_z, float(w))
        m = subset_metrics(candidate, true_z, train_idx)
        accuracy_gain = gain(base_metrics["rmse_z"], m["rmse_z"])
        trend_delta_pp = 100.0 * (m["trend_proxy"] - base_metrics["trend_proxy"])
        # Official scoring gives accuracy substantially more weight than trend.
        score = accuracy_gain + 0.40 * trend_delta_pp
        safe = bool(
            m["rmse_z"] <= base_metrics["rmse_z"] * 1.005
            and m["trend_proxy"] >= base_metrics["trend_proxy"] - 0.005
        )
        row = {
            "weight": float(w),
            **m,
            "accuracy_gain_vs_v8_pct": accuracy_gain,
            "trend_proxy_delta_pp": trend_delta_pp,
            "selection_score": score,
            "safe": safe,
        }
        rows.append(row)
        if safe and score > best_score + 1e-9:
            best_score = score
            best_weight = float(w)
            best_metrics = dict(m)

    return best_weight, best_metrics, rows


def evaluate_model_loso(
    label: str,
    base_z: np.ndarray,
    patch_z: np.ndarray,
    true_z: np.ndarray,
    true_phys: np.ndarray,
    center3: np.ndarray,
    scale3: np.ndarray,
    group_id: np.ndarray,
) -> dict:
    fold_weights: list[float] = []
    fold_rows: dict[str, dict] = {}
    loso = np.asarray(base_z, dtype=np.float64).copy()

    print("\n" + "=" * 112)
    print(f"{label} LOSO")
    print("=" * 112)

    for holdout in range(len(group_id)):
        train_idx = np.asarray([i for i in range(len(group_id)) if i != holdout], dtype=np.int64)
        w, train_metrics, sweep = choose_weight(base_z, patch_z, true_z, train_idx)
        fold_weights.append(w)
        loso[holdout, :, TEMP_IDX] = (
            (1.0 - w) * base_z[holdout, :, TEMP_IDX]
            + w * patch_z[holdout, :, TEMP_IDX]
        )

        idx = np.asarray([holdout], dtype=np.int64)
        base_m = subset_metrics(base_z, true_z, idx)
        cand_m = subset_metrics(loso, true_z, idx)
        acc_gain = gain(base_m["rmse_z"], cand_m["rmse_z"])
        trend_delta_pp = 100.0 * (cand_m["trend_proxy"] - base_m["trend_proxy"])
        fold_rows[str(group_id[holdout])] = {
            "weight": w,
            "train_metrics": train_metrics,
            "base": base_m,
            "candidate": cand_m,
            "accuracy_gain_vs_v8_pct": acc_gain,
            "trend_proxy_delta_pp": trend_delta_pp,
            "train_sweep": sweep,
        }
        print(
            f"{group_id[holdout]:14s}: w={w:.2f} | "
            f"acc_gain={acc_gain:+.2f}% trend_delta={trend_delta_pp:+.2f}pp | "
            f"corr {base_m['diff_corr']:+.4f}->{cand_m['diff_corr']:+.4f} "
            f"dir {base_m['direction_accuracy']:.4f}->{cand_m['direction_accuracy']:.4f}"
        )

    all_idx = np.arange(len(group_id), dtype=np.int64)
    base_all = subset_metrics(base_z, true_z, all_idx)
    loso_all = subset_metrics(loso, true_z, all_idx)
    loso_phys = loso * scale3 + center3
    base_phys = base_z * scale3 + center3

    physical_base_rmse = rmse(base_phys[:, :, TEMP_IDX], true_phys[:, :, TEMP_IDX])
    physical_loso_rmse = rmse(loso_phys[:, :, TEMP_IDX], true_phys[:, :, TEMP_IDX])
    physical_gain = gain(physical_base_rmse, physical_loso_rmse)

    deploy_weight = float(np.median(np.asarray(fold_weights, dtype=np.float64)))
    fixed = blend(base_z, patch_z, deploy_weight)
    fixed_all = subset_metrics(fixed, true_z, all_idx)
    fixed_phys = fixed * scale3 + center3
    fixed_physical_rmse = rmse(fixed_phys[:, :, TEMP_IDX], true_phys[:, :, TEMP_IDX])

    fixed_fold_gains: list[float] = []
    fixed_fold_trend_deltas: list[float] = []
    for i in range(len(group_id)):
        idx = np.asarray([i], dtype=np.int64)
        b = subset_metrics(base_z, true_z, idx)
        c = subset_metrics(fixed, true_z, idx)
        fixed_fold_gains.append(gain(b["rmse_z"], c["rmse_z"]))
        fixed_fold_trend_deltas.append(100.0 * (c["trend_proxy"] - b["trend_proxy"]))

    result = {
        "label": label,
        "fold_weights": fold_weights,
        "folds": fold_rows,
        "loso": {
            "base": base_all,
            "candidate": loso_all,
            "accuracy_gain_vs_v8_pct": gain(base_all["rmse_z"], loso_all["rmse_z"]),
            "trend_proxy_delta_pp": 100.0 * (loso_all["trend_proxy"] - base_all["trend_proxy"]),
            "physical_rmse_v8": physical_base_rmse,
            "physical_rmse_candidate": physical_loso_rmse,
            "physical_gain_vs_v8_pct": physical_gain,
        },
        "deploy_weight": deploy_weight,
        "fixed": {
            "base": base_all,
            "candidate": fixed_all,
            "accuracy_gain_vs_v8_pct": gain(base_all["rmse_z"], fixed_all["rmse_z"]),
            "trend_proxy_delta_pp": 100.0 * (fixed_all["trend_proxy"] - base_all["trend_proxy"]),
            "physical_rmse_v8": physical_base_rmse,
            "physical_rmse_candidate": fixed_physical_rmse,
            "physical_gain_vs_v8_pct": gain(physical_base_rmse, fixed_physical_rmse),
            "fold_accuracy_gains_pct": fixed_fold_gains,
            "fold_trend_proxy_deltas_pp": fixed_fold_trend_deltas,
            "positive_accuracy_folds": int(np.sum(np.asarray(fixed_fold_gains) > 0.0)),
            "nonnegative_trend_folds": int(np.sum(np.asarray(fixed_fold_trend_deltas) >= -1e-9)),
            "min_accuracy_gain_pct": float(np.min(fixed_fold_gains)),
            "min_trend_proxy_delta_pp": float(np.min(fixed_fold_trend_deltas)),
        },
    }
    return result


def main() -> int:
    required = [
        CORPUS_PATH,
        V1_CHECKPOINT,
        V82_CHECKPOINT,
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
        print("Missing required files:")
        for p in missing:
            print("  ", p)
        raise SystemExit(2)

    with open(MODEL_DIR / "ensemble_config.pkl", "rb") as f:
        ensemble_config = pickle.load(f)
    if not v8_enabled(ensemble_config):
        raise RuntimeError("models/ensemble_config.pkl is not V8-compatible")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(CORPUS_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32, copy=False)
    Y = data["Y"].astype(np.float64, copy=False)
    mask = data["mask"].astype(np.float32, copy=False)
    center = data["center"].astype(np.float64, copy=False)
    scale = data["scale"].astype(np.float64, copy=False)
    group_id = data["group_id"].astype(str)
    targets = data["targets"].astype(str).tolist()
    if targets != list(TARGET_COLUMNS):
        raise RuntimeError(f"target mismatch: {targets}")

    v1_z = load_patch_predictions(V1_CHECKPOINT, X, mask, device)
    v82_z = load_patch_predictions(V82_CHECKPOINT, X, mask, device)

    center3 = center[:, None, :]
    scale3 = scale[:, None, :]
    history_phys = X.astype(np.float64) * scale3 + center3
    true_phys = Y * scale3 + center3

    print("=" * 112)
    print("V8.2 PATCHTST TEMPERATURE CANDIDATE - OFFICIAL FIVE-SEQUENCE LOSO")
    print("=" * 112)
    print(f"device             : {device}")
    if device.type == "cuda":
        print(f"gpu                : {torch.cuda.get_device_name(device)}")
    print(f"V8 config version  : {ensemble_config.get('version')}")
    print(f"weight grid        : {WEIGHT_GRID.tolist()}")
    print("comparison         : current V8 vs old V8.1 PatchTST vs new V8.2 trend PatchTST")
    print("IMPORTANT          : offline only; app.py/callback/ensemble_config.pkl are untouched")

    v8_phys = np.empty_like(true_phys)
    for i, gid in enumerate(group_id):
        print(f"V8 inference [{i + 1}/{len(group_id)}] {gid} ...", flush=True)
        history_df = pd.DataFrame(history_phys[i], columns=TARGET_COLUMNS)
        pred = np.asarray(predict_future_v8(history_df), dtype=np.float64)
        if pred.shape != true_phys[i].shape:
            raise RuntimeError(f"unexpected V8 shape {pred.shape} for {gid}")
        v8_phys[i] = pred

    v8_z = (v8_phys - center3) / scale3

    v81_result = evaluate_model_loso(
        "V8.1 old PatchTST",
        v8_z,
        v1_z,
        Y,
        true_phys,
        center3,
        scale3,
        group_id,
    )
    v82_result = evaluate_model_loso(
        "V8.2 trend PatchTST",
        v8_z,
        v82_z,
        Y,
        true_phys,
        center3,
        scale3,
        group_id,
    )

    f = v82_result["fixed"]
    old = v81_result["fixed"]
    gate = bool(
        f["positive_accuracy_folds"] == len(group_id)
        and f["nonnegative_trend_folds"] >= len(group_id) - 1
        and f["physical_gain_vs_v8_pct"] >= 10.0
        and f["trend_proxy_delta_pp"] >= 0.0
        and f["min_accuracy_gain_pct"] > 0.0
        and f["accuracy_gain_vs_v8_pct"] >= old["accuracy_gain_vs_v8_pct"] - 1.0
        and f["trend_proxy_delta_pp"] >= old["trend_proxy_delta_pp"]
    )

    print("\n" + "=" * 112)
    print("V8.2 DECISION")
    print("=" * 112)
    print(
        f"V8.1 fixed w={v81_result['deploy_weight']:.2f}: "
        f"acc_gain={old['accuracy_gain_vs_v8_pct']:+.2f}% "
        f"physical_gain={old['physical_gain_vs_v8_pct']:+.2f}% "
        f"trend_delta={old['trend_proxy_delta_pp']:+.2f}pp"
    )
    print(
        f"V8.2 fixed w={v82_result['deploy_weight']:.2f}: "
        f"acc_gain={f['accuracy_gain_vs_v8_pct']:+.2f}% "
        f"physical_gain={f['physical_gain_vs_v8_pct']:+.2f}% "
        f"trend_delta={f['trend_proxy_delta_pp']:+.2f}pp "
        f"positive_acc={f['positive_accuracy_folds']}/{len(group_id)} "
        f"nonneg_trend={f['nonnegative_trend_folds']}/{len(group_id)}"
    )
    print(f"V8.2 OFFLINE GATE   : {'PASS' if gate else 'REJECT'}")

    result = {
        "model": "v82_patchtst_temperature_trend_candidate",
        "weight_grid": WEIGHT_GRID.tolist(),
        "v81_old_patchtst": v81_result,
        "v82_trend_patchtst": v82_result,
        "offline_gate_pass": gate,
        "gate_rule": (
            "fixed V8.2 candidate: accuracy improves all five folds; trend proxy nonnegative on >=4/5; "
            "pooled physical temperature RMSE gain >=10%; pooled trend proxy nonnegative; "
            "accuracy no worse than old PatchTST by >1pp and trend proxy >= old PatchTST"
        ),
        "important_note": (
            "The real V8.1 online score was not improved, so this gate is intentionally strict. "
            "Do not activate V8.2 online unless this script reports PASS."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics             : {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
