from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, TARGET_COLUMNS
from evaluate_patchtst_vs_v8_official import (
    CORPUS_PATH,
    PATCH_CHECKPOINT,
    config_from_checkpoint,
    gain,
    rmse,
    safe_corr,
)
from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster
from src.inference import predict_future
from src.v8_runtime import v8_enabled


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "models" / "deep" / "patchtst_temperature_candidate.json"
TEMP_NAME = "temperature_c"
TEMP_IDX = TARGET_COLUMNS.index(TEMP_NAME)
WEIGHT_GRID = np.asarray([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=np.float64)


def _choose_train_weight(
    v8_z: np.ndarray,
    patch_z: np.ndarray,
    true_z: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[float, float, float]:
    base = rmse(v8_z[train_idx, :, TEMP_IDX], true_z[train_idx, :, TEMP_IDX])
    best_w = 0.0
    best_r = base
    for w in WEIGHT_GRID:
        pred = (1.0 - w) * v8_z[train_idx, :, TEMP_IDX] + w * patch_z[train_idx, :, TEMP_IDX]
        r = rmse(pred, true_z[train_idx, :, TEMP_IDX])
        if r < best_r - 1e-10:
            best_w = float(w)
            best_r = r
    return best_w, best_r, gain(base, best_r)


def main() -> int:
    required = [
        CORPUS_PATH,
        PATCH_CHECKPOINT,
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

    checkpoint = torch.load(PATCH_CHECKPOINT, map_location=device, weights_only=False)
    model = MaskedPatchTSTForecaster(config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        patch_z = model(
            torch.from_numpy(X).to(device=device, dtype=torch.float32),
            torch.from_numpy(mask).to(device=device, dtype=torch.float32),
        ).cpu().numpy().astype(np.float64)

    center3 = center[:, None, :]
    scale3 = scale[:, None, :]
    history_phys = X.astype(np.float64) * scale3 + center3
    true_phys = Y * scale3 + center3

    v8_phys = np.empty_like(true_phys)
    print("=" * 104)
    print("PATCHTST TEMPERATURE-ONLY CANDIDATE VS CURRENT V8")
    print("=" * 104)
    print(f"device             : {device}")
    if device.type == "cuda":
        print(f"gpu                : {torch.cuda.get_device_name(device)}")
    print(f"V8 config version  : {ensemble_config.get('version')}")
    print(f"target             : {TEMP_NAME} only")
    print(f"weight grid        : {WEIGHT_GRID.tolist()}")
    print("inactive targets   : EXACT current V8")
    print("IMPORTANT          : offline evaluation only; API/callback/config are untouched")

    for i, gid in enumerate(group_id):
        print(f"V8 inference [{i+1}/{len(group_id)}] {gid} ...", flush=True)
        history_df = pd.DataFrame(history_phys[i], columns=TARGET_COLUMNS)
        pred = np.asarray(predict_future(history_df), dtype=np.float64)
        if pred.shape != true_phys[i].shape:
            raise RuntimeError(f"unexpected V8 shape {pred.shape} for {gid}")
        v8_phys[i] = pred

    v8_z = (v8_phys - center3) / scale3

    # True LOSO: each holdout uses a weight selected only from the other four sequences.
    fold_weights: list[float] = []
    fold_gains: list[float] = []
    fold_rows: dict[str, dict] = {}
    loso_pred_z = v8_z.copy()

    print("\n" + "=" * 104)
    print("LOSO TEMPERATURE RESULTS")
    print("=" * 104)
    for holdout in range(len(group_id)):
        train_idx = np.asarray([i for i in range(len(group_id)) if i != holdout], dtype=np.int64)
        w, train_r, train_g = _choose_train_weight(v8_z, patch_z, Y, train_idx)
        loso_pred_z[holdout, :, TEMP_IDX] = (
            (1.0 - w) * v8_z[holdout, :, TEMP_IDX] + w * patch_z[holdout, :, TEMP_IDX]
        )
        base_r = rmse(v8_z[holdout, :, TEMP_IDX], Y[holdout, :, TEMP_IDX])
        patch_r = rmse(patch_z[holdout, :, TEMP_IDX], Y[holdout, :, TEMP_IDX])
        hybrid_r = rmse(loso_pred_z[holdout, :, TEMP_IDX], Y[holdout, :, TEMP_IDX])
        g = gain(base_r, hybrid_r)
        fold_weights.append(w)
        fold_gains.append(g)
        fold_rows[str(group_id[holdout])] = {
            "weight_patchtst": w,
            "train_rmse_z": train_r,
            "train_gain_vs_v8_pct": train_g,
            "v8_rmse_z": base_r,
            "patchtst_rmse_z": patch_r,
            "hybrid_rmse_z": hybrid_r,
            "hybrid_gain_vs_v8_pct": g,
        }
        print(
            f"{group_id[holdout]:14s}: w={w:.2f} train_gain={train_g:+.2f}% | "
            f"V8={base_r:.6f} Patch={patch_r:.6f} Hybrid={hybrid_r:.6f} gain={g:+.2f}%"
        )

    loso_base = rmse(v8_z[:, :, TEMP_IDX], Y[:, :, TEMP_IDX])
    loso_hybrid = rmse(loso_pred_z[:, :, TEMP_IDX], Y[:, :, TEMP_IDX])
    loso_gain = gain(loso_base, loso_hybrid)
    min_fold_gain = float(np.min(fold_gains))
    positive_folds = int(np.sum(np.asarray(fold_gains) > 0.0))

    # Deployment candidate: robust consensus of the LOSO-selected weights.
    deploy_weight = float(np.median(np.asarray(fold_weights, dtype=np.float64)))
    fixed_z = v8_z.copy()
    fixed_z[:, :, TEMP_IDX] = (
        (1.0 - deploy_weight) * v8_z[:, :, TEMP_IDX]
        + deploy_weight * patch_z[:, :, TEMP_IDX]
    )
    fixed_phys = fixed_z * scale3 + center3

    fixed_base_z = rmse(v8_z[:, :, TEMP_IDX], Y[:, :, TEMP_IDX])
    fixed_hybrid_z = rmse(fixed_z[:, :, TEMP_IDX], Y[:, :, TEMP_IDX])
    fixed_gain_z = gain(fixed_base_z, fixed_hybrid_z)
    fixed_base_phys = rmse(v8_phys[:, :, TEMP_IDX], true_phys[:, :, TEMP_IDX])
    fixed_hybrid_phys = rmse(fixed_phys[:, :, TEMP_IDX], true_phys[:, :, TEMP_IDX])
    fixed_gain_phys = gain(fixed_base_phys, fixed_hybrid_phys)

    fixed_fold_gains: list[float] = []
    print("\n" + "=" * 104)
    print(f"FIXED CANDIDATE w_patch={deploy_weight:.2f}")
    print("=" * 104)
    for i, gid in enumerate(group_id):
        b = rmse(v8_z[i, :, TEMP_IDX], Y[i, :, TEMP_IDX])
        h = rmse(fixed_z[i, :, TEMP_IDX], Y[i, :, TEMP_IDX])
        g = gain(b, h)
        fixed_fold_gains.append(g)
        print(f"{gid:14s}: V8={b:.6f} Hybrid={h:.6f} gain={g:+.2f}%")

    dv8 = np.diff(v8_phys[:, :, TEMP_IDX], axis=1).reshape(-1)
    dh = np.diff(fixed_phys[:, :, TEMP_IDX], axis=1).reshape(-1)
    dy = np.diff(true_phys[:, :, TEMP_IDX], axis=1).reshape(-1)
    corr_v8 = safe_corr(dv8, dy)
    corr_fixed = safe_corr(dh, dy)
    corr_delta = corr_fixed - corr_v8

    fixed_min_gain = float(np.min(fixed_fold_gains))
    fixed_positive = int(np.sum(np.asarray(fixed_fold_gains) > 0.0))

    # Strict gate: temperature-only candidate must improve every official sequence,
    # show meaningful pooled gain, and preserve V8's excellent difference correlation.
    gate = bool(
        positive_folds == len(group_id)
        and min_fold_gain > 0.0
        and loso_gain >= 10.0
        and fixed_positive == len(group_id)
        and fixed_min_gain > 0.0
        and fixed_gain_z >= 10.0
        and fixed_gain_phys >= 10.0
        and (not np.isfinite(corr_delta) or corr_delta >= -0.005)
    )

    print("\n" + "-" * 104)
    print(f"LOSO positive folds : {positive_folds}/{len(group_id)}")
    print(f"LOSO pooled gain_z  : {loso_gain:+.2f}%")
    print(f"LOSO min fold gain  : {min_fold_gain:+.2f}%")
    print(f"deploy weight       : {deploy_weight:.2f}")
    print(f"fixed positive folds: {fixed_positive}/{len(group_id)}")
    print(f"fixed pooled gain_z : {fixed_gain_z:+.2f}%")
    print(f"fixed physical gain : {fixed_gain_phys:+.2f}%")
    print(f"fixed min fold gain : {fixed_min_gain:+.2f}%")
    print(f"temperature diffCorr: V8={corr_v8:+.4f} Hybrid={corr_fixed:+.4f} delta={corr_delta:+.4f}")
    print(f"TEMP-ONLY GATE      : {'PASS' if gate else 'REJECT'}")

    result = {
        "model": "v8_plus_patchtst_temperature_only_candidate",
        "target": TEMP_NAME,
        "weight_grid": WEIGHT_GRID.tolist(),
        "fold_weights": fold_weights,
        "fold_gains_vs_v8_pct": fold_gains,
        "folds": fold_rows,
        "loso_positive_folds": positive_folds,
        "loso_pooled_gain_z_pct": loso_gain,
        "loso_min_fold_gain_pct": min_fold_gain,
        "deploy_weight_patchtst": deploy_weight,
        "fixed_fold_gains_vs_v8_pct": fixed_fold_gains,
        "fixed_positive_folds": fixed_positive,
        "fixed_pooled_gain_z_pct": fixed_gain_z,
        "fixed_physical_gain_pct": fixed_gain_phys,
        "fixed_min_fold_gain_pct": fixed_min_gain,
        "v8_temperature_diff_corr": corr_v8,
        "candidate_temperature_diff_corr": corr_fixed,
        "diff_corr_delta": corr_delta,
        "offline_gate_pass": gate,
        "gate_rule": (
            "LOSO and fixed candidate must improve all 5 official sequences; LOSO/fixed normalized pooled gain >=10%; "
            "fixed physical gain >=10%; temperature diffCorr delta >= -0.005"
        ),
        "important_note": (
            "Offline research gate only. All non-temperature targets remain exactly current V8. "
            "No API/callback/online ensemble configuration is modified."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics             : {OUTPUT_PATH}")
    print("NOTE: online V8/API/callback files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
