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
from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster, PatchTSTConfig
from src.inference import predict_future
from src.v8_runtime import v8_enabled


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
PATCH_CHECKPOINT = ROOT / "models" / "deep" / "patchtst_v1_pretrain.pt"
OUTPUT_PATH = ROOT / "models" / "deep" / "patchtst_vs_v8_official_loso.json"
ACTIVE_TARGETS = ["vibration_rms", "temperature_c", "current_a", "pressure_kpa"]
ACTIVE_INDEX = [TARGET_COLUMNS.index(x) for x in ACTIVE_TARGETS]
WEIGHT_GRID = np.asarray([0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00], dtype=np.float64)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def gain(base: float, model: float) -> float:
    return 100.0 * (base - model) / max(abs(base), 1e-12)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def config_from_checkpoint(raw: dict) -> PatchTSTConfig:
    cfg = dict(raw.get("config", {}))
    allowed = {f.name for f in fields(PatchTSTConfig)}
    return PatchTSTConfig(**{k: v for k, v in cfg.items() if k in allowed})


def choose_weight_loso(
    v8_z: np.ndarray,
    patch_z: np.ndarray,
    true_z: np.ndarray,
    train_idx: np.ndarray,
    target_idx: int,
) -> tuple[float, float, float]:
    base = rmse(v8_z[train_idx, :, target_idx], true_z[train_idx, :, target_idx])
    best_w = 0.0
    best_rmse = base
    for w in WEIGHT_GRID:
        pred = (1.0 - w) * v8_z[train_idx, :, target_idx] + w * patch_z[train_idx, :, target_idx]
        score = rmse(pred, true_z[train_idx, :, target_idx])
        # tie-break toward the current V8 branch
        if score < best_rmse - 1e-9:
            best_rmse = score
            best_w = float(w)
    train_gain = gain(base, best_rmse)
    # Conservative rule: do not leave V8 unless the other 4 sequences show >=1% gain.
    if train_gain < 1.0:
        return 0.0, base, 0.0
    return best_w, best_rmse, train_gain


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
        raise RuntimeError(
            f"models/ensemble_config.pkl is not a V8-compatible config: "
            f"version={ensemble_config.get('version')} trajectory_model={ensemble_config.get('trajectory_model')}"
        )

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
    patch_phys = patch_z * scale3 + center3

    print("=" * 108)
    print("PATCHTST V1 vs CURRENT V8 - OFFICIAL FIVE-SEQUENCE OFFLINE LOSO")
    print("=" * 108)
    print(f"device             : {device}")
    if device.type == "cuda":
        print(f"gpu                : {torch.cuda.get_device_name(device)}")
    print(f"V8 config version  : {ensemble_config.get('version')}")
    print(f"V8 trajectory      : {ensemble_config.get('trajectory_model')}")
    print("selection          : leave-one-sequence-out grid blend; inactive targets remain V8")
    print("IMPORTANT          : no API/callback/online config is modified")

    v8_phys = np.empty_like(true_phys)
    for i, gid in enumerate(group_id):
        print(f"V8 inference [{i+1}/{len(group_id)}] {gid} ...", flush=True)
        history_df = pd.DataFrame(history_phys[i], columns=TARGET_COLUMNS)
        pred = np.asarray(predict_future(history_df), dtype=np.float64)
        if pred.shape != true_phys[i].shape:
            raise RuntimeError(f"unexpected V8 shape {pred.shape} for {gid}")
        v8_phys[i] = pred

    v8_z = (v8_phys - center3) / scale3
    hybrid_z = v8_z.copy()
    holdout_rows: dict[str, dict] = {}

    for holdout in range(len(group_id)):
        train_idx = np.asarray([i for i in range(len(group_id)) if i != holdout], dtype=np.int64)
        row: dict[str, dict] = {}
        print("\n" + "-" * 108)
        print(f"holdout: {group_id[holdout]}")
        for j in ACTIVE_INDEX:
            name = TARGET_COLUMNS[j]
            w, train_rmse, train_gain = choose_weight_loso(v8_z, patch_z, Y, train_idx, j)
            hybrid_z[holdout, :, j] = (1.0 - w) * v8_z[holdout, :, j] + w * patch_z[holdout, :, j]
            v8_r = rmse(v8_z[holdout, :, j], Y[holdout, :, j])
            patch_r = rmse(patch_z[holdout, :, j], Y[holdout, :, j])
            hybrid_r = rmse(hybrid_z[holdout, :, j], Y[holdout, :, j])
            row[name] = {
                "weight_patchtst": w,
                "train_rmse_z": train_rmse,
                "train_gain_vs_v8_pct": train_gain,
                "v8_rmse_z": v8_r,
                "patchtst_rmse_z": patch_r,
                "hybrid_rmse_z": hybrid_r,
                "hybrid_gain_vs_v8_pct": gain(v8_r, hybrid_r),
            }
            print(
                f"  {name:16s}: w_patch={w:.2f} train_gain={train_gain:+.2f}% | "
                f"V8={v8_r:.5f} Patch={patch_r:.5f} Hybrid={hybrid_r:.5f} "
                f"gain_vs_V8={gain(v8_r, hybrid_r):+.2f}%"
            )
        holdout_rows[str(group_id[holdout])] = row

    hybrid_phys = hybrid_z * scale3 + center3

    print("\n" + "=" * 108)
    print("POOLED RESULTS")
    print("=" * 108)
    per_target: dict[str, dict] = {}
    active_gains = []
    active_positive = 0
    max_active_degradation = 0.0
    v8_corrs = []
    hybrid_corrs = []

    for j, name in enumerate(TARGET_COLUMNS):
        v8_rz = rmse(v8_z[:, :, j], Y[:, :, j])
        patch_rz = rmse(patch_z[:, :, j], Y[:, :, j])
        hybrid_rz = rmse(hybrid_z[:, :, j], Y[:, :, j])
        v8_rp = rmse(v8_phys[:, :, j], true_phys[:, :, j])
        hybrid_rp = rmse(hybrid_phys[:, :, j], true_phys[:, :, j])
        g = gain(v8_rz, hybrid_rz)
        dv8 = np.diff(v8_phys[:, :, j], axis=1).reshape(-1)
        dh = np.diff(hybrid_phys[:, :, j], axis=1).reshape(-1)
        dy = np.diff(true_phys[:, :, j], axis=1).reshape(-1)
        cv8 = safe_corr(dv8, dy)
        ch = safe_corr(dh, dy)
        per_target[name] = {
            "v8_rmse_z": v8_rz,
            "patchtst_rmse_z": patch_rz,
            "hybrid_rmse_z": hybrid_rz,
            "hybrid_gain_vs_v8_z_pct": g,
            "v8_physical_rmse": v8_rp,
            "hybrid_physical_rmse": hybrid_rp,
            "hybrid_physical_gain_vs_v8_pct": gain(v8_rp, hybrid_rp),
            "v8_diff_corr": cv8,
            "hybrid_diff_corr": ch,
        }
        if j in ACTIVE_INDEX:
            active_gains.append(g)
            if g > 0.0:
                active_positive += 1
            max_active_degradation = max(max_active_degradation, max(0.0, -g))
            if np.isfinite(cv8):
                v8_corrs.append(cv8)
            if np.isfinite(ch):
                hybrid_corrs.append(ch)
        print(
            f"{name:16s}: V8_z={v8_rz:.6f} Patch_z={patch_rz:.6f} Hybrid_z={hybrid_rz:.6f} "
            f"gain_vs_V8={g:+.2f}% | physical_gain={gain(v8_rp, hybrid_rp):+.2f}% "
            f"| diff_corr V8={cv8:+.4f} Hybrid={ch:+.4f}"
        )

    v8_macro6 = float(np.mean([per_target[n]["v8_rmse_z"] for n in TARGET_COLUMNS]))
    hybrid_macro6 = float(np.mean([per_target[n]["hybrid_rmse_z"] for n in TARGET_COLUMNS]))
    macro6_gain = gain(v8_macro6, hybrid_macro6)
    active_macro_gain = float(np.mean(active_gains))
    active_median_gain = float(np.median(active_gains))
    v8_macro_corr = float(np.mean(v8_corrs)) if v8_corrs else float("nan")
    hybrid_macro_corr = float(np.mean(hybrid_corrs)) if hybrid_corrs else float("nan")
    corr_delta = hybrid_macro_corr - v8_macro_corr if np.isfinite(v8_macro_corr) and np.isfinite(hybrid_macro_corr) else float("nan")

    gate = bool(
        active_positive >= 2
        and active_macro_gain >= 1.0
        and macro6_gain >= 0.5
        and max_active_degradation <= 5.0
        and (not np.isfinite(corr_delta) or corr_delta >= -0.01)
    )

    print("\n" + "-" * 108)
    print(f"active positive     : {active_positive}/{len(ACTIVE_TARGETS)}")
    print(f"active macro gain   : {active_macro_gain:+.2f}%")
    print(f"active median gain  : {active_median_gain:+.2f}%")
    print(f"all-6 macro gain    : {macro6_gain:+.2f}%")
    print(f"max active degrade  : {max_active_degradation:.2f}%")
    print(f"active diffCorr     : V8={v8_macro_corr:+.4f} Hybrid={hybrid_macro_corr:+.4f} delta={corr_delta:+.4f}")
    print(f"HYBRID OFFLINE GATE : {'PASS' if gate else 'REJECT'}")

    result = {
        "model": "patchtst_v1_v8_loso_hybrid_offline",
        "active_targets": ACTIVE_TARGETS,
        "weight_grid": WEIGHT_GRID.tolist(),
        "holdout": holdout_rows,
        "pooled": per_target,
        "active_positive_targets": int(active_positive),
        "active_macro_gain_vs_v8_pct": active_macro_gain,
        "active_median_gain_vs_v8_pct": active_median_gain,
        "all6_macro_gain_vs_v8_pct": macro6_gain,
        "max_active_degradation_pct": max_active_degradation,
        "v8_active_macro_diff_corr": v8_macro_corr,
        "hybrid_active_macro_diff_corr": hybrid_macro_corr,
        "diff_corr_delta": corr_delta,
        "offline_gate_pass": gate,
        "gate_rule": (
            ">=2/4 active targets positive, active macro gain >=1%, all-6 macro gain >=0.5%, "
            "no active target degrades >5%, active diffCorr delta >= -0.01"
        ),
        "important_note": (
            "This is an offline research gate only. The five official samples are not representative training data, "
            "and no API/callback/online model configuration is modified."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics             : {OUTPUT_PATH}")
    print("NOTE: online V8/API/callback files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
