from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from src.deep.tcn_forecaster import MaskedTCNForecaster, TCNConfig


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
DEFAULT_CHECKPOINT = ROOT / "models" / "deep" / "tcn_v1_pretrain.pt"
DEFAULT_OUTPUT = ROOT / "models" / "deep" / "tcn_v1_official_loso_calibration.json"

TARGETS = [
    "vibration_rms",
    "temperature_c",
    "current_a",
    "speed_rpm",
    "acoustic_db",
    "pressure_kpa",
]
ACTIVE_TARGETS = ["vibration_rms", "temperature_c", "current_a", "pressure_kpa"]
ACTIVE_INDEX = [TARGETS.index(x) for x in ACTIVE_TARGETS]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(d))))


def gain(base: float, model: float) -> float:
    return 100.0 * (base - model) / max(abs(base), 1e-12)


def config_from_checkpoint(raw: dict) -> TCNConfig:
    cfg = raw.get("config", {})
    allowed = {f.name for f in fields(TCNConfig)}
    return TCNConfig(**{k: v for k, v in dict(cfg).items() if k in allowed})


def per_sequence_alpha(pred_z: np.ndarray, persistence_z: np.ndarray, true_z: np.ndarray) -> float:
    """Optimal residual shrinkage for one sequence, clipped to [0, 1]."""
    r = np.asarray(pred_z - persistence_z, dtype=np.float64).reshape(-1)
    d = np.asarray(true_z - persistence_z, dtype=np.float64).reshape(-1)
    denom = float(np.dot(r, r))
    if not np.isfinite(denom) or denom < 1e-12:
        return 0.0
    alpha = float(np.dot(r, d) / denom)
    if not np.isfinite(alpha):
        return 0.0
    return float(np.clip(alpha, 0.0, 1.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.corpus.is_file():
        raise FileNotFoundError(args.corpus)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )

    data = np.load(args.corpus, allow_pickle=False)
    X = data["X"].astype(np.float32, copy=False)
    Y = data["Y"].astype(np.float64, copy=False)
    mask = data["mask"].astype(np.float32, copy=False)
    center = data["center"].astype(np.float64, copy=False)
    scale = data["scale"].astype(np.float64, copy=False)
    group_id = data["group_id"].astype(str)
    targets = data["targets"].astype(str).tolist()
    if targets != TARGETS:
        raise RuntimeError(f"target mismatch: {targets}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MaskedTCNForecaster(config_from_checkpoint(ckpt)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        tx = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        tm = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
        pred_z = model(tx, tm).cpu().numpy().astype(np.float64)

    persistence_z = np.repeat(X[:, -1:, :].astype(np.float64), Y.shape[1], axis=1)
    center3 = center[:, None, :]
    scale3 = scale[:, None, :]

    folds: list[dict] = []
    calibrated_all = np.zeros_like(pred_z)
    calibrated_all[:] = persistence_z

    print("=" * 100)
    print("TCN V1 OFFICIAL LOSO RESIDUAL CALIBRATION")
    print("=" * 100)
    print(f"device             : {device}")
    if device.type == "cuda":
        print(f"gpu                : {torch.cuda.get_device_name(device)}")
    print(f"samples            : {len(X)}")
    print("adaptation          : frozen TCN; only residual alpha is estimated from the other 4 sequences")
    print("alpha rule          : per-train-sequence optimum -> median -> clip [0,1]")

    for holdout in range(len(X)):
        train_idx = [i for i in range(len(X)) if i != holdout]
        fold = {"holdout": str(group_id[holdout]), "targets": {}}
        print("\n" + "-" * 100)
        print(f"holdout: {group_id[holdout]}")

        for j in ACTIVE_INDEX:
            name = TARGETS[j]
            train_alphas = [
                per_sequence_alpha(pred_z[i, :, j], persistence_z[i, :, j], Y[i, :, j])
                for i in train_idx if mask[i, j] > 0.5
            ]
            alpha = float(np.median(train_alphas)) if train_alphas else 0.0
            calibrated_all[holdout, :, j] = (
                persistence_z[holdout, :, j]
                + alpha * (pred_z[holdout, :, j] - persistence_z[holdout, :, j])
            )

            base = rmse(persistence_z[holdout, :, j], Y[holdout, :, j])
            zero = rmse(pred_z[holdout, :, j], Y[holdout, :, j])
            cal = rmse(calibrated_all[holdout, :, j], Y[holdout, :, j])
            item = {
                "alpha": alpha,
                "train_sequence_alphas": [float(x) for x in train_alphas],
                "persistence_rmse_z": base,
                "zero_shot_rmse_z": zero,
                "calibrated_rmse_z": cal,
                "zero_shot_gain_pct": gain(base, zero),
                "calibrated_gain_pct": gain(base, cal),
                "calibrated_vs_zero_pct": gain(zero, cal),
            }
            fold["targets"][name] = item
            print(
                f"  {name:16s}: alpha={alpha:.3f} | persist={base:.6f} "
                f"zero={zero:.6f} ({item['zero_shot_gain_pct']:+.2f}%) "
                f"cal={cal:.6f} ({item['calibrated_gain_pct']:+.2f}%)"
            )
        folds.append(fold)

    print("\n" + "=" * 100)
    print("POOLED LOSO RESULTS")
    print("=" * 100)

    pooled: dict[str, dict] = {}
    gains = []
    positive = 0
    zero_gains = []
    physical_gains = []

    for j in ACTIVE_INDEX:
        name = TARGETS[j]
        valid = mask[:, j] > 0.5
        base_z = rmse(persistence_z[valid, :, j], Y[valid, :, j])
        zero_z = rmse(pred_z[valid, :, j], Y[valid, :, j])
        cal_z = rmse(calibrated_all[valid, :, j], Y[valid, :, j])

        true_phys = Y[valid, :, j] * scale3[valid, :, j] + center3[valid, :, j]
        base_phys = persistence_z[valid, :, j] * scale3[valid, :, j] + center3[valid, :, j]
        cal_phys = calibrated_all[valid, :, j] * scale3[valid, :, j] + center3[valid, :, j]
        base_p = rmse(base_phys, true_phys)
        cal_p = rmse(cal_phys, true_phys)

        cal_gain = gain(base_z, cal_z)
        zero_gain = gain(base_z, zero_z)
        phys_gain = gain(base_p, cal_p)
        item = {
            "persistence_rmse_z": base_z,
            "zero_shot_rmse_z": zero_z,
            "calibrated_rmse_z": cal_z,
            "zero_shot_gain_pct": zero_gain,
            "calibrated_gain_pct": cal_gain,
            "calibrated_vs_zero_pct": gain(zero_z, cal_z),
            "physical_persistence_rmse": base_p,
            "physical_calibrated_rmse": cal_p,
            "physical_calibrated_gain_pct": phys_gain,
        }
        pooled[name] = item
        gains.append(cal_gain)
        zero_gains.append(zero_gain)
        physical_gains.append(phys_gain)
        positive += int(cal_gain > 0.0)
        print(
            f"{name:16s}: persist_z={base_z:.6f} zero_z={zero_z:.6f} cal_z={cal_z:.6f} "
            f"| zero gain={zero_gain:+.2f}% cal gain={cal_gain:+.2f}% "
            f"| physical cal gain={phys_gain:+.2f}%"
        )

    macro_gain = float(np.mean(gains))
    median_gain = float(np.median(gains))
    macro_zero_gain = float(np.mean(zero_gains))
    macro_phys_gain = float(np.mean(physical_gains))

    # Conservative candidate gate: LOSO calibration must beat persistence on at least 3/4 targets,
    # have positive median normalized gain, and improve macro normalized gain over raw zero-shot.
    gate_pass = bool(positive >= 3 and median_gain > 0.0 and macro_gain > macro_zero_gain)

    print("\n" + "-" * 100)
    print(f"positive targets   : {positive}/{len(ACTIVE_TARGETS)}")
    print(f"macro zero gain_z  : {macro_zero_gain:+.2f}%")
    print(f"macro cal gain_z   : {macro_gain:+.2f}%")
    print(f"median cal gain_z  : {median_gain:+.2f}%")
    print(f"macro physical gain: {macro_phys_gain:+.2f}%")
    print(f"LOSO CALIB GATE    : {'PASS' if gate_pass else 'REJECT'}")

    result = {
        "model": "tcn_v1_official_loso_residual_calibration",
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "active_targets": ACTIVE_TARGETS,
        "folds": folds,
        "pooled": pooled,
        "positive_targets": positive,
        "macro_zero_shot_gain_z_pct": macro_zero_gain,
        "macro_calibrated_gain_z_pct": macro_gain,
        "median_calibrated_gain_z_pct": median_gain,
        "macro_physical_calibrated_gain_pct": macro_phys_gain,
        "gate_pass": gate_pass,
        "gate_rule": "positive LOSO normalized gain on >=3/4 targets, positive median gain, and macro calibrated gain > macro raw zero-shot gain",
        "important_note": "TCN weights remain frozen. Only a per-target residual shrinkage alpha is learned from the other four official sequences in each fold. Online V8/API/callback are untouched.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics            : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
