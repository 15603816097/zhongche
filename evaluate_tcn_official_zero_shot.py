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
DEFAULT_METRICS = ROOT / "models" / "deep" / "tcn_v1_official_zero_shot.json"

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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    a = a[finite]
    b = b[finite]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _direction_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    return float(np.mean(np.sign(a[finite]) == np.sign(b[finite])))


def _volatility_ratio(pred_diff: np.ndarray, true_diff: np.ndarray) -> float:
    p = float(np.std(pred_diff))
    t = float(np.std(true_diff))
    if not np.isfinite(t) or t < 1e-12:
        return float("nan")
    return p / t


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(d))))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(d)))


def _gain(baseline: float, model: float) -> float:
    return 100.0 * (baseline - model) / max(abs(baseline), 1e-12)


def _config_from_checkpoint(raw: dict) -> TCNConfig:
    cfg = raw.get("config", {})
    allowed = {f.name for f in fields(TCNConfig)}
    cfg = {k: v for k, v in dict(cfg).items() if k in allowed}
    return TCNConfig(**cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.corpus.is_file():
        raise FileNotFoundError(f"missing official corpus: {args.corpus}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"missing TCN checkpoint: {args.checkpoint}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = np.load(args.corpus, allow_pickle=False)
    X = data["X"].astype(np.float32, copy=False)
    Y = data["Y"].astype(np.float32, copy=False)
    mask = data["mask"].astype(np.float32, copy=False)
    center = data["center"].astype(np.float32, copy=False)
    scale = data["scale"].astype(np.float32, copy=False)
    group_id = data["group_id"].astype(str)
    targets = data["targets"].astype(str).tolist()
    if targets != TARGETS:
        raise RuntimeError(f"target mismatch: {targets}")
    if X.shape[0] != 5:
        print(f"WARNING: expected 5 official samples, got {X.shape[0]}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = _config_from_checkpoint(checkpoint)
    model = MaskedTCNForecaster(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        tx = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        tm = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
        pred_z = model(tx, tm).cpu().numpy().astype(np.float64)

    true_z = Y.astype(np.float64)
    persistence_z = np.repeat(X[:, -1:, :].astype(np.float64), Y.shape[1], axis=1)

    center64 = center.astype(np.float64)[:, None, :]
    scale64 = scale.astype(np.float64)[:, None, :]
    pred_phys = pred_z * scale64 + center64
    true_phys = true_z * scale64 + center64
    persistence_phys = persistence_z * scale64 + center64

    per_sequence: dict[str, dict] = {}
    pooled: dict[str, dict] = {}

    print("=" * 100)
    print("TCN V1 OFFICIAL-DOMAIN ZERO-SHOT EVALUATION")
    print("=" * 100)
    print(f"device             : {device}")
    if device.type == "cuda":
        print(f"gpu                : {torch.cuda.get_device_name(device)}")
    print(f"checkpoint         : {args.checkpoint}")
    print(f"official corpus    : {args.corpus}")
    print(f"samples            : {len(X)}")
    print(f"active targets     : {ACTIVE_TARGETS}")
    print("NOTE               : no official sample was used to update model weights in this test")

    for i, gid in enumerate(group_id):
        row: dict[str, dict] = {}
        print("\n" + "-" * 100)
        print(gid)
        for j in ACTIVE_INDEX:
            name = TARGETS[j]
            if mask[i, j] <= 0.5:
                continue
            p = pred_phys[i, :, j]
            y = true_phys[i, :, j]
            b = persistence_phys[i, :, j]
            dp = np.diff(p)
            dy = np.diff(y)
            item = {
                "rmse": _rmse(p, y),
                "persistence_rmse": _rmse(b, y),
                "rmse_gain_pct": 0.0,
                "mae": _mae(p, y),
                "persistence_mae": _mae(b, y),
                "diff_corr": _safe_corr(dp, dy),
                "direction_accuracy": _direction_accuracy(dp, dy),
                "volatility_ratio": _volatility_ratio(dp, dy),
            }
            item["rmse_gain_pct"] = _gain(item["persistence_rmse"], item["rmse"])
            row[name] = item
            print(
                f"  {name:16s}: RMSE={item['rmse']:.6f} "
                f"persist={item['persistence_rmse']:.6f} "
                f"gain={item['rmse_gain_pct']:+.2f}% "
                f"diff_corr={item['diff_corr']:+.4f} "
                f"dir={item['direction_accuracy']:.3f} "
                f"vol={item['volatility_ratio']:.3f}"
            )
        per_sequence[str(gid)] = row

    print("\n" + "=" * 100)
    print("POOLED OFFICIAL RESULTS")
    print("=" * 100)

    target_gains: list[float] = []
    positive_targets = 0
    for j in ACTIVE_INDEX:
        name = TARGETS[j]
        valid = mask[:, j] > 0.5
        p = pred_phys[valid, :, j]
        y = true_phys[valid, :, j]
        b = persistence_phys[valid, :, j]
        pz = pred_z[valid, :, j]
        yz = true_z[valid, :, j]
        bz = persistence_z[valid, :, j]

        rmse_phys = _rmse(p, y)
        base_rmse_phys = _rmse(b, y)
        rmse_z = _rmse(pz, yz)
        base_rmse_z = _rmse(bz, yz)
        gain_z = _gain(base_rmse_z, rmse_z)
        gain_phys = _gain(base_rmse_phys, rmse_phys)
        dp = np.diff(p, axis=1).reshape(-1)
        dy = np.diff(y, axis=1).reshape(-1)

        item = {
            "rmse": rmse_phys,
            "persistence_rmse": base_rmse_phys,
            "rmse_gain_pct": gain_phys,
            "rmse_z": rmse_z,
            "persistence_rmse_z": base_rmse_z,
            "rmse_z_gain_pct": gain_z,
            "mae": _mae(p, y),
            "persistence_mae": _mae(b, y),
            "diff_corr": _safe_corr(dp, dy),
            "direction_accuracy": _direction_accuracy(dp, dy),
            "volatility_ratio": _volatility_ratio(dp, dy),
            "samples": int(valid.sum()),
        }
        pooled[name] = item
        target_gains.append(gain_z)
        if gain_z > 0.0:
            positive_targets += 1
        print(
            f"{name:16s}: RMSE_z={rmse_z:.6f} persist_z={base_rmse_z:.6f} "
            f"gain_z={gain_z:+.2f}% | physical RMSE={rmse_phys:.6f} "
            f"gain={gain_phys:+.2f}% | diff_corr={item['diff_corr']:+.4f}"
        )

    macro_gain_z = float(np.mean(target_gains)) if target_gains else float("nan")
    median_gain_z = float(np.median(target_gains)) if target_gains else float("nan")

    # Zero-shot gate is deliberately modest: domain transfer is accepted only if the
    # pretrained model beats persistence on at least half the active targets and the
    # median target gain is positive. Fine-tuning is considered only after this test.
    gate_pass = bool(positive_targets >= 2 and median_gain_z > 0.0)

    print("\n" + "-" * 100)
    print(f"positive targets   : {positive_targets}/{len(ACTIVE_TARGETS)}")
    print(f"macro gain_z       : {macro_gain_z:+.2f}%")
    print(f"median gain_z      : {median_gain_z:+.2f}%")
    print(f"ZERO-SHOT GATE     : {'PASS' if gate_pass else 'REJECT'}")

    result = {
        "model": "tcn_v1_source_pretrain_official_zero_shot",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
        "samples": int(len(X)),
        "active_targets": ACTIVE_TARGETS,
        "per_sequence": per_sequence,
        "pooled": pooled,
        "positive_targets": int(positive_targets),
        "macro_gain_z_pct": macro_gain_z,
        "median_gain_z_pct": median_gain_z,
        "zero_shot_gate_pass": gate_pass,
        "gate_rule": "positive normalized-RMSE gain on >=2/4 active targets and positive median gain",
        "important_note": (
            "This is a zero-shot domain-transfer test. Official samples are evaluation data only; "
            "no fine-tuning or online V8/API/callback modification occurs here."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics            : {args.output}")
    print("NOTE: online V8/API/callback files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
