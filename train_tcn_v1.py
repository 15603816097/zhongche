from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.deep.tcn_forecaster import MaskedTCNForecaster, TCNConfig


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "pretrain_corpus_v1.npz"
MODEL_DIR = ROOT / "models" / "deep"
TARGETS = [
    "vibration_rms",
    "temperature_c",
    "current_a",
    "speed_rpm",
    "acoustic_db",
    "pressure_kpa",
]

# V1 deliberately excludes speed (no external examples) and acoustic (only 5 examples).
# These channels stay on the V8 branch later unless stronger external data are added.
ACTIVE_TARGETS = ["vibration_rms", "temperature_c", "current_a", "pressure_kpa"]
ACTIVE_INDEX = [TARGETS.index(x) for x in ACTIVE_TARGETS]


class CorpusDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, mask: np.ndarray, indices: np.ndarray):
        self.X = X
        self.Y = Y
        self.mask = mask
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        k = self.indices[i]
        return (
            torch.from_numpy(self.X[k]),
            torch.from_numpy(self.Y[k]),
            torch.from_numpy(self.mask[k]),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def modality_balanced_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    active_index: list[int],
) -> torch.Tensor:
    losses = []
    for j in active_index:
        valid = mask[:, j] > 0.5
        if not torch.any(valid):
            continue
        losses.append(torch.mean((pred[valid, :, j] - target[valid, :, j]) ** 2))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def modality_balanced_diff_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    active_index: list[int],
) -> torch.Tensor:
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dt = target[:, 1:, :] - target[:, :-1, :]
    return modality_balanced_mse(dp, dt, mask, active_index)


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    diff_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    level = modality_balanced_mse(pred, target, mask, ACTIVE_INDEX)
    diff = modality_balanced_diff_mse(pred, target, mask, ACTIVE_INDEX)
    return level + float(diff_weight) * diff, level, diff


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    sq_error = {j: 0.0 for j in ACTIVE_INDEX}
    count = {j: 0 for j in ACTIVE_INDEX}
    persistence_sq_error = {j: 0.0 for j in ACTIVE_INDEX}
    diff_sq_error = {j: 0.0 for j in ACTIVE_INDEX}
    diff_count = {j: 0 for j in ACTIVE_INDEX}

    for x, y, mask in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
        pred = model(x, mask)
        persistence = x[:, -1:, :].expand_as(y)

        for j in ACTIVE_INDEX:
            valid = mask[:, j] > 0.5
            if not torch.any(valid):
                continue
            e = pred[valid, :, j] - y[valid, :, j]
            pe = persistence[valid, :, j] - y[valid, :, j]
            sq_error[j] += float(torch.sum(e * e).cpu())
            persistence_sq_error[j] += float(torch.sum(pe * pe).cpu())
            count[j] += int(e.numel())

            de = (pred[valid, 1:, j] - pred[valid, :-1, j]) - (
                y[valid, 1:, j] - y[valid, :-1, j]
            )
            diff_sq_error[j] += float(torch.sum(de * de).cpu())
            diff_count[j] += int(de.numel())

    per_target = {}
    rmse_values = []
    persistence_values = []
    diff_values = []
    for j in ACTIVE_INDEX:
        name = TARGETS[j]
        if count[j] <= 0:
            continue
        rmse = math.sqrt(sq_error[j] / count[j])
        persistence_rmse = math.sqrt(persistence_sq_error[j] / count[j])
        diff_rmse = math.sqrt(diff_sq_error[j] / max(1, diff_count[j]))
        per_target[name] = {
            "rmse_z": rmse,
            "persistence_rmse_z": persistence_rmse,
            "improvement_pct": 100.0 * (persistence_rmse - rmse) / max(persistence_rmse, 1e-12),
            "diff_rmse_z": diff_rmse,
            "points": count[j],
        }
        rmse_values.append(rmse)
        persistence_values.append(persistence_rmse)
        diff_values.append(diff_rmse)

    macro_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")
    macro_persistence = float(np.mean(persistence_values)) if persistence_values else float("nan")
    macro_improvement = (
        100.0 * (macro_persistence - macro_rmse) / max(macro_persistence, 1e-12)
        if np.isfinite(macro_rmse) and np.isfinite(macro_persistence)
        else float("nan")
    )
    return {
        "macro_rmse_z": macro_rmse,
        "macro_persistence_rmse_z": macro_persistence,
        "macro_improvement_pct": macro_improvement,
        "macro_diff_rmse_z": float(np.mean(diff_values)) if diff_values else float("nan"),
        "per_target": per_target,
    }


def build_source_balanced_sampler(source: np.ndarray, train_indices: np.ndarray, seed: int):
    src = source[train_indices]
    unique, counts = np.unique(src, return_counts=True)
    freq = {str(k): int(v) for k, v in zip(unique, counts)}
    weights = np.asarray([1.0 / freq[str(s)] for s in src], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )
    return sampler, freq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--diff-weight", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=5)
    args = parser.parse_args()

    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"missing corpus: {CORPUS_PATH}")

    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = np.load(CORPUS_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32, copy=False)
    Y = data["Y"].astype(np.float32, copy=False)
    mask = data["mask"].astype(np.float32, copy=False)
    split = data["split"].astype(str)
    source = data["source"].astype(str)
    group_id = data["group_id"].astype(str)
    targets = data["targets"].astype(str).tolist()
    if targets != TARGETS:
        raise RuntimeError(f"target mismatch: {targets}")

    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) <= 0:
        raise RuntimeError("train/val/test split is incomplete")

    train_ds = CorpusDataset(X, Y, mask, train_idx)
    val_ds = CorpusDataset(X, Y, mask, val_idx)
    test_ds = CorpusDataset(X, Y, mask, test_idx)

    sampler, source_counts = build_source_balanced_sampler(source, train_idx, args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = TCNConfig(hidden_channels=args.hidden, num_blocks=args.blocks)
    model = MaskedTCNForecaster(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "tcn_v1_pretrain.pt"
    metrics_path = MODEL_DIR / "tcn_v1_metrics.json"

    print("=" * 100)
    print("TRAIN TCN V1 - SOURCE DOMAIN PRETRAIN")
    print("=" * 100)
    print(f"device             : {device}")
    if device.type == "cuda":
        print(f"gpu                : {torch.cuda.get_device_name(device)}")
    print(f"corpus             : {CORPUS_PATH}")
    print(f"X/Y                : {X.shape} / {Y.shape}")
    print(f"train/val/test     : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"train groups       : {len(np.unique(group_id[train_idx]))}")
    print(f"source counts      : {source_counts}")
    print(f"active targets     : {ACTIVE_TARGETS}")
    print("excluded V1        : speed_rpm(no external data), acoustic_db(only 5 samples)")
    print("sampler            : inverse-source-frequency balanced")
    print(f"parameters         : {sum(p.numel() for p in model.parameters()):,}")

    # Persistence baseline before training, for an interpretable source-domain gate.
    baseline_val = evaluate(model, val_loader, device)
    print("\nInitial model is persistence by construction")
    print(
        f"val macro RMSE     : {baseline_val['macro_rmse_z']:.6f} "
        f"(persistence {baseline_val['macro_persistence_rmse_z']:.6f})"
    )

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        running_level = 0.0
        running_diff = 0.0
        batches = 0

        for x, y, m in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            m = m.to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, m)
            loss, level_loss, diff_loss = total_loss(pred, y, m, args.diff_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += float(loss.detach().cpu())
            running_level += float(level_loss.detach().cpu())
            running_diff += float(diff_loss.detach().cpu())
            batches += 1

        val_metrics = evaluate(model, val_loader, device)
        val_rmse = float(val_metrics["macro_rmse_z"])
        scheduler.step(val_rmse)
        lr = float(optimizer.param_groups[0]["lr"])

        epoch_row = {
            "epoch": epoch,
            "train_loss": running / max(1, batches),
            "train_level_loss": running_level / max(1, batches),
            "train_diff_loss": running_diff / max(1, batches),
            "val_macro_rmse_z": val_rmse,
            "val_macro_improvement_pct": float(val_metrics["macro_improvement_pct"]),
            "lr": lr,
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:03d} | loss={epoch_row['train_loss']:.5f} "
            f"| val={val_rmse:.5f} "
            f"| vs persistence={epoch_row['val_macro_improvement_pct']:+.2f}% "
            f"| lr={lr:.2e}"
        )

        if val_rmse < best_val - 1e-5:
            best_val = val_rmse
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "targets": TARGETS,
                    "active_targets": ACTIVE_TARGETS,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "normalization": "history median/IQR from pretrain_corpus_v1",
                    "note": "source-domain TCN only; not activated online",
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - started

    # Conservative source-domain gate: require >=1% macro improvement over persistence on VAL.
    # Test is reported only as a sanity check and is not used to choose the checkpoint.
    gate_pass = bool(val_metrics["macro_improvement_pct"] >= 1.0)

    result = {
        "model": "tcn_v1_source_pretrain",
        "device": str(device),
        "best_epoch": int(best_epoch),
        "elapsed_seconds": float(elapsed),
        "active_targets": ACTIVE_TARGETS,
        "excluded_targets": {
            "speed_rpm": "0 external samples",
            "acoustic_db": "only 5 external samples; excluded from V1 to avoid false confidence",
        },
        "train_source_counts_before_balancing": source_counts,
        "val": val_metrics,
        "test": test_metrics,
        "gate": {
            "criterion": "validation macro normalized RMSE improves persistence by at least 1%",
            "pass": gate_pass,
        },
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "important_note": "This gate is source-domain only. Passing does not imply a better official score or authorize online deployment.",
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("TCN V1 RESULT")
    print("=" * 100)
    print(f"best epoch         : {best_epoch}")
    print(f"elapsed            : {elapsed:.1f}s")
    print(
        f"VAL macro          : {val_metrics['macro_rmse_z']:.6f} "
        f"vs persistence {val_metrics['macro_persistence_rmse_z']:.6f} "
        f"({val_metrics['macro_improvement_pct']:+.2f}%)"
    )
    print(
        f"TEST macro         : {test_metrics['macro_rmse_z']:.6f} "
        f"vs persistence {test_metrics['macro_persistence_rmse_z']:.6f} "
        f"({test_metrics['macro_improvement_pct']:+.2f}%)"
    )
    print("\nVAL per target:")
    for name, row in val_metrics["per_target"].items():
        print(
            f"  {name:16s}: rmse={row['rmse_z']:.6f} "
            f"persist={row['persistence_rmse_z']:.6f} "
            f"gain={row['improvement_pct']:+.2f}%"
        )
    print("\nTEST per target:")
    for name, row in test_metrics["per_target"].items():
        print(
            f"  {name:16s}: rmse={row['rmse_z']:.6f} "
            f"persist={row['persistence_rmse_z']:.6f} "
            f"gain={row['improvement_pct']:+.2f}%"
        )
    print(f"\nTCN V1 SOURCE-DOMAIN GATE: {'PASS' if gate_pass else 'REJECT'}")
    print(f"checkpoint         : {checkpoint_path}")
    print(f"metrics            : {metrics_path}")
    print("NOTE: online V8/API/callback files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
