from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster, PatchTSTConfig
from train_tcn_v1 import (
    ACTIVE_TARGETS,
    TARGETS,
    CorpusDataset,
    build_source_balanced_sampler,
    evaluate,
    seed_everything,
    total_loss,
)


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "pretrain_corpus_v1.npz"
MODEL_DIR = ROOT / "models" / "deep"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--diff-weight", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
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

    config = PatchTSTConfig(
        input_length=X.shape[1],
        output_channels=X.shape[2],
        horizon=Y.shape[1],
        patch_length=args.patch_length,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.heads,
        num_layers=args.layers,
        dim_feedforward=args.d_model * 2,
    )
    model = MaskedPatchTSTForecaster(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "patchtst_v1_pretrain.pt"
    metrics_path = MODEL_DIR / "patchtst_v1_metrics.json"

    print("=" * 100)
    print("TRAIN PATCHTST V1 - SOURCE DOMAIN PRETRAIN")
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
    print(f"patches            : length={config.patch_length}, stride={config.stride}, count={config.num_patches}")
    print(f"transformer        : d_model={config.d_model}, heads={config.n_heads}, layers={config.num_layers}")
    print(f"parameters         : {sum(p.numel() for p in model.parameters()):,}")

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
                    "note": "source-domain PatchTST only; not activated online",
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

    gate_pass = bool(val_metrics["macro_improvement_pct"] >= 1.0)
    result = {
        "model": "patchtst_v1_source_pretrain",
        "device": str(device),
        "best_epoch": int(best_epoch),
        "elapsed_seconds": float(elapsed),
        "config": asdict(config),
        "active_targets": ACTIVE_TARGETS,
        "val": val_metrics,
        "test": test_metrics,
        "source_counts_train": source_counts,
        "source_domain_gate_pass": gate_pass,
        "gate_rule": "validation macro normalized-RMSE improvement >=1% over persistence",
        "history": history,
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("PATCHTST V1 RESULT")
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
    for name, item in val_metrics["per_target"].items():
        print(
            f"  {name:16s}: rmse={item['rmse_z']:.6f} "
            f"persist={item['persistence_rmse_z']:.6f} "
            f"gain={item['improvement_pct']:+.2f}%"
        )
    print("\nTEST per target:")
    for name, item in test_metrics["per_target"].items():
        print(
            f"  {name:16s}: rmse={item['rmse_z']:.6f} "
            f"persist={item['persistence_rmse_z']:.6f} "
            f"gain={item['improvement_pct']:+.2f}%"
        )

    print(f"\nPATCHTST V1 SOURCE-DOMAIN GATE: {'PASS' if gate_pass else 'REJECT'}")
    print(f"checkpoint         : {checkpoint_path}")
    print(f"metrics            : {metrics_path}")
    print("NOTE: online V8/API/callback files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
