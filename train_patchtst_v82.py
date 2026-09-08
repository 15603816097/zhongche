from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster, PatchTSTConfig
from train_tcn_v1 import (
    TARGETS,
    CorpusDataset,
    build_source_balanced_sampler,
    seed_everything,
)


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "pretrain_corpus_v1.npz"
MODEL_DIR = ROOT / "models" / "deep"
V1_CHECKPOINT = MODEL_DIR / "patchtst_v1_pretrain.pt"
V82_CHECKPOINT = MODEL_DIR / "patchtst_v82_trend.pt"
V82_METRICS = MODEL_DIR / "patchtst_v82_trend_metrics.json"
TEMP_NAME = "temperature_c"
TEMP_IDX = TARGETS.index(TEMP_NAME)


def config_from_checkpoint(raw: dict) -> PatchTSTConfig:
    cfg = dict(raw.get("config", {}))
    allowed = {f.name for f in fields(PatchTSTConfig)}
    return PatchTSTConfig(**{k: v for k, v in cfg.items() if k in allowed})


def temperature_trend_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    diff_weight: float,
    direction_weight: float,
    endpoint_weight: float,
    direction_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = mask[:, TEMP_IDX] > 0.5
    if not torch.any(valid):
        zero = pred.sum() * 0.0
        return zero, {"level": zero, "diff": zero, "direction": zero, "endpoint": zero}

    p = pred[valid, :, TEMP_IDX]
    y = target[valid, :, TEMP_IDX]

    level = torch.mean((p - y) ** 2)
    dp = p[:, 1:] - p[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    diff = torch.mean((dp - dy) ** 2)
    endpoint = torch.mean((p[:, -1] - y[:, -1]) ** 2)

    # Bounded, differentiable sign agreement. Tiny true changes are down-weighted so
    # the optimizer does not manufacture oscillation merely to satisfy a sign loss.
    scale = max(float(direction_scale), 1e-3)
    importance = torch.clamp(torch.abs(dy) / 0.05, min=0.0, max=1.0)
    soft_same_sign = torch.tanh(scale * dp) * torch.tanh(scale * dy)
    direction_terms = (1.0 - soft_same_sign) * importance
    direction = direction_terms.sum() / torch.clamp(importance.sum(), min=1.0)

    total = (
        level
        + float(diff_weight) * diff
        + float(direction_weight) * direction
        + float(endpoint_weight) * endpoint
    )
    return total, {
        "level": level,
        "diff": diff,
        "direction": direction,
        "endpoint": endpoint,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate_temperature(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    pred_all: list[np.ndarray] = []
    true_all: list[np.ndarray] = []
    last_all: list[np.ndarray] = []

    for x, y, mask in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
        valid = mask[:, TEMP_IDX] > 0.5
        if not torch.any(valid):
            continue
        pred = model(x, mask)
        pred_all.append(pred[valid, :, TEMP_IDX].cpu().numpy().astype(np.float64))
        true_all.append(y[valid, :, TEMP_IDX].cpu().numpy().astype(np.float64))
        last_all.append(x[valid, -1, TEMP_IDX].cpu().numpy().astype(np.float64))

    if not pred_all:
        raise RuntimeError("no valid temperature samples in evaluation loader")

    p = np.concatenate(pred_all, axis=0)
    y = np.concatenate(true_all, axis=0)
    last = np.concatenate(last_all, axis=0)
    persistence = np.repeat(last[:, None], y.shape[1], axis=1)

    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    persistence_rmse = float(np.sqrt(np.mean((persistence - y) ** 2)))
    dp = np.diff(p, axis=1)
    dy = np.diff(y, axis=1)
    diff_rmse = float(np.sqrt(np.mean((dp - dy) ** 2)))
    endpoint_rmse = float(np.sqrt(np.mean((p[:, -1] - y[:, -1]) ** 2)))

    threshold = max(1e-6, 0.10 * float(np.std(dy)))
    meaningful = np.abs(dy) >= threshold
    if np.any(meaningful):
        direction_accuracy = float(np.mean(np.sign(dp[meaningful]) == np.sign(dy[meaningful])))
    else:
        direction_accuracy = float("nan")

    return {
        "rmse_z": rmse,
        "persistence_rmse_z": persistence_rmse,
        "improvement_vs_persistence_pct": 100.0 * (persistence_rmse - rmse) / max(persistence_rmse, 1e-12),
        "diff_rmse_z": diff_rmse,
        "diff_corr": _safe_corr(dp, dy),
        "direction_accuracy": direction_accuracy,
        "direction_threshold_z": threshold,
        "endpoint_rmse_z": endpoint_rmse,
        "samples": int(p.shape[0]),
    }


def _selection_metric(metrics: dict) -> float:
    # RMSE remains dominant; difference RMSE is the trend-aware tie breaker.
    return float(metrics["rmse_z"] + 0.35 * metrics["diff_rmse_z"])


def save_checkpoint(model, config, epoch: int, val: dict, args, note: str) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "targets": TARGETS,
            "active_targets": [TEMP_NAME],
            "epoch": int(epoch),
            "val_metrics": val,
            "base_checkpoint": str(V1_CHECKPOINT.name),
            "training": {
                "diff_weight": float(args.diff_weight),
                "direction_weight": float(args.direction_weight),
                "endpoint_weight": float(args.endpoint_weight),
                "direction_scale": float(args.direction_scale),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            },
            "normalization": "same normalized source corpus as patchtst_v1_pretrain.pt",
            "note": note,
        },
        V82_CHECKPOINT,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--diff-weight", type=float, default=0.50)
    parser.add_argument("--direction-weight", type=float, default=0.12)
    parser.add_argument("--endpoint-weight", type=float, default=0.08)
    parser.add_argument("--direction-scale", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    for path in (CORPUS_PATH, V1_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"missing required file: {path}")

    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )

    data = np.load(CORPUS_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32, copy=False)
    Y = data["Y"].astype(np.float32, copy=False)
    mask = data["mask"].astype(np.float32, copy=False)
    split = data["split"].astype(str)
    source = data["source"].astype(str)
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

    base = torch.load(V1_CHECKPOINT, map_location=device, weights_only=False)
    config = config_from_checkpoint(base)
    model = MaskedPatchTSTForecaster(config).to(device)
    model.load_state_dict(base["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    initial_val = evaluate_temperature(model, val_loader, device)
    initial_test = evaluate_temperature(model, test_loader, device)
    best_metric = _selection_metric(initial_val)
    best_epoch = 0
    bad_epochs = 0
    save_checkpoint(
        model,
        config,
        0,
        initial_val,
        args,
        "V8.2 initial checkpoint equals frozen V1 weights; online inference is not modified",
    )

    print("=" * 104)
    print("TRAIN PATCHTST V8.2 - TEMPERATURE TREND FINE-TUNE")
    print("=" * 104)
    print(f"device             : {device}")
    if device.type == "cuda":
        print(f"gpu                : {torch.cuda.get_device_name(device)}")
    print(f"source checkpoint  : {V1_CHECKPOINT}")
    print(f"output checkpoint  : {V82_CHECKPOINT}")
    print(f"train/val/test     : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"source counts      : {source_counts}")
    print(f"target             : {TEMP_NAME} only")
    print(
        f"loss               : level + {args.diff_weight}*diff + "
        f"{args.direction_weight}*direction + {args.endpoint_weight}*endpoint"
    )
    print(
        f"initial val        : rmse={initial_val['rmse_z']:.6f} "
        f"diff={initial_val['diff_rmse_z']:.6f} "
        f"corr={initial_val['diff_corr']:+.4f} dir={initial_val['direction_accuracy']:.4f}"
    )

    history: list[dict] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "level": 0.0, "diff": 0.0, "direction": 0.0, "endpoint": 0.0}
        batches = 0

        for x, y, m in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            m = m.to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, m)
            loss, parts = temperature_trend_loss(
                pred,
                y,
                m,
                diff_weight=args.diff_weight,
                direction_weight=args.direction_weight,
                endpoint_weight=args.endpoint_weight,
                direction_scale=args.direction_scale,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            for key in ("level", "diff", "direction", "endpoint"):
                totals[key] += float(parts[key].detach().cpu())
            batches += 1

        val = evaluate_temperature(model, val_loader, device)
        selection = _selection_metric(val)
        scheduler.step(selection)
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            **{f"train_{k}": v / max(1, batches) for k, v in totals.items()},
            "val": val,
            "selection_metric": selection,
            "lr": lr,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | loss={row['train_loss']:.5f} "
            f"| val_rmse={val['rmse_z']:.5f} diff={val['diff_rmse_z']:.5f} "
            f"corr={val['diff_corr']:+.4f} dir={val['direction_accuracy']:.4f} "
            f"| select={selection:.5f} lr={lr:.2e}"
        )

        if selection < best_metric - 1e-5:
            best_metric = selection
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                model,
                config,
                epoch,
                val,
                args,
                "temperature-only trend-aware source-domain fine-tune; not activated online",
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    best = torch.load(V82_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    final_val = evaluate_temperature(model, val_loader, device)
    final_test = evaluate_temperature(model, test_loader, device)
    elapsed = time.time() - started

    # This is only a source-domain sanity gate. Official five-sequence LOSO is the
    # decisive gate and is performed by evaluate_patchtst_v82_temperature_candidate.py.
    source_gate = bool(
        final_val["rmse_z"] <= initial_val["rmse_z"] * 1.02
        and final_val["diff_rmse_z"] <= initial_val["diff_rmse_z"] * 1.01
        and (
            not math.isfinite(initial_val["direction_accuracy"])
            or not math.isfinite(final_val["direction_accuracy"])
            or final_val["direction_accuracy"] >= initial_val["direction_accuracy"] - 0.005
        )
    )

    result = {
        "model": "patchtst_v82_temperature_trend_finetune",
        "base_checkpoint": str(V1_CHECKPOINT),
        "checkpoint": str(V82_CHECKPOINT),
        "best_epoch": int(best_epoch),
        "elapsed_seconds": float(elapsed),
        "config": asdict(config),
        "target": TEMP_NAME,
        "loss": {
            "diff_weight": args.diff_weight,
            "direction_weight": args.direction_weight,
            "endpoint_weight": args.endpoint_weight,
            "direction_scale": args.direction_scale,
        },
        "initial_val": initial_val,
        "initial_test": initial_test,
        "final_val": final_val,
        "final_test": final_test,
        "source_counts_train": source_counts,
        "source_domain_gate_pass": source_gate,
        "history": history,
        "important_note": "Does not modify V8/V8.1 online API, callback, or ensemble_config.pkl.",
    }
    V82_METRICS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 104)
    print("PATCHTST V8.2 SOURCE RESULT")
    print("=" * 104)
    print(f"best epoch          : {best_epoch}")
    print(f"elapsed             : {elapsed:.1f}s")
    print(
        f"VAL V1 -> V8.2      : rmse {initial_val['rmse_z']:.6f} -> {final_val['rmse_z']:.6f} | "
        f"diff {initial_val['diff_rmse_z']:.6f} -> {final_val['diff_rmse_z']:.6f} | "
        f"corr {initial_val['diff_corr']:+.4f} -> {final_val['diff_corr']:+.4f} | "
        f"dir {initial_val['direction_accuracy']:.4f} -> {final_val['direction_accuracy']:.4f}"
    )
    print(
        f"TEST V1 -> V8.2     : rmse {initial_test['rmse_z']:.6f} -> {final_test['rmse_z']:.6f} | "
        f"diff {initial_test['diff_rmse_z']:.6f} -> {final_test['diff_rmse_z']:.6f} | "
        f"corr {initial_test['diff_corr']:+.4f} -> {final_test['diff_corr']:+.4f} | "
        f"dir {initial_test['direction_accuracy']:.4f} -> {final_test['direction_accuracy']:.4f}"
    )
    print(f"SOURCE GATE          : {'PASS' if source_gate else 'REJECT'}")
    print(f"checkpoint           : {V82_CHECKPOINT}")
    print(f"metrics              : {V82_METRICS}")
    print("NEXT: run official five-sequence LOSO before any online activation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
