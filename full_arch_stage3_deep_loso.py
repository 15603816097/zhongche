from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR, HORIZON, TARGET_COLUMNS
from src.data_cleaner import clean_sequence
from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster, PatchTSTConfig
from src.deep.tcn_forecaster import MaskedTCNForecaster, TCNConfig
from v9_analog_multiscale_diagnostic import evaluate_rows

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "artifacts" / "full_arch" / "expert_cache"
OUT_DIR = ROOT / "artifacts" / "full_arch" / "deep_loso"
MODEL_DIR = ROOT / "models" / "full_arch"
SEQUENCES = [f"sequence{i:04d}" for i in range(1, 6)]
N_TARGETS = len(TARGET_COLUMNS)
EPS = 1e-12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_clean_history(name: str) -> np.ndarray:
    path = DATA_DIR / name / "history.csv"
    df = pd.read_csv(path)
    df = clean_sequence(df)
    return df[TARGET_COLUMNS].to_numpy(dtype=np.float32)


def load_future(name: str) -> np.ndarray:
    path = DATA_DIR / name / "future.csv"
    df = pd.read_csv(path)
    arr = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)[:HORIZON]
    if arr.shape != (HORIZON, N_TARGETS):
        raise RuntimeError(f"{name}: bad future shape {arr.shape}")
    return arr


def load_v8(name: str) -> np.ndarray:
    path = CACHE_DIR / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing stage2 cache: {path}")
    with np.load(path) as z:
        arr = np.asarray(z["v8"], dtype=np.float32)
    if arr.shape != (HORIZON, N_TARGETS):
        raise RuntimeError(f"{name}: bad v8 cache shape {arr.shape}")
    return arr


class RollingDataset(Dataset):
    def __init__(
        self,
        histories: list[np.ndarray],
        input_length: int,
        horizon: int,
        stride: int,
    ) -> None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for hist in histories:
            last_start = len(hist) - input_length - horizon
            if last_start < 0:
                continue
            starts = list(range(0, last_start + 1, stride))
            if starts[-1] != last_start:
                starts.append(last_start)
            for s in starts:
                x = hist[s : s + input_length].astype(np.float32, copy=True)
                y = hist[s + input_length : s + input_length + horizon].astype(
                    np.float32, copy=True
                )
                mean = x.mean(axis=0, keepdims=True)
                std = x.std(axis=0, keepdims=True)
                std = np.maximum(std, 1e-4)
                xs.append((x - mean) / std)
                ys.append((y - mean) / std)

        if not xs:
            raise RuntimeError("rolling dataset is empty")
        self.x = np.stack(xs, axis=0)
        self.y = np.stack(ys, axis=0)
        self.mask = np.ones((len(xs), N_TARGETS), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.mask[idx]),
        )


def trend_aware_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    level = torch.mean((pred - truth) ** 2)
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dy = truth[:, 1:, :] - truth[:, :-1, :]
    diff = torch.mean((dp - dy) ** 2)
    endpoint = torch.mean((pred[:, -1, :] - truth[:, -1, :]) ** 2)

    importance = torch.clamp(torch.abs(dy) / 0.05, 0.0, 1.0)
    same_sign = torch.tanh(4.0 * dp) * torch.tanh(4.0 * dy)
    direction = ((1.0 - same_sign) * importance).sum() / torch.clamp(
        importance.sum(), min=1.0
    )
    return level + 0.20 * diff + 0.08 * endpoint + 0.03 * direction


def make_model(kind: str, input_length: int) -> nn.Module:
    if kind == "tcn":
        return MaskedTCNForecaster(
            TCNConfig(
                output_channels=N_TARGETS,
                horizon=HORIZON,
                hidden_channels=96,
                num_blocks=5,
                kernel_size=3,
                dropout=0.10,
            )
        )
    if kind == "patchtst":
        return MaskedPatchTSTForecaster(
            PatchTSTConfig(
                input_length=input_length,
                output_channels=N_TARGETS,
                horizon=HORIZON,
                patch_length=24,
                stride=12,
                d_model=96,
                n_heads=4,
                num_layers=3,
                dim_feedforward=192,
                dropout=0.10,
            )
        )
    raise ValueError(kind)


def train_model(
    kind: str,
    train_histories: list[np.ndarray],
    *,
    input_length: int,
    stride: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[float], int]:
    seed_everything(seed)
    ds = RollingDataset(train_histories, input_length, HORIZON, stride)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
        drop_last=False,
    )

    model = make_model(kind, input_length).to(device)
    lr = 1.2e-3 if kind == "tcn" else 8e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs, 1), eta_min=lr * 0.08
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x, mask)
                loss = trend_aware_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()
        losses.append(total / max(seen, 1))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"      {kind:8s} epoch {epoch:02d}/{epochs} "
                f"loss={losses[-1]:.6f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    return model, losses, len(ds)


@torch.no_grad()
def infer_model(
    model: nn.Module,
    history: np.ndarray,
    *,
    input_length: int,
    device: torch.device,
) -> np.ndarray:
    x_raw = history[-input_length:].astype(np.float32, copy=True)
    mean = x_raw.mean(axis=0, keepdims=True)
    std = np.maximum(x_raw.std(axis=0, keepdims=True), 1e-4)
    x = (x_raw - mean) / std

    xt = torch.from_numpy(x[None]).to(device)
    mask = torch.ones((1, N_TARGETS), dtype=torch.float32, device=device)
    model.eval()
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        pred_n = model(xt, mask)
    pred_n = pred_n.float().cpu().numpy()[0]
    pred = pred_n * std + mean
    return np.asarray(pred, dtype=np.float64)


def eval_one_target(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    anchor: float,
) -> dict:
    b = evaluate_rows(
        truth.reshape(1, -1),
        base.reshape(1, -1),
        np.asarray([anchor], dtype=np.float64),
    )
    c = evaluate_rows(
        truth.reshape(1, -1),
        candidate.reshape(1, -1),
        np.asarray([anchor], dtype=np.float64),
    )
    rmse_ratio = float(c["rmse"] / max(b["rmse"], EPS))
    proxy_gain = float(
        (b["proxy_loss"] - c["proxy_loss"]) / max(abs(b["proxy_loss"]), EPS)
    )
    trend_gain = float(c["trend_core"] - b["trend_core"])
    return {
        "rmse_ratio": rmse_ratio,
        "proxy_gain": proxy_gain,
        "trend_gain": trend_gain,
        "candidate_rmse": float(c["rmse"]),
        "base_rmse": float(b["rmse"]),
        "candidate_proxy": float(c["proxy_loss"]),
        "base_proxy": float(b["proxy_loss"]),
        "candidate_trend": float(c["trend_core"]),
        "base_trend": float(b["trend_core"]),
    }


def pooled_summary(
    all_truth: np.ndarray,
    all_v8: np.ndarray,
    all_pred: np.ndarray,
    anchors: np.ndarray,
) -> dict:
    by_target = {}
    for j, target in enumerate(TARGET_COLUMNS):
        fold_rows = []
        for i, name in enumerate(SEQUENCES):
            row = eval_one_target(
                all_truth[i, :, j],
                all_v8[i, :, j],
                all_pred[i, :, j],
                float(anchors[i, j]),
            )
            row["sequence"] = name
            fold_rows.append(row)

        b = evaluate_rows(
            all_truth[:, :, j],
            all_v8[:, :, j],
            anchors[:, j],
        )
        c = evaluate_rows(
            all_truth[:, :, j],
            all_pred[:, :, j],
            anchors[:, j],
        )
        by_target[target] = {
            "positive_proxy_folds": int(sum(r["proxy_gain"] > 0 for r in fold_rows)),
            "nonnegative_trend_folds": int(sum(r["trend_gain"] >= 0 for r in fold_rows)),
            "max_rmse_ratio": float(max(r["rmse_ratio"] for r in fold_rows)),
            "pooled_rmse_ratio": float(c["rmse"] / max(b["rmse"], EPS)),
            "pooled_proxy_gain": float(
                (b["proxy_loss"] - c["proxy_loss"])
                / max(abs(b["proxy_loss"]), EPS)
            ),
            "pooled_trend_gain": float(c["trend_core"] - b["trend_core"]),
            "folds": fold_rows,
        }
    flat_v8 = float(np.sqrt(np.mean((all_truth - all_v8) ** 2)))
    flat_pred = float(np.sqrt(np.mean((all_truth - all_pred) ** 2)))
    return {
        "flat_rmse_v8": flat_v8,
        "flat_rmse_candidate": flat_pred,
        "flat_rmse_ratio": flat_pred / max(flat_v8, EPS),
        "targets": by_target,
    }


def save_checkpoint(
    path: Path,
    kind: str,
    model: nn.Module,
    input_length: int,
    training_meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model.config)
    torch.save(
        {
            "kind": kind,
            "input_length": input_length,
            "config": cfg,
            "state_dict": model.state_dict(),
            "training_meta": training_meta,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-length", type=int, default=144)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not CACHE_DIR.is_dir():
        raise FileNotFoundError("stage2 cache missing; run stage2 first")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 3 requires CUDA; torch.cuda.is_available() is False")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    histories = {name: load_clean_history(name) for name in SEQUENCES}
    futures = {name: load_future(name) for name in SEQUENCES}
    v8 = {name: load_v8(name) for name in SEQUENCES}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("FULL ARCHITECTURE - STAGE 3 DEEP EXPERT LOSO")
    print("=" * 118)
    print("device       :", torch.cuda.get_device_name(0))
    print("input length :", args.input_length)
    print("horizon      :", HORIZON)
    print("stride       :", args.stride)
    print("epochs/fold  :", args.epochs)
    print("final epochs :", args.final_epochs)
    print("batch size   :", args.batch_size)
    print("data policy  : train from official history.csv only; future.csv is evaluation only")
    print("LOSO policy  : holdout sequence history is excluded from that fold's training")
    print()

    predictions: dict[str, list[np.ndarray]] = {"tcn": [], "patchtst": []}
    all_truth = []
    all_v8 = []
    anchors = []
    fold_manifest: list[dict] = []

    for holdout_idx, holdout_name in enumerate(SEQUENCES):
        train_names = [n for n in SEQUENCES if n != holdout_name]
        train_histories = [histories[n] for n in train_names]
        holdout_history = histories[holdout_name]
        truth = futures[holdout_name].astype(np.float64)
        base = v8[holdout_name].astype(np.float64)
        anchor = holdout_history[-1].astype(np.float64)

        print("-" * 118)
        print(
            f"FOLD {holdout_idx+1}/5 holdout={holdout_name} "
            f"train={train_names}"
        )
        print("-" * 118)

        fold_info = {
            "holdout": holdout_name,
            "train_sequences": train_names,
            "models": {},
        }

        for kind_idx, kind in enumerate(("tcn", "patchtst")):
            started = time.perf_counter()
            model, losses, n_samples = train_model(
                kind,
                train_histories,
                input_length=args.input_length,
                stride=args.stride,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=4200 + holdout_idx * 10 + kind_idx,
                device=device,
            )
            pred = infer_model(
                model,
                holdout_history,
                input_length=args.input_length,
                device=device,
            )
            elapsed = time.perf_counter() - started
            predictions[kind].append(pred)

            per_target = {}
            for j, target in enumerate(TARGET_COLUMNS):
                per_target[target] = eval_one_target(
                    truth[:, j], base[:, j], pred[:, j], float(anchor[j])
                )

            fold_path = OUT_DIR / f"{kind}_{holdout_name}.npz"
            np.savez_compressed(
                fold_path,
                prediction=pred,
                truth=truth,
                v8=base,
                anchor=anchor,
            )
            fold_info["models"][kind] = {
                "train_samples": n_samples,
                "seconds": elapsed,
                "final_train_loss": losses[-1],
                "prediction_file": str(fold_path.relative_to(ROOT)),
                "targets": per_target,
            }

            mean_proxy = float(
                np.mean([v["proxy_gain"] for v in per_target.values()])
            )
            print(
                f"  {kind:8s}: samples={n_samples} time={elapsed:.1f}s "
                f"loss={losses[-1]:.5f} mean_proxy_gain={100*mean_proxy:+.2f}%"
            )
            for target in TARGET_COLUMNS:
                m = per_target[target]
                print(
                    f"    {target:15s} rmse_ratio={m['rmse_ratio']:.4f} "
                    f"proxy={100*m['proxy_gain']:+6.2f}% "
                    f"trend={m['trend_gain']:+.4f}"
                )
            del model
            torch.cuda.empty_cache()

        fold_manifest.append(fold_info)
        all_truth.append(truth)
        all_v8.append(base)
        anchors.append(anchor)

    truth_arr = np.stack(all_truth, axis=0)
    v8_arr = np.stack(all_v8, axis=0)
    anchor_arr = np.stack(anchors, axis=0)

    summary = {}
    print("\n" + "=" * 118)
    print("LOSO SUMMARY")
    print("=" * 118)
    for kind in ("tcn", "patchtst"):
        pred_arr = np.stack(predictions[kind], axis=0)
        s = pooled_summary(truth_arr, v8_arr, pred_arr, anchor_arr)
        summary[kind] = s
        print(
            f"\n{kind.upper()} flat_rmse_ratio={s['flat_rmse_ratio']:.6f} "
            f"candidate={s['flat_rmse_candidate']:.6f} "
            f"V8={s['flat_rmse_v8']:.6f}"
        )
        for target in TARGET_COLUMNS:
            m = s["targets"][target]
            print(
                f"  {target:15s} positive_proxy={m['positive_proxy_folds']}/5 "
                f"nonneg_trend={m['nonnegative_trend_folds']}/5 "
                f"max_rmse={m['max_rmse_ratio']:.4f} "
                f"pooled_rmse={m['pooled_rmse_ratio']:.4f} "
                f"proxy={100*m['pooled_proxy_gain']:+6.2f}% "
                f"trend={m['pooled_trend_gain']:+.4f}"
            )

    # Train production deep experts on all official histories. These are candidates only;
    # later OOD/gating stages decide whether a target/horizon may use them.
    final_models = {}
    print("\n" + "=" * 118)
    print("TRAIN FINAL DEEP EXPERTS ON ALL OFFICIAL HISTORIES")
    print("=" * 118)
    all_histories = [histories[n] for n in SEQUENCES]
    for kind_idx, kind in enumerate(("tcn", "patchtst")):
        started = time.perf_counter()
        model, losses, n_samples = train_model(
            kind,
            all_histories,
            input_length=args.input_length,
            stride=args.stride,
            epochs=args.final_epochs,
            batch_size=args.batch_size,
            seed=9900 + kind_idx,
            device=device,
        )
        ckpt = MODEL_DIR / f"{kind}_official_history.pt"
        meta = {
            "training_sequences": SEQUENCES,
            "data_policy": "official history.csv rolling windows only",
            "input_length": args.input_length,
            "horizon": HORIZON,
            "stride": args.stride,
            "epochs": args.final_epochs,
            "samples": n_samples,
            "final_loss": losses[-1],
        }
        save_checkpoint(ckpt, kind, model, args.input_length, meta)
        final_models[kind] = {
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "samples": n_samples,
            "seconds": time.perf_counter() - started,
            "final_loss": losses[-1],
        }
        print(
            f"{kind:8s}: checkpoint={ckpt} samples={n_samples} "
            f"loss={losses[-1]:.6f}"
        )
        del model
        torch.cuda.empty_cache()

    manifest = {
        "stage": 3,
        "input_length": args.input_length,
        "horizon": HORIZON,
        "stride": args.stride,
        "epochs": args.epochs,
        "final_epochs": args.final_epochs,
        "batch_size": args.batch_size,
        "data_policy": (
            "LOSO fold training uses only history.csv from the other four sequences; "
            "future.csv is used only for offline evaluation."
        ),
        "folds": fold_manifest,
        "summary": summary,
        "final_models": final_models,
        "important_note": (
            "Deep experts are candidates, not replacements for V8. Later confidence/OOD/"
            "gate stages may assign zero weight to unsafe targets or horizons."
        ),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nmanifest:", manifest_path)
    print("STAGE 3 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
