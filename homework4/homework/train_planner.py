"""
Usage:
    python3 -m homework.train_planner \
        --model mlp_planner \
        --train_split drive_data/train \
        --val_split drive_data/val \
        --epochs 20 --batch_size 128 --lr 1e-3

Models:
    - mlp_planner (Part 1a)
    - transformer_planner (Part 1b)
    - cnn_planner (Part 2)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def masked_l1(preds: torch.Tensor, labels: torch.Tensor, labels_mask: torch.Tensor) -> torch.Tensor:
    # preds, labels: (b, n, 2), mask: (b, n)
    error = (preds - labels).abs()
    error = error * labels_mask[..., None]
    denom = labels_mask.sum().clamp_min(1).to(error.dtype)
    return error.sum() / denom


def get_transform_pipeline(model_name: str) -> str:
    if model_name in ("mlp_planner", "transformer_planner"):
        return "state_only"
    elif model_name == "cnn_planner":
        return "default"
    else:
        raise ValueError(f"Unknown model {model_name}")


def forward_for_model(model_name: str, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if model_name in ("mlp_planner", "transformer_planner"):
        return model(batch["track_left"], batch["track_right"])  # (b, n, 2)
    elif model_name == "cnn_planner":
        return model(batch["image"])  # (b, n, 2)
    else:
        raise ValueError(f"Unknown model {model_name}")


def run_epoch(
    model_name: str,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    metric = PlannerMetric()

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            preds = forward_for_model(model_name, model, batch)
            loss = masked_l1(preds, batch["waypoints"], batch["waypoints_mask"])

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.item()) * preds.shape[0]
        metric.add(preds, batch["waypoints"], batch["waypoints_mask"])

        cur = metric.compute()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lon": f"{cur['longitudinal_error']:.3f}", "lat": f"{cur['lateral_error']:.3f}"})

    n_samples = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 1
    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, metric.compute()


def main():
    parser = argparse.ArgumentParser("Train planners for Homework 4")
    parser.add_argument("--model", type=str, default="mlp_planner", choices=["mlp_planner", "transformer_planner", "cnn_planner"]) 
    parser.add_argument("--train_split", type=str, default="drive_data/train")
    parser.add_argument("--val_split", type=str, default="drive_data/val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    transform = get_transform_pipeline(args.model)
    train_loader = load_data(
        args.train_split,
        transform_pipeline=transform,
        return_dataloader=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        args.val_split,
        transform_pipeline=transform,
        return_dataloader=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = load_model(args.model, with_weights=False)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = math.inf
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_metrics = run_epoch(args.model, model, train_loader, optimizer, device, desc="train")
        val_loss, val_metrics = run_epoch(args.model, model, val_loader, None, device, desc="val")

        print(f"  train: loss={train_loss:.4f} lon={train_metrics['longitudinal_error']:.3f} lat={train_metrics['lateral_error']:.3f}")
        print(f"  val  : loss={val_loss:.4f} lon={val_metrics['longitudinal_error']:.3f} lat={val_metrics['lateral_error']:.3f}")

        lr_scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_metrics = val_metrics
            if not args.no_save:
                out_path = save_model(model)
                print(f"  Saved checkpoint to {out_path}")

    if best_metrics is not None:
        print(
            f"Best val: lon={best_metrics['longitudinal_error']:.3f} lat={best_metrics['lateral_error']:.3f} l1={best_metrics['l1_error']:.3f}"
        )


if __name__ == "__main__":
    main()
