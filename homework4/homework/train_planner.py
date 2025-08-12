
"""Train planners for Homework 4.
Usage examples:
  python3 -m homework.train_planner --model mlp_planner --epochs 40 --batch_size 128 --lr 1e-3 --save
  python3 -m homework.train_planner --model transformer_planner --epochs 60 --batch_size 128 --lr 3e-4 --save
  python3 -m homework.train_planner --model cnn_planner --epochs 40 --batch_size 64 --lr 1e-3 --save
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from .models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model, calculate_model_size_mb
from .metrics import PlannerMetric
from .datasets.road_dataset import RoadDataset
from .datasets.road_transforms import EgoTrackProcessor, NormalizeImage, ToTensor


def get_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "mlp_planner":
        return MLPPlanner()
    if name == "transformer_planner":
        return TransformerPlanner()
    if name == "cnn_planner":
        return CNNPlanner()
    raise ValueError(f"Unknown model '{name}'")


def get_loss() -> nn.Module:
    # Waypoints are real-valued -> L1 works well for coordinates
    return nn.L1Loss(reduction="none")


def collate_tracks(batch):
    # Each item is a dict from RoadDataset
    # We'll return what each model needs in a consistent structure.
    out = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = [b[k] for b in batch]

    def stack_if_tensor(key):
        if torch.is_tensor(out[key][0]):
            out[key] = torch.stack(out[key], dim=0)

    for k in ["track_left", "track_right", "waypoints", "waypoints_mask"]:
        stack_if_tensor(k)

    # Images (for CNN) if provided by dataset transform
    if "image" in out and torch.is_tensor(out["image"][0]):
        out["image"] = torch.stack(out["image"], dim=0)

    return out


def build_loaders(train_split: str, val_split: str, model_name: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    # Transforms: leverage the dataset's helper to build correct egocentric tracks
    # For CNN, also include image normalization
    track_tf = EgoTrackProcessor()
    img_tf = [ToTensor(), NormalizeImage()]

    train_ds = RoadDataset(split=train_split, track_transform=track_tf, image_transform=img_tf)
    val_ds = RoadDataset(split=val_split, track_transform=track_tf, image_transform=img_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_tracks, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_tracks, pin_memory=True)
    return train_loader, val_loader


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device, model_name: str) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    metric = PlannerMetric()

    for batch in loader:
        # Move tensors to device
        for k in ["track_left", "track_right", "waypoints", "waypoints_mask", "image"]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        # Forward based on model type
        if isinstance(model, CNNPlanner):
            preds = model(batch["image"])  # (B, n_waypoints, 2)
        else:
            preds = model(batch["track_left"], batch["track_right"])

        labels = batch["waypoints"]
        mask = batch["waypoints_mask"].bool()

        # Loss (element-wise), then mask and average
        loss_elems = loss_fn(preds, labels)  # (B, n_wp, 2)
        loss_mask = mask.unsqueeze(-1).expand_as(loss_elems)
        loss = loss_elems[loss_mask].mean() if loss_mask.any() else loss_elems.mean()

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        metric.update(preds.detach(), labels.detach(), mask.detach())

    return metric.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--train_split", type=str, default="drive_data/train")
    parser.add_argument("--val_split", type=str, default="drive_data/val")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model).to(device)

    loss_fn = get_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, val_loader = build_loaders(args.train_split, args.val_split, args.model, args.batch_size, args.num_workers)

    best_val = float("inf")
    best_stats = None
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, loss_fn, optimizer, device, args.model)
        val_stats = run_epoch(model, val_loader, loss_fn, None, device, args.model)

        msg = (f"Epoch {epoch:03d} | Train L1: {train_stats['l1_error']:.4f} "
               f"(Long {train_stats['longitudinal_error']:.4f}, Lat {train_stats['lateral_error']:.4f}) | "
               f"Val L1: {val_stats['l1_error']:.4f} "
               f"(Long {val_stats['longitudinal_error']:.4f}, Lat {val_stats['lateral_error']:.4f})")
        print(msg)

        if val_stats['l1_error'] < best_val:
            best_val = val_stats['l1_error']
            best_stats = val_stats
            if args.save:
                path = save_model(model)
                print(f"Saved best model -> {path}")

    print("\nTraining complete.")
    print(f"Best Val -> L1: {best_val:.4f}; details: {best_stats}")
    print(f"Model size ~ {calculate_model_size_mb(model):.2f} MB")


if __name__ == "__main__":
    main()
