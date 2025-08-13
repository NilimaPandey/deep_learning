from __future__ import annotations
import argparse

import torch
from torch.utils.data import DataLoader

from .models import build_model, save_model
from .metrics import PlannerMetric
from .datasets.road_dataset import RoadDataset


def collate_batch(batch):
    """Convert list[dict] -> dict[str, tensor/list] for DataLoader."""
    out = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if torch.is_tensor(vals[0]):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out


def make_loader(split_path: str, transform_pipeline: str, batch_size: int, num_workers: int, shuffle: bool):
    ds = RoadDataset(split_path, transform_pipeline=transform_pipeline)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )


# add near the top
import torch.nn.functional as F

def masked_smooth_l1_loss(preds, targets, mask, beta: float = 1.0, lat_weight: float = 2.0):
    """
    Weighted Smooth L1 (Huber) over valid waypoints.
    lat_weight > 1.0 puts extra emphasis on lateral (y) error.
    """
    diff = preds - targets                     # (B, n, 2)
    mask2 = mask.unsqueeze(-1).to(diff.dtype)  # (B, n, 1)
    diff = diff * mask2

    loss_x = F.smooth_l1_loss(diff[..., 0], torch.zeros_like(diff[..., 0]),
                              beta=beta, reduction="sum")
    loss_y = F.smooth_l1_loss(diff[..., 1], torch.zeros_like(diff[..., 1]),
                              beta=beta, reduction="sum")
    denom = mask2.sum().clamp_min(1.0)
    return (loss_x + lat_weight * loss_y) / denom



def train_one_epoch(model, loader, optimizer, device):
    model.train()
    metric = PlannerMetric()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        # Choose inputs by model type
        if model.__class__.__name__ == "CNNPlanner":
            images = batch["image"].to(device, non_blocking=True)  # (B, 3, H, W)
            preds = model(images)
        else:
            tl = batch["track_left"].to(device, non_blocking=True)
            tr = batch["track_right"].to(device, non_blocking=True)
            preds = model(tl, tr)

        targets = batch["waypoints"].to(device, non_blocking=True)
        mask = batch["waypoints_mask"].to(device, non_blocking=True).bool()

        loss = masked_smooth_l1_loss(preds, targets, mask, beta=1.0, lat_weight=2.0)
   # in train_one_epoch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item())
        metric.update(preds, targets, mask)

    stats = metric.compute()
    stats["loss"] = total_loss / max(1, len(loader))
    return stats


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = PlannerMetric()
    total_loss = 0.0

    for batch in loader:
        if model.__class__.__name__ == "CNNPlanner":
            images = batch["image"].to(device, non_blocking=True)
            preds = model(images)
        else:
            tl = batch["track_left"].to(device, non_blocking=True)
            tr = batch["track_right"].to(device, non_blocking=True)
            preds = model(tl, tr)

        targets = batch["waypoints"].to(device, non_blocking=True)
        mask = batch["waypoints_mask"].to(device, non_blocking=True).bool()

        loss = masked_smooth_l1_loss(preds, targets, mask, beta=1.0, lat_weight=2.0)
        total_loss += float(loss.item())
        metric.update(preds, targets, mask)

    stats = metric.compute()
    stats["loss"] = total_loss / max(1, len(loader))
    return stats


def train(
    model_name: str,
    transform_pipeline: str | None = None,
    num_workers: int = 2,
    lr: float = 1e-3,
    batch_size: int = 128,
    num_epoch: int = 20,
    train_split: str = "drive_data/train",
    val_split: str = "drive_data/val",
    save: bool = True,
):
    """
    Programmatic training entrypoint (for Colab).
    Returns the final validation stats dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)

    # Default: state_only for state models, image "default" for CNN
    if transform_pipeline is None:
        transform_pipeline = "default" if model_name == "cnn_planner" else "state_only"

    train_loader = make_loader(train_split, transform_pipeline, batch_size, num_workers, shuffle=True)
    val_loader = make_loader(val_split, transform_pipeline, batch_size, num_workers, shuffle=False)

    # CHANGED: add weight decay + cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)


    best_l1 = float("inf")
    best_path = None
    for epoch in range(1, num_epoch + 1):
        tr_stats = train_one_epoch(model, train_loader, optimizer, device)
        va_stats = evaluate(model, val_loader, device)

        scheduler.step()  # CHANGED: step the LR schedule

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr_stats['loss']:.4f} L1 {tr_stats['l1_error']:.3f} "
            f"(long {tr_stats['longitudinal_error']:.3f} lat {tr_stats['lateral_error']:.3f}) | "
            f"val loss {va_stats['loss']:.4f} L1 {va_stats['l1_error']:.3f} "
            f"(long {va_stats['longitudinal_error']:.3f} lat {va_stats['lateral_error']:.3f}) | "
            f"lr {cur_lr:.2e}"
        )

        if save and va_stats["l1_error"] < best_l1:
            best_l1 = va_stats["l1_error"]
            best_path = save_model(model)
            print(f"Saved new best model to {best_path} (L1={best_l1:.3f})")

    if save and best_path is None:
        best_path = save_model(model)
        print(f"Saved model to {best_path}")

    return va_stats



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["linear_planner", "mlp_planner", "transformer_planner", "cnn_planner"], required=True)
    p.add_argument("--train_split", type=str, default="drive_data/train")
    p.add_argument("--val_split", type=str, default="drive_data/val")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--transform", type=str, choices=["default", "state_only"], default=None)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    transform_pipeline = args.transform if args.transform is not None else ("default" if args.model == "cnn_planner" else "state_only")

    train_loader = make_loader(args.train_split, transform_pipeline, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(args.val_split, transform_pipeline, args.batch_size, args.num_workers, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_l1 = float("inf")
    best_path = None
    for epoch in range(1, args.epochs + 1):
        tr_stats = train_one_epoch(model, train_loader, optimizer, device)
        va_stats = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr_stats['loss']:.4f} L1 {tr_stats['l1_error']:.3f} "
            f"(long {tr_stats['longitudinal_error']:.3f} lat {tr_stats['lateral_error']:.3f}) | "
            f"val loss {va_stats['loss']:.4f} L1 {va_stats['l1_error']:.3f} "
            f"(long {va_stats['longitudinal_error']:.3f} lat {va_stats['lateral_error']:.3f})"
        )

        if args.save and va_stats["l1_error"] < best_l1:
            best_l1 = va_stats["l1_error"]
            best_path = save_model(model)
            print(f"Saved new best model to {best_path} (L1={best_l1:.3f})")

    if args.save and best_path is None:
        best_path = save_model(model)
        print(f"Saved model to {best_path}")


if __name__ == "__main__":
    main()
