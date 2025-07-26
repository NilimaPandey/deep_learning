import torch
import torch.nn as nn
import torch.optim as optim
from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to your data
train_dir = "drive_data/train"
val_dir = "drive_data/val"

# Data loaders
train_loader = load_data(train_dir, transform_pipeline="default", batch_size=32, shuffle=True)
val_loader = load_data(val_dir, transform_pipeline="default", batch_size=32, shuffle=False)

# Model, loss, optimizer
model = Detector().to(device)
seg_criterion = nn.CrossEntropyLoss()
depth_criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
best_val_iou = 0.0

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch["image"].to(device)
        seg_labels = batch["track"].to(device)      # (B, H, W)
        depth_labels = batch["depth"].to(device)    # (B, H, W)

        optimizer.zero_grad()
        seg_logits, depth_preds = model(images)     # (B, 3, H, W), (B, H, W)
        seg_loss = seg_criterion(seg_logits, seg_labels)
        depth_loss = depth_criterion(depth_preds, depth_labels)
        loss = seg_loss + depth_loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_metric = DetectionMetric(num_classes=3)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            seg_labels = batch["track"].to(device)
            depth_labels = batch["depth"].to(device)
            seg_logits, depth_preds = model(images)
            preds = seg_logits.argmax(dim=1)
            val_metric.add(preds, seg_labels, depth_preds, depth_labels)
    val_results = val_metric.compute()
    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Val IoU: {val_results['iou']:.4f} | "
        f"Val Depth Error: {val_results['abs_depth_error']:.4f} | "
        f"Val Lane Depth Error: {val_results['tp_depth_error']:.4f}"
    )

    # Save best model
    if val_results["iou"] > best_val_iou:
        best_val_iou = val_results["iou"]
        save_model(model)
        print("Saved new best model!")

print("Training complete. Best Val IoU:", best_val_iou)