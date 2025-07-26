import torch
import torch.nn as nn
import torch.optim as optim
from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to your data
train_dir = "classification_data/train"
val_dir = "classification_data/val"

# Data loaders
train_loader = load_data(train_dir, transform_pipeline="aug", batch_size=128, shuffle=True)
val_loader = load_data(val_dir, transform_pipeline="default", batch_size=128, shuffle=False)

# Model, loss, optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_metric = AccuracyMetric()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_metric.add(logits.argmax(dim=1), labels)
    train_acc = train_metric.compute()["accuracy"]

    model.eval()
    val_metric = AccuracyMetric()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            val_metric.add(logits.argmax(dim=1), labels)
    val_acc = val_metric.compute()["accuracy"]

    print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_model(model)
        print("Saved new best model!")

print("Training complete. Best Val Acc:", best_val_acc)