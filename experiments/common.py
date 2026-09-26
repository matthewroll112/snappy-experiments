from pathlib import Path

import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import SnappyDataset, collate_fn, get_train_transforms, get_val_transforms


def set_seed(seed):
  """Set random seeds for reproducible experiments."""

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(data_dir, batch_size, num_workers, image_size=(640, 640)):
  """Create training and validation dataloaders."""

  train_dataset = SnappyDataset(root=data_dir, split="train", transform=get_train_transforms(image_size))
  val_dataset = SnappyDataset(root=data_dir, split="val", transform=get_val_transforms(image_size))

  train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers, 
    pin_memory=True,
    collate_fn=collate_fn
  )
  val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    pin_memory=True, 
    collate_fn=collate_fn
  )

  return train_loader, val_loader


def create_optimizer(model, learning_rate, weight_decay=0.0):
  """Create the optimizer used for an experiment."""

  return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_scaler(device):
  """Create an AMP gradient scaler."""

  return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


def create_csv(path, train_metric_names, val_metric_names):
  """Create the experiment history CSV."""

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  fieldnames = ["epoch", "lr", *train_metric_names, *val_metric_names]

  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()


def append_csv(path, epoch, learning_rate, train_metrics, val_metrics):
  """Append one epoch to the experiment history CSV."""

  row = {
    "epoch": epoch,
    "lr": learning_rate,
    **train_metrics,
    **val_metrics
  }

  with Path(path).open("a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=row.keys())
    writer.writerow(row)


def save_checkpoint(path, model, optimizer, scaler, epoch, train_metrics, val_metrics):
  """Save the current best checkpoint."""

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "train_metrics": train_metrics,
    "val_metrics": val_metrics
  }

  torch.save(checkpoint, path)


def run_training(model, criterion, train_loader, val_loader, train_fn, val_fn, optimizer, scaler, device, class_names, epochs, output_dir, experiment_name):
  """Run a complete object detection experiment."""

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  csv_path = output_dir / f"{experiment_name}.csv"
  checkpoint_path = output_dir / f"{experiment_name}_best.pth"

  best_map = -1.0
  train_metric_names = None
  val_metric_names = None

  for epoch in range(1, epochs + 1):
    train_metrics = train_fn(model, train_loader, criterion, optimizer, scaler, device, epoch)
    val_metrics = val_fn(model, val_loader, device, class_names, epoch)

    learning_rate = optimizer.param_groups[0]["lr"]

    if train_metric_names is None:
      train_metric_names = list(train_metrics.keys())
      val_metric_names = list(val_metrics.keys())
      create_csv(csv_path, train_metric_names, val_metric_names)

    append_csv(csv_path, epoch, learning_rate, train_metrics, val_metrics)

    current_map = val_metrics["map"]

    if current_map > best_map:
      best_map = current_map
      save_checkpoint(checkpoint_path, model, optimizer, scaler, epoch, train_metrics, val_metrics)

    train_summary = " | ".join(f"{name}: {value:.4f}" for name, value in train_metrics.items())
    val_summary = " | ".join(f"{name}: {value:.4f}" for name, value in val_metrics.items())

    print(f"Epoch {epoch:03d}/{epochs:03d} | LR: {learning_rate:.6f} | {train_summary} | {val_summary} | Best mAP: {best_map:.4f}")