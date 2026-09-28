import torch
from tqdm import tqdm

from .targets import prepare_yolo_targets


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
  model.train()

  total_loss = 0.0
  total_box = 0.0
  total_cls = 0.0
  total_dfl = 0.0

  progress_bar = tqdm(loader, desc=f"Train {epoch:03d}", leave=False, dynamic_ncols=True)

  for rgb, depth, targets in progress_bar:
    rgb = rgb.to(device, non_blocking=True)
    depth = depth.to(device, non_blocking=True)

    yolo_targets = prepare_yolo_targets(targets, device)

    optimizer.zero_grad(set_to_none=True)

    predictions = model(rgb, depth)
    loss, loss_items = criterion(predictions, yolo_targets)
    loss = loss.sum()

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total_box += loss_items[0].item()
    total_cls += loss_items[1].item()
    total_dfl += loss_items[2].item()

    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

  num_batches = len(loader)

  return {
    "loss": total_loss / num_batches,
    "loss_box": total_box / num_batches,
    "loss_cls": total_cls / num_batches,
    "loss_dfl": total_dfl / num_batches
  }