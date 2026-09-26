import torch
from tqdm import tqdm

from .targets import prepare_rtdetr_targets


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
  """Train an RT-DETR model for one epoch."""

  model.train()

  total_loss = 0.0
  total_vfl = 0.0
  total_bbox = 0.0
  total_giou = 0.0

  use_amp = device.type == "cuda"

  progress_bar = tqdm(loader, desc=f"Train {epoch:03d}", leave=False, dynamic_ncols=True)

  for rgb, depth, targets in progress_bar:
    rgb = rgb.to(device, non_blocking=True)
    depth = depth.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]

    detr_targets = prepare_rtdetr_targets(targets)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
      outputs = model(rgb, depth, detr_targets)
      loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    loss_dict = outputs.loss_dict

    total_loss += loss.item()
    total_vfl += loss_dict.get("loss_vfl", torch.tensor(0.0)).item()
    total_bbox += loss_dict.get("loss_bbox", torch.tensor(0.0)).item()
    total_giou += loss_dict.get("loss_giou", torch.tensor(0.0)).item()

    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

  num_batches = len(loader)

  return {
    "loss": total_loss / num_batches,
    "loss_vfl": total_vfl / num_batches,
    "loss_bbox": total_bbox / num_batches,
    "loss_giou": total_giou / num_batches
  }