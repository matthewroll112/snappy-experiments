import torch
from tqdm import tqdm

from .targets import prepare_efficientdet_targets


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
  model.train()

  total_loss = 0.0
  total_class_loss = 0.0
  total_box_loss = 0.0

  progress_bar = tqdm(loader, desc=f"Train {epoch:03d}", leave=False, dynamic_ncols=True)

  for rgb, depth, targets in progress_bar:
    rgb = rgb.to(device, non_blocking=True)
    depth = depth.to(device, non_blocking=True)

    image_height, image_width = rgb.shape[-2:]

    efficientdet_targets = prepare_efficientdet_targets(
      targets,
      image_height,
      image_width,
      device
    )

    optimizer.zero_grad(set_to_none=True)

    class_outputs, box_outputs = model(rgb, depth)

    loss, class_loss, box_loss = criterion(
      class_outputs,
      box_outputs,
      efficientdet_targets
    )

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total_class_loss += class_loss.item()
    total_box_loss += box_loss.item()

    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

  num_batches = len(loader)

  return {
    "loss": total_loss / num_batches,
    "loss_class": total_class_loss / num_batches,
    "loss_box": total_box_loss / num_batches
  }