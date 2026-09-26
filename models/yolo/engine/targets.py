import torch


def prepare_yolo_targets(targets, device):
  """Prepare normalized cxcywh targets for the YOLO detection loss."""

  batch_idx = []
  cls = []
  bboxes = []

  for image_idx, target in enumerate(targets):
    num_objects = len(target["labels"])

    if num_objects == 0:
      continue

    batch_idx.append(torch.full((num_objects,), image_idx, dtype=torch.long, device=device))
    cls.append(target["labels"].to(device).view(-1, 1))
    bboxes.append(target["boxes"].to(device))

  if not batch_idx:
    return {
      "batch_idx": torch.empty(0, dtype=torch.long, device=device),
      "cls": torch.empty((0, 1), dtype=torch.float32, device=device),
      "bboxes": torch.empty((0, 4), dtype=torch.float32, device=device)
    }

  return {
    "batch_idx": torch.cat(batch_idx),
    "cls": torch.cat(cls).float(),
    "bboxes": torch.cat(bboxes)
  }