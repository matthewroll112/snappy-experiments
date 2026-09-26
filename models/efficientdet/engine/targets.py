import torch


def normalized_cxcywh_to_yxyx(boxes, image_height, image_width):
  """Convert normalized cxcywh boxes to absolute yxyx boxes."""

  cx, cy, w, h = boxes.unbind(dim=-1)

  x1 = (cx - w / 2) * image_width
  y1 = (cy - h / 2) * image_height
  x2 = (cx + w / 2) * image_width
  y2 = (cy + h / 2) * image_height

  return torch.stack([y1, x1, y2, x2], dim=-1)

def prepare_efficientdet_targets(targets, image_height, image_width, device):
  """Prepare targets for EfficientDet anchor assignment."""

  boxes = []
  classes = []

  for target in targets:
    target_boxes = target["boxes"].to(device, non_blocking=True)
    target_labels = target["labels"].to(device, non_blocking=True)

    target_boxes = normalized_cxcywh_to_yxyx(
      target_boxes,
      image_height,
      image_width
    )

    # EfficientDet anchor labeler expects foreground classes to start at 1
    target_labels = target_labels + 1

    boxes.append(target_boxes)
    classes.append(target_labels)

  return {
    "bbox": boxes,
    "cls": classes
  }