import torch

def normalized_cxcywh_to_xyxy(boxes, image_height, image_width):
  """Convert normalized cxcywh boxes to absolute xyxy."""

  cx, cy, w, h = boxes.unbind(dim=-1)

  x1 = (cx - w / 2) * image_width
  y1 = (cy - h / 2) * image_height
  x2 = (cx + w / 2) * image_width
  y2 = (cy + h / 2) * image_height

  return torch.stack([x1, y1, x2, y2], dim=-1)