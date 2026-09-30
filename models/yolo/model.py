from copy import deepcopy

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from .head import YOLOHead

from .backbones.utils import convert_to_four_channel
from .backbones import MidFusionBackbone, LateFusionBackbone, CrossModalBackbone


MODEL_NAME = "yolov11l.pt"


def load_pretrained_model(num_classes):
  """Create a native YOLOv8n for the required number of classes and load compatible pretrained weights."""

  pretrained = YOLO(MODEL_NAME).model

  model = DetectionModel(
    cfg=deepcopy(pretrained.yaml),
    ch=3,
    nc=num_classes,
    verbose=False
  )

  model.load(pretrained)

  return model


class RGBYOLODetector(nn.Module):
  """Pretrained RGB YOLOv8n baseline."""

  def __init__(self, num_classes=2):
    super().__init__()
    self.model = load_pretrained_model(num_classes)

  @property
  def detect(self):
    return self.model.model[-1]

  def forward(self, rgb, depth=None):
    return self.model(rgb)


class RGBDYOLODetector(nn.Module):
  """RGB-D YOLOv8n detector with configurable fusion."""

  def __init__(self, num_classes, fusion_type):
    super().__init__()

    pretrained_model = load_pretrained_model(num_classes)

    if fusion_type == "early":
      convert_to_four_channel(pretrained_model)
      self.yolo_model = pretrained_model
      self.backbone = None
      self.head = None

    elif fusion_type == "mid":
      self.yolo_model = None
      self.backbone = MidFusionBackbone(pretrained_model)
      self.head = YOLOHead(pretrained_model)

    elif fusion_type == "late":
      self.yolo_model = None
      self.backbone = LateFusionBackbone(pretrained_model)
      self.head = YOLOHead(pretrained_model)

    elif fusion_type == "cafim_gcffm":
      self.yolo_model = None
      self.backbone = CrossModalBackbone(pretrained_model)
      self.head = YOLOHead(pretrained_model)

    else:
      raise ValueError(f"Unknown fusion type: {fusion_type}")

  @property
  def detect(self):
    if self.yolo_model is not None:
      return self.yolo_model.model[-1]

    return self.head.detect

  def forward(self, rgb, depth):
    if self.yolo_model is not None:
      x = torch.cat([rgb, depth], dim=1)
      return self.yolo_model(x)

    p3, p4, p5 = self.backbone(rgb, depth)
    return self.head(p3, p4, p5)