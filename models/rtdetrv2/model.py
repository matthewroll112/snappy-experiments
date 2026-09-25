import torch
import torch.nn as nn

from transformers import RTDetrV2ForObjectDetection
from transformers.utils import logging

from .backbones import CrossModalBackbone, LateFusionBackbone, MidFusionBackbone
from .backbones.utils import convert_to_four_channel


logging.set_verbosity_error()
MODEL_NAME = "PekingU/rtdetr_v2_r50vd"

def load_pretrained_model(num_classes: int) -> RTDetrV2ForObjectDetection:
  """Load the pretrained RT-DETRv2 model with a custom number of classes."""

  return RTDetrV2ForObjectDetection.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
  )

class RGBRTDETRDetector(nn.Module):
  """RGB RT-DETRv2 baseline."""

  def __init__(self, num_classes: int):
    super().__init__()

    self.model = load_pretrained_model(num_classes)

  def forward(self, rgb: torch.Tensor, depth=None, targets=None):
    return self.model(pixel_values=rgb, labels=targets)

class RGBDRTDETRDetector(nn.Module):
  """RGB-D RT-DETRv2 detector with configurable fusion."""

  def __init__(self, num_classes: int, fusion_type: str):
    super().__init__()

    self.model = load_pretrained_model(num_classes)
    pretrained_backbone = self.model.model.backbone.model

    if fusion_type == "early":
      convert_to_four_channel(pretrained_backbone)
    elif fusion_type == "mid":
      self.model.model.backbone.model = MidFusionBackbone(pretrained_backbone)
    elif fusion_type == "late":
      self.model.model.backbone.model = LateFusionBackbone(pretrained_backbone)
    elif fusion_type == "cross_modal":
      self.model.model.backbone.model = CrossModalBackbone(pretrained_backbone)
    else:
      raise ValueError(f"Unknown fusion type: {fusion_type}")

  def forward(self, rgb: torch.Tensor, depth: torch.Tensor, targets=None):
    x = torch.cat([rgb, depth], dim=1)
    return self.model(pixel_values=x, labels=targets)