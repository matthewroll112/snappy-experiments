import torch
import torch.nn as nn

from transformers import RTDetrV2ForObjectDetection
from transformers.utils import logging

from .backbones import CrossModalBackbone, LateFusionBackbone, MidFusionBackbone, LatePartialFusionBackbone
from .backbones.utils import convert_to_four_channel

from src.types import FusionType, FusionStage


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

  def __init__(self, num_classes: int, fusion_stage: FusionStage, fusion_type: FusionType):
    super().__init__()

    self.model = load_pretrained_model(num_classes)
    pretrained_backbone = self.model.model.backbone.model

    if fusion_stage == "early":
      convert_to_four_channel(pretrained_backbone)
    elif fusion_stage == "mid":
      self.model.model.backbone.model = MidFusionBackbone(pretrained_backbone, fusion_type)
    elif fusion_stage == "late":
      self.model.model.backbone.model = LateFusionBackbone(pretrained_backbone, fusion_type)
    elif fusion_stage == "late_partial":
      self.model.model.backbone.model = LatePartialFusionBackbone(pretrained_backbone, fusion_type)
    elif fusion_stage == "cafim_gcffm":
      self.model.model.backbone.model = CrossModalBackbone(pretrained_backbone)
    else:
      raise ValueError(f"Unknown fusion type: {fusion_stage}")

  def forward(self, rgb: torch.Tensor, depth: torch.Tensor, targets=None):
    x = torch.cat([rgb, depth], dim=1)
    return self.model(pixel_values=x, labels=targets)