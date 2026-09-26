import torch
import torch.nn as nn

from effdet import create_model

from .backbones import MidFusionBackbone, LateFusionBackbone, CrossModalBackbone
from .backbones.utils import convert_to_four_channel


MODEL_NAME = "tf_efficientdet_d0"
IMAGE_SIZE = (640, 640)


def load_pretrained_model(num_classes):
  """Load pretrained EfficientDet-D0 configured for 640x640 input."""

  return create_model(
    MODEL_NAME,
    bench_task="",
    num_classes=num_classes,
    pretrained=True,
    image_size=IMAGE_SIZE
  )


class RGBEfficientDetDetector(nn.Module):
  """Pretrained RGB EfficientDet-D0 baseline."""

  def __init__(self, num_classes):
    super().__init__()
    self.model = load_pretrained_model(num_classes)

  def forward(self, rgb, depth=None):
    return self.model(rgb)


class RGBDEfficientDetDetector(nn.Module):
  """RGB-D EfficientDet-D0 detector with configurable fusion."""

  def __init__(self, num_classes, fusion_type):
    super().__init__()

    self.model = load_pretrained_model(num_classes)
    self.fusion_type = fusion_type

    if fusion_type == "early":
      convert_to_four_channel(self.model.backbone)
      self.backbone = None
    elif fusion_type == "mid":
      self.backbone = MidFusionBackbone(self.model.backbone)
    elif fusion_type == "late":
      self.backbone = LateFusionBackbone(self.model.backbone)
    elif fusion_type == "cafim_gcffm":
      self.backbone = CrossModalBackbone(self.model.backbone)
    else:
      raise ValueError(f"Unknown fusion type: {fusion_type}")

  def forward(self, rgb, depth):
    if self.fusion_type == "early":
      x = torch.cat([rgb, depth], dim=1)
      return self.model(x)

    features = self.backbone(rgb, depth)

    features = self.model.fpn(features)

    class_outputs = self.model.class_net(features)
    box_outputs = self.model.box_net(features)

    return class_outputs, box_outputs