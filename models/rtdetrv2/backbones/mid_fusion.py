import copy

import torch
import torch.nn as nn
from transformers.modeling_outputs import BackboneOutput

from src.fusion import ConvFusion

from .utils import create_depth_embedder

class MidFusionBackbone(nn.Module):
  """RGB-D mid-fusion ResNet50 backbone for RT-DETR."""

  def __init__(self, pretrained_backbone):
    super().__init__()

    self.rgb_embedder = copy.deepcopy(pretrained_backbone.embedder)
    self.depth_embedder = create_depth_embedder(pretrained_backbone.embedder)

    self.rgb_c2 = copy.deepcopy(pretrained_backbone.encoder.stages[0])
    self.rgb_c3 = copy.deepcopy(pretrained_backbone.encoder.stages[1])

    self.depth_c2 = copy.deepcopy(pretrained_backbone.encoder.stages[0])
    self.depth_c3 = copy.deepcopy(pretrained_backbone.encoder.stages[1])

    self.c4 = copy.deepcopy(pretrained_backbone.encoder.stages[2])
    self.c5 = copy.deepcopy(pretrained_backbone.encoder.stages[3])

    self.fusion = ConvFusion(512)

  def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
    rgb = pixel_values[:, :3]
    depth = pixel_values[:, 3:4]

    rgb = self.rgb_embedder(rgb)
    depth = self.depth_embedder(depth)

    rgb = self.rgb_c2(rgb)
    depth = self.depth_c2(depth)

    rgb = self.rgb_c3(rgb)
    depth = self.depth_c3(depth)

    c3 = self.fusion(rgb, depth)

    c4 = self.c4(c3)
    c5 = self.c5(c4)

    return BackboneOutput(feature_maps=(c3, c4, c5))