import copy

import torch
import torch.nn as nn
from transformers.modeling_outputs import BackboneOutput

from src.fusion import ConvFusion

from .utils import create_depth_embedder


class LateFusionBackbone(nn.Module):
  """RGB-D late-fusion ResNet50 backbone for RT-DETR."""

  def __init__(self, pretrained_backbone):
    super().__init__()

    self.rgb_embedder = copy.deepcopy(pretrained_backbone.embedder)
    self.depth_embedder = create_depth_embedder(pretrained_backbone.embedder)

    self.rgb_c2 = copy.deepcopy(pretrained_backbone.encoder.stages[0])
    self.rgb_c3 = copy.deepcopy(pretrained_backbone.encoder.stages[1])
    self.rgb_c4 = copy.deepcopy(pretrained_backbone.encoder.stages[2])
    self.rgb_c5 = copy.deepcopy(pretrained_backbone.encoder.stages[3])

    self.depth_c2 = copy.deepcopy(pretrained_backbone.encoder.stages[0])
    self.depth_c3 = copy.deepcopy(pretrained_backbone.encoder.stages[1])
    self.depth_c4 = copy.deepcopy(pretrained_backbone.encoder.stages[2])
    self.depth_c5 = copy.deepcopy(pretrained_backbone.encoder.stages[3])

    self.fusion_c3 = ConvFusion(512)
    self.fusion_c4 = ConvFusion(1024)
    self.fusion_c5 = ConvFusion(2048)

  def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
    rgb = pixel_values[:, :3]
    depth = pixel_values[:, 3:4]

    rgb = self.rgb_embedder(rgb)
    depth = self.depth_embedder(depth)

    rgb = self.rgb_c2(rgb)
    depth = self.depth_c2(depth)

    rgb_c3 = self.rgb_c3(rgb)
    depth_c3 = self.depth_c3(depth)

    rgb_c4 = self.rgb_c4(rgb_c3)
    depth_c4 = self.depth_c4(depth_c3)

    rgb_c5 = self.rgb_c5(rgb_c4)
    depth_c5 = self.depth_c5(depth_c4)

    c3 = self.fusion_c3(rgb_c3, depth_c3)
    c4 = self.fusion_c4(rgb_c4, depth_c4)
    c5 = self.fusion_c5(rgb_c5, depth_c5)

    return BackboneOutput(feature_maps=(c3, c4, c5))