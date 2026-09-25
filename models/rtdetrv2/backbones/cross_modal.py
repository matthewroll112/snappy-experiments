import copy

import torch
import torch.nn as nn
from transformers.modeling_outputs import BackboneOutput

from src.fusion import CAFIM, GCFFM

from .utils import create_depth_embedder


class CrossModalBackbone(nn.Module):
  """Two-stream RGB-D ResNet50 backbone using CAFIM and GCFFM."""

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

    self.cafim_c3 = CAFIM(512)
    self.cafim_c4 = CAFIM(1024)
    self.cafim_c5 = CAFIM(2048)

    self.gcffm_c3 = GCFFM(512)
    self.gcffm_c4 = GCFFM(1024)
    self.gcffm_c5 = GCFFM(2048)

  def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
    rgb = pixel_values[:, :3]
    depth = pixel_values[:, 3:4]

    # Stem
    rgb = self.rgb_embedder(rgb)
    depth = self.depth_embedder(depth)

    # C2
    rgb = self.rgb_c2(rgb)
    depth = self.depth_c2(depth)

    # C3
    rgb = self.rgb_c3(rgb)
    depth = self.depth_c3(depth)

    rgb, depth = self.cafim_c3([rgb, depth])
    c3 = self.gcffm_c3([rgb, depth])

    # C4
    rgb = self.rgb_c4(rgb)
    depth = self.depth_c4(depth)

    rgb, depth = self.cafim_c4([rgb, depth])
    c4 = self.gcffm_c4([rgb, depth])

    # C5
    rgb = self.rgb_c5(rgb)
    depth = self.depth_c5(depth)

    rgb, depth = self.cafim_c5([rgb, depth])
    c5 = self.gcffm_c5([rgb, depth])

    return BackboneOutput(feature_maps=(c3, c4, c5))