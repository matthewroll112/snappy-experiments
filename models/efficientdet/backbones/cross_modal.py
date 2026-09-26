import copy

import torch.nn as nn

from src.fusion import CAFIM, GCFFM

from .utils import create_depth_stem


class CrossModalBackbone(nn.Module):
  """EfficientNet-B0 backbone using CAFIM and GCFFM at P3, P4 and P5."""

  def __init__(self, pretrained_backbone):
    super().__init__()

    blocks = pretrained_backbone.blocks

    # RGB backbone
    self.rgb_stem = copy.deepcopy(pretrained_backbone.conv_stem)
    self.rgb_bn1 = copy.deepcopy(pretrained_backbone.bn1)

    self.rgb_group0 = copy.deepcopy(blocks[0])
    self.rgb_group1 = copy.deepcopy(blocks[1])
    self.rgb_group2 = copy.deepcopy(blocks[2])
    self.rgb_group3 = copy.deepcopy(blocks[3])
    self.rgb_group4 = copy.deepcopy(blocks[4])
    self.rgb_group5 = copy.deepcopy(blocks[5])
    self.rgb_group6 = copy.deepcopy(blocks[6])

    # Depth backbone
    self.depth_stem = create_depth_stem(pretrained_backbone)
    self.depth_bn1 = copy.deepcopy(pretrained_backbone.bn1)

    self.depth_group0 = copy.deepcopy(blocks[0])
    self.depth_group1 = copy.deepcopy(blocks[1])
    self.depth_group2 = copy.deepcopy(blocks[2])
    self.depth_group3 = copy.deepcopy(blocks[3])
    self.depth_group4 = copy.deepcopy(blocks[4])
    self.depth_group5 = copy.deepcopy(blocks[5])
    self.depth_group6 = copy.deepcopy(blocks[6])

    # P3 cross-modal interaction
    self.cafim_p3 = CAFIM(40)
    self.gcffm_p3 = GCFFM(40)

    # P4 cross-modal interaction
    self.cafim_p4 = CAFIM(112)
    self.gcffm_p4 = GCFFM(112)

    # P5 cross-modal interaction
    self.cafim_p5 = CAFIM(320)
    self.gcffm_p5 = GCFFM(320)

  def forward(self, rgb, depth):
    # RGB and depth to P3
    rgb = self.rgb_stem(rgb)
    rgb = self.rgb_bn1(rgb)
    rgb = self.rgb_group0(rgb)
    rgb = self.rgb_group1(rgb)
    rgb = self.rgb_group2(rgb)

    depth = self.depth_stem(depth)
    depth = self.depth_bn1(depth)
    depth = self.depth_group0(depth)
    depth = self.depth_group1(depth)
    depth = self.depth_group2(depth)

    # P3 cross-modal interaction
    rgb, depth = self.cafim_p3([rgb, depth])
    p3 = self.gcffm_p3([rgb, depth])

    # RGB and depth to P4
    rgb = self.rgb_group3(rgb)
    rgb = self.rgb_group4(rgb)

    depth = self.depth_group3(depth)
    depth = self.depth_group4(depth)

    # P4 cross-modal interaction
    rgb, depth = self.cafim_p4([rgb, depth])
    p4 = self.gcffm_p4([rgb, depth])

    # RGB and depth to P5
    rgb = self.rgb_group5(rgb)
    rgb = self.rgb_group6(rgb)

    depth = self.depth_group5(depth)
    depth = self.depth_group6(depth)

    # P5 cross-modal interaction
    rgb, depth = self.cafim_p5([rgb, depth])
    p5 = self.gcffm_p5([rgb, depth])

    return [p3, p4, p5]