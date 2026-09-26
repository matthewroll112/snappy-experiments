import copy

import torch.nn as nn

from src.fusion import ConvFusion

from .utils import create_depth_stem


class MidFusionBackbone(nn.Module):
  """EfficientNet-B0 backbone with RGB-D convolutional mid fusion at P3."""

  def __init__(self, pretrained_backbone):
    super().__init__()

    blocks = pretrained_backbone.blocks

    # RGB stream to P3
    self.rgb_stem = copy.deepcopy(pretrained_backbone.conv_stem)
    self.rgb_bn1 = copy.deepcopy(pretrained_backbone.bn1)
    self.rgb_group0 = copy.deepcopy(blocks[0])
    self.rgb_group1 = copy.deepcopy(blocks[1])
    self.rgb_group2 = copy.deepcopy(blocks[2])

    # Depth stream to P3
    self.depth_stem = create_depth_stem(pretrained_backbone)
    self.depth_bn1 = copy.deepcopy(pretrained_backbone.bn1)
    self.depth_group0 = copy.deepcopy(blocks[0])
    self.depth_group1 = copy.deepcopy(blocks[1])
    self.depth_group2 = copy.deepcopy(blocks[2])

    # Mid fusion at P3
    self.fusion = ConvFusion(40)

    # Shared deeper backbone
    self.group3 = copy.deepcopy(blocks[3])
    self.group4 = copy.deepcopy(blocks[4])
    self.group5 = copy.deepcopy(blocks[5])
    self.group6 = copy.deepcopy(blocks[6])

  def forward(self, rgb, depth):
    # RGB stream to P3
    rgb = self.rgb_stem(rgb)
    rgb = self.rgb_bn1(rgb)
    rgb = self.rgb_group0(rgb)
    rgb = self.rgb_group1(rgb)
    rgb = self.rgb_group2(rgb)

    # Depth stream to P3
    depth = self.depth_stem(depth)
    depth = self.depth_bn1(depth)
    depth = self.depth_group0(depth)
    depth = self.depth_group1(depth)
    depth = self.depth_group2(depth)

    # Fuse at P3
    p3 = self.fusion(rgb, depth)

    # Shared backbone to P4
    x = self.group3(p3)
    p4 = self.group4(x)

    # Shared backbone to P5
    x = self.group5(p4)
    p5 = self.group6(x)

    return [p3, p4, p5]