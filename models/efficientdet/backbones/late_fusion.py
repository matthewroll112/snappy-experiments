import copy

import torch.nn as nn

from src.fusion import ConvFusion

from .utils import create_depth_stem


class LateFusionBackbone(nn.Module):
  """EfficientNet-B0 backbone with RGB-D late fusion at P3, P4 and P5."""

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

    # Late fusion
    self.fusion_p3 = ConvFusion(40)
    self.fusion_p4 = ConvFusion(112)
    self.fusion_p5 = ConvFusion(320)

  def forward(self, rgb, depth):
    # RGB stream
    rgb = self.rgb_stem(rgb)
    rgb = self.rgb_bn1(rgb)

    rgb = self.rgb_group0(rgb)
    rgb = self.rgb_group1(rgb)
    rgb_p3 = self.rgb_group2(rgb)

    rgb = self.rgb_group3(rgb_p3)
    rgb_p4 = self.rgb_group4(rgb)

    rgb = self.rgb_group5(rgb_p4)
    rgb_p5 = self.rgb_group6(rgb)

    # Depth stream
    depth = self.depth_stem(depth)
    depth = self.depth_bn1(depth)

    depth = self.depth_group0(depth)
    depth = self.depth_group1(depth)
    depth_p3 = self.depth_group2(depth)

    depth = self.depth_group3(depth_p3)
    depth_p4 = self.depth_group4(depth)

    depth = self.depth_group5(depth_p4)
    depth_p5 = self.depth_group6(depth)

    # Fuse backbone features
    p3 = self.fusion_p3(rgb_p3, depth_p3)
    p4 = self.fusion_p4(rgb_p4, depth_p4)
    p5 = self.fusion_p5(rgb_p5, depth_p5)

    return [p3, p4, p5]