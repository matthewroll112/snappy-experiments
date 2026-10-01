import copy

import torch.nn as nn

from src.fusion import create_fusion

from .utils import create_depth_stem
from src.types import FusionType


class MidFusionBackbone(nn.Module):
  """RGB-D YOLO11l backbone with convolutional mid fusion at P3."""

  def __init__(self, pretrained_model: nn.Module, fusion_type: FusionType):
    super().__init__()

    layers = pretrained_model.model

    # RGB backbone to P3
    self.rgb_conv1 = copy.deepcopy(layers[0])
    self.rgb_conv2 = copy.deepcopy(layers[1])
    self.rgb_c3k2_1 = copy.deepcopy(layers[2])
    self.rgb_conv3 = copy.deepcopy(layers[3])
    self.rgb_c3k2_2 = copy.deepcopy(layers[4])

    # Depth backbone to P3
    self.depth_conv1 = create_depth_stem(layers[0])
    self.depth_conv2 = copy.deepcopy(layers[1])
    self.depth_c3k2_1 = copy.deepcopy(layers[2])
    self.depth_conv3 = copy.deepcopy(layers[3])
    self.depth_c3k2_2 = copy.deepcopy(layers[4])

    # P3 fusion
    self.fusion = create_fusion(fusion_type, 256)

    # Shared backbone
    self.conv4 = copy.deepcopy(layers[5])
    self.c3k2_3 = copy.deepcopy(layers[6])
    self.conv5 = copy.deepcopy(layers[7])
    self.c3k2_4 = copy.deepcopy(layers[8])
    self.sppf = copy.deepcopy(layers[9])
    self.c2psa = copy.deepcopy(layers[10])

  def forward(self, rgb, depth):
    # RGB stream
    rgb = self.rgb_conv1(rgb)
    rgb = self.rgb_conv2(rgb)
    rgb = self.rgb_c3k2_1(rgb)
    rgb = self.rgb_conv3(rgb)
    rgb = self.rgb_c3k2_2(rgb)

    # Depth stream
    depth = self.depth_conv1(depth)
    depth = self.depth_conv2(depth)
    depth = self.depth_c3k2_1(depth)
    depth = self.depth_conv3(depth)
    depth = self.depth_c3k2_2(depth)

    # Mid fusion at P3
    p3 = self.fusion(rgb, depth)

    # Shared backbone
    p4 = self.conv4(p3)
    p4 = self.c3k2_3(p4)

    p5 = self.conv5(p4)
    p5 = self.c3k2_4(p5)
    p5 = self.sppf(p5)
    p5 = self.c2psa(p5)

    return p3, p4, p5