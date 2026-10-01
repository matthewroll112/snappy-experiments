import copy

import torch.nn as nn

from src.fusion import create_fusion

from .utils import create_depth_stem
from src.types import FusionType


class LatePartialFusionBackbone(nn.Module):
  """RGB-D YOLO11l backbone with convolutional late fusion at P4 and P5."""

  def __init__(self, pretrained_model: nn.Module, fusion_type: FusionType):
    super().__init__()

    layers = pretrained_model.model

    # RGB backbone
    self.rgb_conv1 = copy.deepcopy(layers[0])
    self.rgb_conv2 = copy.deepcopy(layers[1])
    self.rgb_c3k2_1 = copy.deepcopy(layers[2])
    self.rgb_conv3 = copy.deepcopy(layers[3])
    self.rgb_c3k2_2 = copy.deepcopy(layers[4])
    self.rgb_conv4 = copy.deepcopy(layers[5])
    self.rgb_c3k2_3 = copy.deepcopy(layers[6])
    self.rgb_conv5 = copy.deepcopy(layers[7])
    self.rgb_c3k2_4 = copy.deepcopy(layers[8])
    self.rgb_sppf = copy.deepcopy(layers[9])
    self.rgb_c2psa = copy.deepcopy(layers[10])

    # Depth backbone
    self.depth_conv1 = create_depth_stem(layers[0])
    self.depth_conv2 = copy.deepcopy(layers[1])
    self.depth_c3k2_1 = copy.deepcopy(layers[2])
    self.depth_conv3 = copy.deepcopy(layers[3])
    self.depth_c3k2_2 = copy.deepcopy(layers[4])
    self.depth_conv4 = copy.deepcopy(layers[5])
    self.depth_c3k2_3 = copy.deepcopy(layers[6])
    self.depth_conv5 = copy.deepcopy(layers[7])
    self.depth_c3k2_4 = copy.deepcopy(layers[8])
    self.depth_sppf = copy.deepcopy(layers[9])
    self.depth_c2psa = copy.deepcopy(layers[10])

    # Late fusion at P4 and P5 only
    self.fusion_p4 = create_fusion(fusion_type, 512)
    self.fusion_p5 = create_fusion(fusion_type, 512)

  def forward(self, rgb, depth):
    # RGB stream
    rgb = self.rgb_conv1(rgb)
    rgb = self.rgb_conv2(rgb)
    rgb = self.rgb_c3k2_1(rgb)

    rgb = self.rgb_conv3(rgb)
    rgb_p3 = self.rgb_c3k2_2(rgb)

    rgb = self.rgb_conv4(rgb_p3)
    rgb_p4 = self.rgb_c3k2_3(rgb)

    rgb = self.rgb_conv5(rgb_p4)
    rgb = self.rgb_c3k2_4(rgb)
    rgb = self.rgb_sppf(rgb)
    rgb_p5 = self.rgb_c2psa(rgb)

    # Depth stream
    depth = self.depth_conv1(depth)
    depth = self.depth_conv2(depth)
    depth = self.depth_c3k2_1(depth)

    depth = self.depth_conv3(depth)
    depth_p3 = self.depth_c3k2_2(depth)

    depth = self.depth_conv4(depth_p3)
    depth_p4 = self.depth_c3k2_3(depth)

    depth = self.depth_conv5(depth_p4)
    depth = self.depth_c3k2_4(depth)
    depth = self.depth_sppf(depth)
    depth_p5 = self.depth_c2psa(depth)

    # Keep P3 RGB-only, fuse RGB-D at P4 and P5
    p3 = rgb_p3
    p4 = self.fusion_p4(rgb_p4, depth_p4)
    p5 = self.fusion_p5(rgb_p5, depth_p5)

    return p3, p4, p5