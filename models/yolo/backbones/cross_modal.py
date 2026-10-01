import copy

import torch.nn as nn

from src.fusion import CAFIM, GCFFM

from .utils import create_depth_stem


class CrossModalBackbone(nn.Module):
  """RGB-D YOLO11l backbone using CAFIM and GCFFM at P3, P4 and P5."""

  def __init__(self, pretrained_model: nn.Module):
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

    # P3 cross-modal interaction
    self.cafim_p3 = CAFIM(256)
    self.gcffm_p3 = GCFFM(256)

    # P4 cross-modal interaction
    self.cafim_p4 = CAFIM(512)
    self.gcffm_p4 = GCFFM(512)

    # P5 cross-modal interaction
    self.cafim_p5 = CAFIM(512)
    self.gcffm_p5 = GCFFM(512)

  def forward(self, rgb, depth):
    # RGB and depth to P3
    rgb = self.rgb_conv1(rgb)
    rgb = self.rgb_conv2(rgb)
    rgb = self.rgb_c3k2_1(rgb)
    rgb = self.rgb_conv3(rgb)
    rgb = self.rgb_c3k2_2(rgb)

    depth = self.depth_conv1(depth)
    depth = self.depth_conv2(depth)
    depth = self.depth_c3k2_1(depth)
    depth = self.depth_conv3(depth)
    depth = self.depth_c3k2_2(depth)

    # P3 cross-modal interaction
    rgb, depth = self.cafim_p3([rgb, depth])
    p3 = self.gcffm_p3([rgb, depth])

    # RGB and depth to P4
    rgb = self.rgb_conv4(rgb)
    rgb = self.rgb_c3k2_3(rgb)

    depth = self.depth_conv4(depth)
    depth = self.depth_c3k2_3(depth)

    # P4 cross-modal interaction
    rgb, depth = self.cafim_p4([rgb, depth])
    p4 = self.gcffm_p4([rgb, depth])

    # RGB and depth to P5
    rgb = self.rgb_conv5(rgb)
    rgb = self.rgb_c3k2_4(rgb)
    rgb = self.rgb_sppf(rgb)
    rgb = self.rgb_c2psa(rgb)

    depth = self.depth_conv5(depth)
    depth = self.depth_c3k2_4(depth)
    depth = self.depth_sppf(depth)
    depth = self.depth_c2psa(depth)

    # P5 cross-modal interaction
    rgb, depth = self.cafim_p5([rgb, depth])
    p5 = self.gcffm_p5([rgb, depth])

    return p3, p4, p5