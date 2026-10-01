import copy

import torch.nn as nn


class YOLOHead(nn.Module):
  """Pretrained YOLO11 neck and detection head."""

  def __init__(self, pretrained_model: nn.Module):
    super().__init__()

    layers = pretrained_model.model

    self.upsample1 = copy.deepcopy(layers[11])
    self.concat1 = copy.deepcopy(layers[12])
    self.c3k2_1 = copy.deepcopy(layers[13])

    self.upsample2 = copy.deepcopy(layers[14])
    self.concat2 = copy.deepcopy(layers[15])
    self.c3k2_2 = copy.deepcopy(layers[16])

    self.conv1 = copy.deepcopy(layers[17])
    self.concat3 = copy.deepcopy(layers[18])
    self.c3k2_3 = copy.deepcopy(layers[19])

    self.conv2 = copy.deepcopy(layers[20])
    self.concat4 = copy.deepcopy(layers[21])
    self.c3k2_4 = copy.deepcopy(layers[22])

    self.detect = copy.deepcopy(layers[23])

  def forward(self, p3, p4, p5):
    x = self.upsample1(p5)
    x = self.concat1([x, p4])
    p4_head = self.c3k2_1(x)

    x = self.upsample2(p4_head)
    x = self.concat2([x, p3])
    p3_head = self.c3k2_2(x)

    x = self.conv1(p3_head)
    x = self.concat3([x, p4_head])
    p4_head = self.c3k2_3(x)

    x = self.conv2(p4_head)
    x = self.concat4([x, p5])
    p5_head = self.c3k2_4(x)

    return self.detect([p3_head, p4_head, p5_head])