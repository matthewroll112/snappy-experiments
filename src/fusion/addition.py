import torch.nn as nn

class AdditionFusion(nn.Module):
  """FUSE RGB and depth features using element-wise addition"""
  def forward(self, rgb, depth):
    return rgb + depth