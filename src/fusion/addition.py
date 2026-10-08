import torch.nn as nn

class AdditionFusion(nn.Module):
  """Fuse RGB and depth features using element-wise addition"""
  def __init__(self, channels=None):
    super().__init__()
  def forward(self, rgb, depth):
    return rgb + depth