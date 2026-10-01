import torch
import torch.nn as nn

class GatedFusion(nn.Module):
  """Fuse depth into RGB using a learned spatial-channel gate"""
  def __init__(self, channels):
    super().__init__()

    self.gate = nn.Sequential(
      nn.Conv2d(channels * 2, channels, kernel_size=1),
      nn.Sigmoid()
    )

  def forward(self, rgb, depth):
    gate = self.gate(torch.cat([rgb, depth], dim=1))
    return rgb + gate * depth