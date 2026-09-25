import torch
import torch.nn as nn

class ConvFusion(nn.Module):
  def __init__(self, channels: int):
    super().__init__()

    self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

  def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    return self.conv(torch.cat([rgb, depth], dim=1))
