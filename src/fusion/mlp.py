import torch
import torch.nn as nn

class MLPFusion(nn.Module):
  """Spatial MLP fusion from Orfaig et al. using conv"""
  def __init__(self, channels):
    super().__init__()

    input_channels = channels * 2

    self.mlp = nn.Sequential(
      nn.Conv2d(input_channels, input_channels, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(input_channels, channels, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(channels, channels, kernel_size=1)
    )

  def forward(self, rgb, depth):
    x = torch.cat([rgb, depth], dim=1)
    return self.mlp(x)