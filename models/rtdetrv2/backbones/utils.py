import copy

import torch
import torch.nn as nn


def create_depth_embedder(rgb_embedder: nn.Module) -> nn.Module:
  """Create a single-channel depth embedder from a pretrained RGB embedder."""

  depth_embedder = copy.deepcopy(rgb_embedder)

  old_conv = rgb_embedder.embedder[0].convolution

  new_conv = nn.Conv2d(
    in_channels=1,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    dilation=old_conv.dilation,
    groups=old_conv.groups,
    bias=old_conv.bias is not None
  )

  with torch.no_grad():
    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    if old_conv.bias is not None:
      new_conv.bias.copy_(old_conv.bias)

  depth_embedder.embedder[0].convolution = new_conv
  depth_embedder.num_channels = 1

  return depth_embedder


def convert_to_four_channel(backbone: nn.Module) -> None:
  """Convert a pretrained RGB backbone to accept RGB-D input."""

  embedder = backbone.embedder
  old_conv = embedder.embedder[0].convolution

  new_conv = nn.Conv2d(
    in_channels=4,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    dilation=old_conv.dilation,
    groups=old_conv.groups,
    bias=old_conv.bias is not None
  )

  with torch.no_grad():
    new_conv.weight[:, :3].copy_(old_conv.weight)
    new_conv.weight[:, 3:4].copy_(old_conv.weight.mean(dim=1, keepdim=True))

    if old_conv.bias is not None:
      new_conv.bias.copy_(old_conv.bias)

  embedder.embedder[0].convolution = new_conv
  embedder.num_channels = 4