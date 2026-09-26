import copy

import torch


def create_depth_stem(rgb_backbone):
  """Create a single-channel EfficientNet stem from a pretrained RGB backbone."""

  depth_stem = copy.deepcopy(rgb_backbone.conv_stem)
  old_conv = rgb_backbone.conv_stem

  new_conv = type(old_conv)(
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

  depth_stem = new_conv

  return depth_stem


def convert_to_four_channel(backbone):
  """Convert an EfficientNet backbone stem from RGB to RGB-D input."""

  old_conv = backbone.conv_stem

  new_conv = type(old_conv)(
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

  backbone.conv_stem = new_conv