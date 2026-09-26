import torch
import torch.nn as nn

import copy


def convert_to_four_channel(model):
  """Convert a pretrained YOLOv8 model to accept RGB-D input."""

  stem = model.model[0]
  old_conv = stem.conv

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

  stem.conv = new_conv

def configure_detection_head(model, num_classes):
  """Change the YOLO detection head to the required number of classes."""

  detect = model.model[-1]

  for branch in detect.cv3:
    old_conv = branch[-1]

    branch[-1] = nn.Conv2d(
      in_channels=old_conv.in_channels,
      out_channels=num_classes,
      kernel_size=old_conv.kernel_size,
      stride=old_conv.stride,
      padding=old_conv.padding,
      bias=old_conv.bias is not None
    )

  detect.nc = num_classes
  detect.no = detect.reg_max * 4 + num_classes

def create_depth_stem(rgb_stem):
  """Create a single-channel YOLO stem from a pretrained RGB stem."""

  depth_stem = copy.deepcopy(rgb_stem)
  old_conv = rgb_stem.conv

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

  depth_stem.conv = new_conv

  return depth_stem