import cv2
import numpy as np
import torch


class DetectionTransform:
  """Base transform for RGB-D detection data."""

  def __call__(self, rgb, depth, target):
    raise NotImplementedError

class Compose:
  """Apply a sequence of RGB-D detection transforms."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, rgb, depth, target):
    for transform in self.transforms:
      rgb, depth, target = transform(rgb, depth, target)

    return rgb, depth, target

class Resize(DetectionTransform):
  """Resize RGB and depth to a fixed spatial size."""

  def __init__(self, size):
    self.height, self.width = size

  def __call__(self, rgb, depth, target):
    rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    return rgb, depth, target

class ToTensor(DetectionTransform):
  """Convert RGB and depth arrays to float tensors."""

  def __call__(self, rgb, depth, target):
    rgb = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    depth = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).float()

    return rgb, depth, target

class Normalize(DetectionTransform):
  """Normalize RGB and depth values."""

  def __init__(self, rgb_scale=255.0, depth_scale=255.0):
    self.rgb_scale = rgb_scale
    self.depth_scale = depth_scale

  def __call__(self, rgb, depth, target):
    rgb = rgb / self.rgb_scale
    depth = depth / self.depth_scale

    return rgb, depth, target


def get_train_transforms(target_size=(640, 640)):
  return Compose([
    Resize(target_size),
    ToTensor(),
    Normalize()
  ])

def get_val_transforms(target_size=(640, 640)):
  return Compose([
    Resize(target_size),
    ToTensor(),
    Normalize()
  ])