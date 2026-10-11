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

class RandomHorizontalFlip(DetectionTransform):
  """Randomly flip RGB, depth and bounidng boxes horizontally"""
  
  def __init__(self, prob=0.5):
    self.prob = prob

  def __call__(self, rgb, depth, target):
    if np.random.random() >= self.prob:
      return rgb, depth, target

    rgb = np.ascontiguousarray(np.flip(rgb, axis=1))
    depth = np.ascontiguousarray(np.flip(depth, axis=1))

    boxes = target["boxes"].clone()

    if len(boxes) > 0:
      boxes[:,0] = 1.0 - boxes[:,0]

    target = target.copy()
    target["boxes"] = boxes

    return rgb, depth, target

class RandomVerticalFlip(DetectionTransform):
  """Randomly flip RGB, depth and bounding boxes vertically."""

  def __init__(self, prob=0.5):
    self.prob = prob

  def __call__(self, rgb, depth, target):
    if np.random.random() >= self.prob:
      return rgb, depth, target

    rgb = np.ascontiguousarray(np.flip(rgb, axis=0))
    depth = np.ascontiguousarray(np.flip(depth, axis=0))

    boxes = target["boxes"].clone()

    if len(boxes) > 0:
      boxes[:, 1] = 1.0 - boxes[:, 1]

    target = target.copy()
    target["boxes"] = boxes

    return rgb, depth, target

class RandomCrop(DetectionTransform):
  """Randomly crop RGB and depth while updating normalised cxcywh boxes"""
  
  def __init__(self, prob=0.5, min_scale = 0.8):
    self.prob = prob
    self.min_scale = min_scale

  def __call__(self, rgb, depth, target):
    if np.random.random() >= self.prob:
      return rgb, depth, target

    height, width = rgb.shape[:2]

    scale = np.random.uniform(self.min_scale, 1.0)

    crop_height = int(height * scale)
    crop_width = int(width * scale)

    max_y = height - crop_height
    max_x = width - crop_width

    top = np.random.randint(0, max_y + 1)
    left = np.random.randint(0, max_x + 1)

    bottom = top + crop_height
    right = left + crop_width

    rgb = rgb[top:bottom, left:right]
    depth = depth[top:bottom, left:right]

    boxes = target["boxes"].clone()
    labels = target["labels"].clone()

    if len(boxes) > 0:
      cx = boxes[:,0] * width
      cy = boxes[:,1] * height
      box_width = boxes[:,2] * width
      box_height = boxes[:,3] * height

      x1 = cx - box_width / 2
      y1 = cy - box_height / 2
      x2 = cx + box_width / 2
      y2 = cy + box_height / 2

      x1 = x1.clamp(left, right) - left
      y1 = y1.clamp(top, bottom) - top
      x2 = x2.clamp(left, right) - left
      y2 = y2.clamp(top, bottom) - top

      new_width = x2 - x1
      new_height = y2 - y1

      original_area = box_width * box_height
      new_area = new_width * new_height

      keep = (
        (new_width > 1) &
        (new_height > 1) &
        (new_area / original_area >= 0.3)
      )

      x1 = x1[keep]
      y1 = y1[keep]
      x2 = x2[keep]
      y2 = y2[keep]
      labels = labels[keep]

      cx = (x1 + x2) / 2 / crop_width
      cy = (y1 + y2) / 2 / crop_height
      box_width = (x2 - x1) / crop_width
      box_height = (y2 - y1) / crop_height

      boxes = torch.stack([cx, cy, box_width, box_height], dim=1)

    target = target.copy()
    target["boxes"] = boxes
    target["labels"] = labels

    return rgb, depth, target


class RandomBrightness(DetectionTransform):
  """Randomly change RGB brightness"""

  def __init__(self, prob=0.5, max_change=0.15):
    self.prob = prob
    self.max_change = max_change

  def __call__(self, rgb, depth, target):
    if np.random.random() >= self.prob:
      return rgb, depth, target

    factor = np.random.uniform(1.0 - self.max_change, 1.0 + self.max_change)

    rgb = rgb.astype(np.float32) * factor
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb, depth, target

class RandomDepthBlur(DetectionTransform):
  """Randomly blur the depth map to simulate oversmoothed depth estimation"""

  def __init__(self, prob=0.3, kernel_sizes=(3, 5, 7)):
    self.prob = prob
    self.kernel_sizes = kernel_sizes

  def __call__(self, rgb, depth, target):
    if np.random.random() >= self.prob:
      return rgb, depth, target

    kernel_size = int(np.random.choice(self.kernel_sizes))
    depth = cv2.GaussianBlur(depth, (kernel_size, kernel_size), 0)

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
    RandomHorizontalFlip(prob=0.5),
    RandomVerticalFlip(prob=0.5),
    RandomCrop(prob=0.5, min_scale=0.8),
    RandomBrightness(prob=0.5, max_change=0.15),
    RandomDepthBlur(prob=0.4, kernel_sizes=(3, 5, 7)),
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