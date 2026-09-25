from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SnappyDataset(Dataset):
  """
  RGB-D fish detection dataset.

  Images are stored as four-page TIFF files:
    pages 0-2: RGB
    page 3: depth

  Targets use normalized cxcywh boxes.
  """

  def __init__(self, root, split="train", transform=None):
    self.root = Path(root)
    self.split = split
    self.transform = transform

    self.image_dir = self.root / "images" / split
    self.label_dir = self.root / "labels" / split

    self.image_paths = sorted(self.image_dir.glob("*.tiff"))

    if len(self.image_paths) == 0:
      raise RuntimeError(f"No TIFF images found in {self.image_dir}")

    print(f"Loaded {len(self.image_paths)} {split} images")

  def __len__(self):
    return len(self.image_paths)

  def _load_tiff(self, path):
    success, frames = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)

    if not success:
      raise RuntimeError(f"Could not read TIFF: {path}")

    if len(frames) != 4:
      raise RuntimeError(f"{path} contains {len(frames)} pages, expected 4")

    rgb = np.stack(frames[:3], axis=-1)
    depth = frames[3]

    return rgb, depth

  def _load_labels(self, path):
    boxes = []
    labels = []

    if path.exists():
      lines = path.read_text().strip().splitlines()

      for line in lines:
        values = line.split()

        if len(values) != 5:
          raise ValueError(f"Invalid YOLO label in {path}: {line}")

        class_id, cx, cy, w, h = values

        labels.append(int(class_id))
        boxes.append([float(cx), float(cy), float(w), float(h)])

    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor(labels, dtype=torch.int64)

    return boxes, labels

  def __getitem__(self, index):
    image_path = self.image_paths[index]
    label_path = self.label_dir / f"{image_path.stem}.txt"

    rgb, depth = self._load_tiff(image_path)
    boxes, labels = self._load_labels(label_path)

    target = {
      "boxes": boxes,
      "labels": labels
    }

    if self.transform is not None:
      rgb, depth, target = self.transform(rgb, depth, target)

    return rgb, depth, target