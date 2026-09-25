import torch

def collate_fn(batch):
  rgb, depth, targets = zip(*batch)

  rgb = torch.stack(rgb, dim=0)
  depth = torch.stack(depth, dim=0)

  return rgb, depth, list(targets)