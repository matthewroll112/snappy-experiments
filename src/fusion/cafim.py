import torch
import torch.nn as nn

class CAFIM(nn.Module):
  """
  Coordinate Attention Feature Interaction Module (Zhao et al, 2023)

  Input:
    x = [rgb, depth]:
      rgb: [B, C, H, W]
      depth: [B, C, H, W]

  Output:
    [rgb_out, depth_out]
    both [B, C, H, W]
  """

  def __init__(self, channels: int, reduction: int = 16):
    super().__init__()

    hidden_channels = max(channels // reduction, 8)

    self.mlp = nn.Sequential(
      nn.Linear(channels, hidden_channels),
      nn.ReLU(),
      nn.Linear(hidden_channels, hidden_channels),
      nn.ReLU(),
      nn.Linear(hidden_channels, hidden_channels),
      nn.ReLU(),
      nn.Linear(hidden_channels, hidden_channels),
      nn.ReLU(),
      nn.Linear(hidden_channels, hidden_channels),
      nn.ReLU(),
    )

    self.rgb_h_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1)
    self.rgb_w_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1)
    self.depth_h_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1)
    self.depth_w_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1)

  def forward(self, x):
    rgb, depth = x

    if rgb.shape != depth.shape:
      raise ValueError(f"Need RGB and Depth features with identical shapes: rgb{rgb.shape} and depth{depth.shape}")

    B, C, H, W = rgb.shape

    # ================= Direction aware pooling =================
    # [B, C, H, W] -> [B, C, H, 1]
    rgb_h = rgb.mean(dim=3, keepdim=True)
    depth_h = depth.mean(dim=3, keepdim=True)

    # [B, C, H, W] -> [B, C, 1, W]
    rgb_w = rgb.mean(dim=2, keepdim=True)
    depth_w = depth.mean(dim=2, keepdim=True)

    # [B, C, 1, W] -> [B, C, W, 1]
    rgb_w = rgb_w.permute(0, 1, 3, 2)
    depth_w = depth_w.permute(0, 1, 3, 2)

    # ================= Cross-modal descriptor interaction =================
    fused = torch.cat([rgb_h, rgb_w, depth_h, depth_w], dim=2)
    fused = fused.squeeze(-1)
    fused = fused.transpose(1, 2)
    fused = self.mlp(fused)
    fused = fused.transpose(1, 2)
    fused = fused.unsqueeze(-1)

    # ================= Split back =================
    rgb_h_f, rgb_w_f, depth_h_f, depth_w_f = torch.split(fused, [H, W, H, W], dim=2)

    rgb_w_f = rgb_w_f.permute(0, 1, 3, 2)
    depth_w_f = depth_w_f.permute(0, 1, 3, 2)

    # ================= Coordinate attention weights =================
    rgb_h_weight = torch.sigmoid(self.rgb_h_proj(rgb_h_f))
    rgb_w_weight = torch.sigmoid(self.rgb_w_proj(rgb_w_f))
    depth_h_weight = torch.sigmoid(self.depth_h_proj(depth_h_f))
    depth_w_weight = torch.sigmoid(self.depth_w_proj(depth_w_f))

    # ================= Cross-modal residual interaction =================
    rgb_out = depth * rgb_h_weight * rgb_w_weight + rgb
    depth_out = rgb * depth_h_weight * depth_w_weight + depth

    return [rgb_out, depth_out]
