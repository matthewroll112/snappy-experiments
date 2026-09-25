import torch
import torch.nn as nn
import torch.nn.functional as F

class GCFFM(nn.Module):
  """
  Gated Cross-Attention Feature Fusion Module (Zhao et al, 2023)

  Input:
    x = [rgb, depth]:
      rgb: [B, C, H, W]
      depth: [B, C, H, W]

  Output:
    fused: [B, C, H, W]
  """
  def __init__(self, channels: int, attention_reduction: int = 8):
    super().__init__()

    attn_channels = max(channels // attention_reduction, 1)

    # RGB Q, K, V
    self.rgb_q = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
    self.rgb_k = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
    self.rgb_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    # Depth Q, K, V
    self.depth_q = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
    self.depth_k = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
    self.depth_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    # Gated fusion
    self.gate = nn.Conv2d(2 * channels, channels, kernel_size=1)

    # Final fusion
    self.fusion = nn.Sequential(
      nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
      nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
      nn.ReLU(),
      nn.Conv2d(channels, channels, kernel_size=1, bias=False),
    )

    self.skip = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)

  def forward(self, x):
    rgb, depth = x

    if rgb.shape != depth.shape:
      raise ValueError(f"Need RGB and Depth features with identical shapes: rgb{rgb.shape} and depth{depth.shape}")

    B, C, H, W = rgb.shape
    N = H * W

    # ================= Q, K, V projections =================
    rgb_q = self.rgb_q(rgb).flatten(2)
    rgb_k = self.rgb_k(rgb).flatten(2)
    rgb_v = self.rgb_v(rgb).flatten(2)

    depth_q = self.depth_q(depth).flatten(2)
    depth_k = self.depth_k(depth).flatten(2)
    depth_v = self.depth_v(depth).flatten(2)

    # ================= Spatial attention maps =================
    depth_attention = torch.bmm(depth_q.transpose(1, 2), depth_k)
    depth_attention = F.softmax(depth_attention, dim=1)

    rgb_attention = torch.bmm(rgb_q.transpose(1, 2), rgb_k)
    rgb_attention = F.softmax(rgb_attention, dim=1)

    # ================= Cross guided feature enhancement =================
    rgb_enhanced = torch.bmm(rgb_v, depth_attention)
    depth_enhanced = torch.bmm(depth_v, rgb_attention)

    rgb_enhanced = rgb_enhanced.view(B, C, H, W)
    depth_enhanced = depth_enhanced.view(B, C, H, W)

    Br = rgb_enhanced + rgb
    Bd = depth_enhanced + depth

    # ================= Gated fusion =================
    combined = torch.cat([Br, Bd], dim=1)

    Gr = torch.sigmoid(self.gate(combined))
    Gd = 1.0 - Gr

    # Weighted complimentary fusion
    weighted = Br * Gr + Bd * Gd
    product = Br * Bd

    # ================= Final fusion =================
    fusion_input = torch.cat([weighted, product], dim=1)

    out = self.fusion(fusion_input)
    out = out + self.skip(fusion_input)

    return out