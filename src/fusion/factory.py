from .mlp import MLPFusion
from .conv import ConvFusion
from .gated import GatedFusion
from .addition import AdditionFusion

def create_fusion(fusion_type, channels):
  if fusion_type == "conv":
    return ConvFusion(channels)

  if fusion_type == "add":
    return AdditionFusion(channels)

  if fusion_type == "gated":
    return GatedFusion(channels)

  if fusion_type == "mlp":
    return MLPFusion(channels)

  raise ValueError(f"Unknown fusion type: {fusion_type}")