from typing import Literal

FusionType = Literal["conv", "add", "gated", "mlp"]
FusionStage = Literal["early", "mid", "late", "late_partial", "cafim_gcffm"]