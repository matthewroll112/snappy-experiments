import torch.nn as nn

from effdet.anchors import Anchors, AnchorLabeler
from effdet.loss import DetectionLoss


class EfficientDetCriterion(nn.Module):
  """Native EfficientDet anchor assignment and detection loss"""

  def __init__(self, config):
    super().__init__()

    self.anchors = Anchors.from_config(config)
    self.anchor_labeler = AnchorLabeler(
      self.anchors,
      config.num_classes,
      match_threshold=0.5
    )
    self.loss_fn = DetectionLoss(config)

  def forward(self, class_outputs, box_outputs, targets):
    cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
      targets["bbox"],
      targets["cls"]
    )

    return self.loss_fn(
      class_outputs,
      box_outputs,
      cls_targets,
      box_targets,
      num_positives
    )