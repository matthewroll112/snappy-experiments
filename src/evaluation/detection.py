import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DetectionEvaluator:
  """Calculate common object detection metrics using xyxy boxes."""

  def __init__(self, device, class_names, confidence_threshold=0.05, iou_threshold=0.5):
    self.class_names = class_names
    self.confidence_threshold = confidence_threshold
    self.iou_threshold = iou_threshold

    self.metric = MeanAveragePrecision(
      box_format="xyxy",
      iou_type="bbox",
      class_metrics=True
    ).to(device)

    self.true_positives = 0
    self.false_positives = 0
    self.false_negatives = 0

  def update(self, predictions, targets):
    self.metric.update(predictions, targets)

    for prediction, target in zip(predictions, targets):
      tp, fp, fn = self._match_predictions(prediction, target)

      self.true_positives += tp
      self.false_positives += fp
      self.false_negatives += fn

  def compute(self):
    metrics = self.metric.compute()

    precision = (
      self.true_positives / (self.true_positives + self.false_positives)
      if self.true_positives + self.false_positives > 0
      else 0.0
    )

    recall = (
      self.true_positives / (self.true_positives + self.false_negatives)
      if self.true_positives + self.false_negatives > 0
      else 0.0
    )

    results = {
      "map": metrics["map"].item(),
      "map_50": metrics["map_50"].item(),
      "precision": precision,
      "recall": recall
    }

    for class_id, class_map in zip(metrics["classes"], metrics["map_per_class"]):
      class_id = class_id.item()
      results[f"map_{self.class_names[class_id]}"] = class_map.item()

    return results

  def reset(self):
    self.metric.reset()

    self.true_positives = 0
    self.false_positives = 0
    self.false_negatives = 0

  def _match_predictions(self, prediction, target):
    pred_scores = prediction["scores"]
    keep = pred_scores >= self.confidence_threshold

    pred_boxes = prediction["boxes"][keep]
    pred_scores = pred_scores[keep]
    pred_labels = prediction["labels"][keep]

    target_boxes = target["boxes"]
    target_labels = target["labels"]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for class_id in self.class_names:
      class_pred_mask = pred_labels == class_id
      class_target_mask = target_labels == class_id

      class_pred_boxes = pred_boxes[class_pred_mask]
      class_pred_scores = pred_scores[class_pred_mask]
      class_target_boxes = target_boxes[class_target_mask]

      if len(class_pred_boxes) == 0:
        false_negatives += len(class_target_boxes)
        continue

      if len(class_target_boxes) == 0:
        false_positives += len(class_pred_boxes)
        continue

      order = torch.argsort(class_pred_scores, descending=True)
      class_pred_boxes = class_pred_boxes[order]

      ious = self._box_iou(class_pred_boxes, class_target_boxes)
      matched_targets = set()

      for pred_index in range(len(class_pred_boxes)):
        best_iou, best_target = torch.max(ious[pred_index], dim=0)
        target_index = best_target.item()

        if best_iou.item() >= self.iou_threshold and target_index not in matched_targets:
          true_positives += 1
          matched_targets.add(target_index)
        else:
          false_positives += 1

      false_negatives += len(class_target_boxes) - len(matched_targets)

    return true_positives, false_positives, false_negatives

  @staticmethod
  def _box_iou(boxes1, boxes2):
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - intersection

    return intersection / union.clamp(min=1e-6)