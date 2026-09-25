from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DetectionEvaluator:
  """Calculate common object detection metrics using xyxy boxes."""

  def __init__(self, device, class_names):
    self.class_names = class_names

    self.metric = MeanAveragePrecision(
      box_format="xyxy",
      iou_type="bbox",
      class_metrics=True
    ).to(device)

  def update(self, predictions, targets):
    self.metric.update(predictions, targets)

  def compute(self):
    metrics = self.metric.compute()

    results = {
      "map": metrics["map"].item(),
      "map_50": metrics["map_50"].item()
    }

    for class_id, class_map in zip(metrics["classes"], metrics["map_per_class"]):
      class_id = class_id.item()
      results[f"map_{self.class_names[class_id]}"] = class_map.item()

    return results

  def reset(self):
    self.metric.reset()