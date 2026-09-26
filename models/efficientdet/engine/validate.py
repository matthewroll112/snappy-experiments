import torch
from tqdm import tqdm

from effdet.anchors import Anchors
from effdet.bench import _batch_detection, _post_process

from src.evaluation import DetectionEvaluator, normalized_cxcywh_to_xyxy


@torch.inference_mode()
def validate_efficientdet(model, loader, device, class_names, epoch=None):
  model.eval()

  evaluator = DetectionEvaluator(device, class_names)
  use_amp = device.type == "cuda"

  config = model.model.config
  anchors = Anchors.from_config(config).to(device)

  progress_bar = tqdm(
    loader,
    desc=f"Val {epoch:03d}" if epoch is not None else "Validation",
    leave=False,
    dynamic_ncols=True
  )

  for rgb, depth, targets in progress_bar:
    rgb = rgb.to(device, non_blocking=True)
    depth = depth.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]

    image_height, image_width = rgb.shape[-2:]

    metric_targets = [
      {
        "boxes": normalized_cxcywh_to_xyxy(target["boxes"], image_height, image_width),
        "labels": target["labels"]
      }
      for target in targets
    ]

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
      class_outputs, box_outputs = model(rgb, depth)

    class_outputs, box_outputs, indices, classes = _post_process(
      class_outputs,
      box_outputs,
      num_levels=config.num_levels,
      num_classes=config.num_classes,
      max_detection_points=config.max_detection_points
    )

    detections = _batch_detection(
      rgb.shape[0],
      class_outputs,
      box_outputs,
      anchors.boxes,
      indices,
      classes,
      None,
      None,
      max_det_per_image=config.max_det_per_image,
      soft_nms=config.soft_nms
    )

    metric_predictions = []

    for detection in detections:
      valid = detection[:, 4] > 0
      detection = detection[valid]

      if len(detection) == 0:
        metric_predictions.append({
          "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
          "scores": torch.empty(0, dtype=torch.float32, device=device),
          "labels": torch.empty(0, dtype=torch.int64, device=device)
        })

        continue

      metric_predictions.append({
        "boxes": detection[:, :4],
        "scores": detection[:, 4],
        "labels": detection[:, 5].long() - 1
      })

    evaluator.update(metric_predictions, metric_targets)

  return evaluator.compute()