import torch
from tqdm import tqdm

from ultralytics.utils.ops import non_max_suppression

from src.evaluation import DetectionEvaluator, normalized_cxcywh_to_xyxy


@torch.inference_mode()
def validate_yolo(model, loader, device, class_names, epoch=None, conf_threshold=0.001, iou_threshold=0.7):
  model.eval()

  evaluator = DetectionEvaluator(device, class_names)
  use_amp = device.type == "cuda"

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
      outputs = model(rgb, depth)

    predictions = non_max_suppression(
      outputs,
      conf_thres=conf_threshold,
      iou_thres=iou_threshold,
      nc=len(class_names)
    )

    metric_predictions = []

    for prediction in predictions:
      if len(prediction) == 0:
        metric_predictions.append({
          "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
          "scores": torch.empty(0, dtype=torch.float32, device=device),
          "labels": torch.empty(0, dtype=torch.int64, device=device)
        })

        continue

      metric_predictions.append({
        "boxes": prediction[:, :4],
        "scores": prediction[:, 4],
        "labels": prediction[:, 5].long()
      })

    evaluator.update(metric_predictions, metric_targets)

  return evaluator.compute()