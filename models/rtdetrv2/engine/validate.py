import torch

from tqdm import tqdm

from src.evaluation import DetectionEvaluator, normalized_cxcywh_to_xyxy


@torch.inference_mode()
def validate_rtdetr(model, loader, device, class_names, epoch=None):
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

    probabilities = outputs.logits.sigmoid()
    predictions = []

    for scores, boxes in zip(probabilities, outputs.pred_boxes):
      confidence, labels = scores.max(dim=-1)

      predictions.append({
        "boxes": normalized_cxcywh_to_xyxy(boxes, image_height, image_width),
        "scores": confidence,
        "labels": labels
      })

    evaluator.update(predictions, metric_targets)

  return evaluator.compute()