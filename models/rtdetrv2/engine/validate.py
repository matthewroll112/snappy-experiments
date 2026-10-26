import time

import torch
from tqdm import tqdm
from transformers import RTDetrImageProcessor

from src.evaluation import DetectionEvaluator, normalized_cxcywh_to_xyxy


@torch.inference_mode()
def validate_rtdetr(model, loader, device, class_names, epoch=None, measure_inference=False):
  model.eval()

  evaluator = DetectionEvaluator(device, class_names)
  processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")

  progress_bar = tqdm(
    loader,
    desc=f"Val {epoch:03d}" if epoch is not None else "Validation",
    leave=False,
    dynamic_ncols=True
  )

  total_inference_time = 0.0
  total_images = 0

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

    if measure_inference and device.type == "cuda":
      torch.cuda.synchronize()

    start_time = time.perf_counter()
    outputs = model(rgb, depth)

    if measure_inference and device.type == "cuda":
      torch.cuda.synchronize()

    if measure_inference:
      total_inference_time += time.perf_counter() - start_time
      total_images += rgb.shape[0]

    target_sizes = torch.tensor([[image_height, image_width]] * rgb.shape[0], device=device)

    predictions = processor.post_process_object_detection(
      outputs,
      target_sizes=target_sizes,
      threshold=0.001
    )

    evaluator.update(predictions, metric_targets)

  results = evaluator.compute()

  if measure_inference:
    results["inference_time_ms"] = total_inference_time / total_images * 1000

  return results