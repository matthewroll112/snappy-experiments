import torch

from tqdm import tqdm

from src.evaluation import DetectionEvaluator, normalized_cxcywh_to_xyxy

from transformers import RTDetrImageProcessor


@torch.inference_mode()
def validate_rtdetr(model, loader, device, class_names, epoch=None):
  model.eval()

  evaluator = DetectionEvaluator(device, class_names)

  processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")

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

    outputs = model(rgb, depth)

    target_sizes = torch.tensor([[image_height, image_width]] * rgb.shape[0], device=device)

    predictions = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.001)

    evaluator.update(predictions, metric_targets)

  return evaluator.compute()