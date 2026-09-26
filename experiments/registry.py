from ultralytics.utils.loss import v8DetectionLoss

from models.efficientdet import RGBDEfficientDetDetector, RGBEfficientDetDetector
from models.efficientdet.engine import EfficientDetCriterion
from models.efficientdet.engine import train_one_epoch as train_efficientdet
from models.efficientdet.engine import validate_efficientdet

from models.rtdetrv2 import RGBDRTDETRDetector, RGBRTDETRDetector
from models.rtdetrv2.engine import train_one_epoch as train_rtdetr
from models.rtdetrv2.engine import validate_rtdetr

from models.yolo import RGBDYOLODetector, RGBYOLODetector
from models.yolo.engine import train_one_epoch as train_yolo
from models.yolo.engine import validate_yolo


def create_yolo_components(modality, fusion, num_classes, device):
  """Create YOLO model, criterion and engine functions."""

  if modality == "rgb":
    model = RGBYOLODetector(num_classes).to(device)

  elif modality == "rgbd":
    if fusion is None:
      raise ValueError("RGB-D YOLO experiment requires a fusion type")

    model = RGBDYOLODetector(num_classes, fusion).to(device)

  else:
    raise ValueError(f"Unknown modality: {modality}")

  criterion = v8DetectionLoss(model)

  return {
    "model": model,
    "criterion": criterion,
    "train_fn": train_yolo,
    "val_fn": validate_yolo
  }


def create_rtdetr_components(modality, fusion, num_classes, device):
  """Create RT-DETR model and engine functions."""

  if modality == "rgb":
    model = RGBRTDETRDetector(num_classes).to(device)

  elif modality == "rgbd":
    if fusion is None:
      raise ValueError("RGB-D RT-DETR experiment requires a fusion type")

    model = RGBDRTDETRDetector(num_classes, fusion).to(device)

  else:
    raise ValueError(f"Unknown modality: {modality}")

  return {
    "model": model,
    "criterion": None,
    "train_fn": train_rtdetr,
    "val_fn": validate_rtdetr
  }


def create_efficientdet_components(modality, fusion, num_classes, device):
  """Create EfficientDet model, criterion and engine functions."""

  if modality == "rgb":
    model = RGBEfficientDetDetector(num_classes).to(device)

  elif modality == "rgbd":
    if fusion is None:
      raise ValueError("RGB-D EfficientDet experiment requires a fusion type")

    model = RGBDEfficientDetDetector(num_classes, fusion).to(device)

  else:
    raise ValueError(f"Unknown modality: {modality}")

  criterion = EfficientDetCriterion(model.model.config).to(device)

  return {
    "model": model,
    "criterion": criterion,
    "train_fn": train_efficientdet,
    "val_fn": validate_efficientdet
  }


def create_experiment_components(model_type, modality, fusion, num_classes, device):
  """Create model-specific components for an experiment."""

  if model_type == "yolo":
    return create_yolo_components(modality, fusion, num_classes, device)

  if model_type == "rtdetrv2":
    return create_rtdetr_components(modality, fusion, num_classes, device)

  if model_type == "efficientdet":
    return create_efficientdet_components(modality, fusion, num_classes, device)

  raise ValueError(f"Unknown model type: {model_type}")