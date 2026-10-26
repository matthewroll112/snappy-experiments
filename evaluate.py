from pathlib import Path

import argparse
import csv

import torch
import yaml

from experiments.common import create_dataloaders, set_seed
from experiments.registry import create_experiment_components


DATASETS = {
  "base": Path("data/snappy-v2"),
  "promptda": Path("data/snappy-v2-promptda"),
  "da2": Path("data/snappy-v2-da2"),
  "lingbot": Path("data/snappy-v2-lingbot")
}

RESULTS_DIR = Path("results/final")

NUM_CLASSES = 2
CLASS_NAMES = {
  0: "JMA",
  1: "EMA"
}


def load_config(path):
  """Load a validation configuration file."""

  with Path(path).open("r") as file:
    return yaml.safe_load(file)


def load_checkpoint(model, checkpoint_path, device):
  """Load trained model weights."""

  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

  if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

  model.load_state_dict(checkpoint)

  return model


def save_results(model_type, results):
  """Save final evaluation results."""

  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  output_path = RESULTS_DIR / f"{model_type}.csv"

  fieldnames = [
    "name",
    "dataset",
    "modality",
    "fusion_stage",
    "fusion_type",
    "seed",
    "map",
    "map_50",
    "precision",
    "recall",
    "inference_time_ms"
  ]

  file_exists = output_path.exists()

  with output_path.open("a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    if not file_exists:
      writer.writeheader()

    writer.writerow(results)

  return output_path


def evaluate_config(config, device):
  """Evaluate all models defined in a validation config."""

  model_type = config["model"]
  validation = config.get("validation", {})

  batch_size = validation.get("batch_size", 1)
  num_workers = validation.get("num_workers", 4)

  print()
  print("=" * 80)
  print(f"MODEL: {model_type.upper()}")
  print("=" * 80)

  for experiment in config["experiments"]:
    name = experiment["name"]
    dataset_name = experiment["dataset"]
    modality = experiment["modality"]
    fusion_stage = experiment.get("fusion_stage")
    fusion_type = experiment.get("fusion_type")
    seed = experiment["seed"]
    checkpoint_path = Path(experiment["checkpoint"])

    if dataset_name not in DATASETS:
      raise ValueError(f"Unknown dataset: {dataset_name}")

    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print()
    print("-" * 80)
    print(f"Experiment:   {name}")
    print(f"Dataset:      {dataset_name}")
    print(f"Modality:     {modality}")
    print(f"Fusion stage: {fusion_stage}")
    print(f"Fusion type:  {fusion_type}")
    print(f"Seed:         {seed}")
    print(f"Checkpoint:   {checkpoint_path}")
    print("-" * 80)

    set_seed(seed)

    _, val_loader = create_dataloaders(
      DATASETS[dataset_name],
      batch_size=batch_size,
      num_workers=num_workers
    )

    components = create_experiment_components(
      model_type=model_type,
      modality=modality,
      fusion_stage=fusion_stage,
      fusion_type=fusion_type,
      num_classes=NUM_CLASSES,
      device=device
    )

    model = components["model"]
    val_fn = components["validate"]

    model = load_checkpoint(model, checkpoint_path, device)
    model.to(device)

    metrics = val_fn(
      model=model,
      loader=val_loader,
      device=device,
      class_names=CLASS_NAMES,
      measure_inference=True
    )

    results = {
      "name": name,
      "dataset": dataset_name,
      "modality": modality,
      "fusion_stage": fusion_stage,
      "fusion_type": fusion_type,
      "seed": seed,
      "map": metrics["map"],
      "map_50": metrics["map_50"],
      "precision": metrics["precision"],
      "recall": metrics["recall"],
      "inference_time_ms": metrics["inference_time_ms"]
    }

    print(f"mAP50-95:       {metrics['map']:.4f}")
    print(f"mAP50:          {metrics['map_50']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"Inference time: {metrics['inference_time_ms']:.2f} ms/image")

    output_path = save_results(model_type, results)
    print(f"Saved to:       {output_path}")


def main():
  parser = argparse.ArgumentParser(description="Evaluate final trained models")
  parser.add_argument("configs", nargs="+", help="Validation YAML configuration files")
  parser.add_argument("--device", default="cuda", help="Device to use")

  args = parser.parse_args()

  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but is not available")

  device = torch.device(args.device)

  for config_path in args.configs:
    config = load_config(config_path)
    evaluate_config(config, device)


if __name__ == "__main__":
  main()