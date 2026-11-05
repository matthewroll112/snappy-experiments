from pathlib import Path

import argparse
import csv

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.collate import collate_fn
from experiments.common import create_dataloaders, set_seed
from experiments.registry import create_experiment_components


DATASETS = {
  "base": Path("data/snappy-v2"),
  "promptda": Path("data/snappy-v2-promptda"),
  "da2": Path("data/snappy-v2-da2"),
  "lingbot": Path("data/snappy-v2-lingbot")
}

SUBSET_FILES = {
  "water_occlusion": Path("data/subsets/water_occlusion.txt"),
  "overlap": Path("data/subsets/overlap.txt")
}

RESULTS_DIR = Path("results/final")

NUM_CLASSES = 2
CLASS_NAMES = {
  0: "JMA",
  1: "EMA"
}


class DepthAblationDataset(Dataset):
  """
  Wrapper for inference-time depth ablation.

  Modes:
    normal:   original aligned depth
    muted:    depth replaced with zeros
    shuffled: depth taken from a different validation image
  """

  def __init__(self, dataset, mode="normal", seed=13):
    self.dataset = dataset
    self.mode = mode

    if mode not in {"normal", "muted", "shuffled"}:
      raise ValueError(f"Unknown depth mode: {mode}")

    if mode == "shuffled":
      self.depth_indices = self._create_derangement(len(dataset), seed)

  def _create_derangement(self, n, seed):
    if n < 2:
      raise ValueError("Shuffled depth evaluation requires at least two images")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()

    depth_indices = [None] * n

    for i in range(n):
      rgb_index = order[i]
      depth_index = order[(i + 1) % n]
      depth_indices[rgb_index] = depth_index

    if any(index == depth_indices[index] for index in range(n)):
      raise RuntimeError("Depth derangement contains an identity pairing")

    return depth_indices

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    rgb, depth, target = self.dataset[index]

    if self.mode == "muted":
      depth = torch.zeros_like(depth)

    elif self.mode == "shuffled":
      shuffled_index = self.depth_indices[index]
      _, depth, _ = self.dataset[shuffled_index]

    return rgb, depth, target


def load_config(path):
  """Load a validation configuration file."""

  with Path(path).open("r") as file:
    return yaml.safe_load(file)


def load_checkpoint(model, checkpoint_path, device):
  """Load trained model weights."""

  checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
  )

  if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

  model.load_state_dict(checkpoint)

  return model


def load_subset_names(path):
  """Load image filenames for an evaluation subset."""

  if not path.exists():
    raise FileNotFoundError(f"Subset file not found: {path}")

  with path.open("r") as file:
    return {line.strip() for line in file if line.strip()}


def create_subset_loader(loader, subset_name):
  """Create a validation loader containing only selected images."""

  if subset_name == "all":
    return loader

  if subset_name not in SUBSET_FILES:
    raise ValueError(f"Unknown subset: {subset_name}")

  subset_names = load_subset_names(SUBSET_FILES[subset_name])

  indices = [
    index
    for index, image_path in enumerate(loader.dataset.image_paths)
    if image_path.name in subset_names
  ]

  if len(indices) == 0:
    raise RuntimeError(f"No validation images matched subset: {subset_name}")

  subset = Subset(loader.dataset, indices)

  return DataLoader(
    subset,
    batch_size=loader.batch_size,
    shuffle=False,
    num_workers=loader.num_workers,
    pin_memory=True,
    collate_fn=collate_fn
  )


def create_depth_ablation_loader(loader, mode, seed):
  """Create a validation loader with modified depth input."""

  if mode == "normal":
    return loader

  dataset = DepthAblationDataset(
    loader.dataset,
    mode=mode,
    seed=seed
  )

  return DataLoader(
    dataset,
    batch_size=loader.batch_size,
    shuffle=False,
    num_workers=loader.num_workers,
    pin_memory=True,
    collate_fn=collate_fn
  )


def save_results(model_type, results):
  """Append final evaluation results to CSV."""

  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  output_path = RESULTS_DIR / f"{model_type}.csv"

  fieldnames = [
    "name",
    "subset",
    "depth_mode",
    "dataset",
    "modality",
    "fusion_stage",
    "fusion_type",
    "seed",
    "map",
    "map_50",
    "precision",
    "recall",
    "inference_time_ms",
    "num_images"
  ]

  file_exists = output_path.exists()

  with output_path.open("a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    if not file_exists:
      writer.writeheader()

    writer.writerow(results)

  return output_path


def evaluate_model(
  model,
  val_fn,
  loader,
  device,
  name,
  subset_name,
  depth_mode,
  dataset_name,
  modality,
  fusion_stage,
  fusion_type,
  seed
):
  """Run one evaluation condition and save the results."""

  print()
  print(f"Subset:       {subset_name}")
  print(f"Depth mode:   {depth_mode}")
  print(f"Images:       {len(loader.dataset)}")

  metrics = val_fn(
    model=model,
    loader=loader,
    device=device,
    class_names=CLASS_NAMES,
    measure_inference=True
  )

  results = {
    "name": name,
    "subset": subset_name,
    "depth_mode": depth_mode,
    "dataset": dataset_name,
    "modality": modality,
    "fusion_stage": fusion_stage,
    "fusion_type": fusion_type,
    "seed": seed,
    "map": metrics["map"],
    "map_50": metrics["map_50"],
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "inference_time_ms": metrics["inference_time_ms"],
    "num_images": len(loader.dataset)
  }

  print(f"mAP50-95:       {metrics['map']:.4f}")
  print(f"mAP50:          {metrics['map_50']:.4f}")
  print(f"Precision:      {metrics['precision']:.4f}")
  print(f"Recall:         {metrics['recall']:.4f}")
  print(f"Inference time: {metrics['inference_time_ms']:.2f} ms/image")

  return results


def evaluate_config(config, device):
  """Evaluate all models defined in a validation config."""

  model_type = config["model"]
  validation = config.get("validation", {})

  batch_size = validation.get("batch_size", 1)
  num_workers = validation.get("num_workers", 4)
  subsets = validation.get("subsets", ["all"])
  depth_modes = validation.get(
    "depth_modes",
    ["normal", "muted", "shuffled"]
  )

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
      raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
      )

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
    val_fn = components["val_fn"]

    model = load_checkpoint(
      model,
      checkpoint_path,
      device
    )

    model.to(device)
    model.eval()

    evaluation_results = []

    # Condition-specific evaluation.
    # Uses normal aligned depth for RGB-D models.
    for subset_name in subsets:
      subset_loader = create_subset_loader(
        val_loader,
        subset_name
      )

      results = evaluate_model(
        model=model,
        val_fn=val_fn,
        loader=subset_loader,
        device=device,
        name=name,
        subset_name=subset_name,
        depth_mode="normal",
        dataset_name=dataset_name,
        modality=modality,
        fusion_stage=fusion_stage,
        fusion_type=fusion_type,
        seed=seed
      )

      evaluation_results.append(results)

    # Depth-utilisation analysis.
    # Only applicable to RGB-D models and only on the full validation set.
    if modality == "rgbd":
      for depth_mode in depth_modes:
        if depth_mode == "normal":
          continue

        depth_loader = create_depth_ablation_loader(
          val_loader,
          mode=depth_mode,
          seed=seed
        )

        results = evaluate_model(
          model=model,
          val_fn=val_fn,
          loader=depth_loader,
          device=device,
          name=name,
          subset_name="all",
          depth_mode=depth_mode,
          dataset_name=dataset_name,
          modality=modality,
          fusion_stage=fusion_stage,
          fusion_type=fusion_type,
          seed=seed
        )

        evaluation_results.append(results)

    for results in evaluation_results:
      output_path = save_results(
        model_type,
        results
      )

    print()
    print(f"Saved to: {output_path}")


def main():
  parser = argparse.ArgumentParser(
    description="Evaluate final trained models"
  )

  parser.add_argument(
    "configs",
    nargs="+",
    help="Validation YAML configuration files"
  )

  parser.add_argument(
    "--device",
    default="cuda",
    help="Device to use"
  )

  args = parser.parse_args()

  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
      "CUDA requested but is not available"
    )

  device = torch.device(args.device)

  for config_path in args.configs:
    config = load_config(config_path)
    evaluate_config(config, device)


if __name__ == "__main__":
  main()