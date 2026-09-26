from pathlib import Path

import argparse
import traceback

import torch
import yaml

from experiments.common import create_dataloaders, create_optimizer, create_scaler, run_training, set_seed
from experiments.registry import create_experiment_components


DATASETS = {
  "base": Path("data/snappy-v2"),
  "promptda": Path("data/snappy-v2-promptda"),
  "da2": Path("data/snappy-v2-da2"),
  "lingbot": Path("data/snappy-v2-lingbot")
}

RESULTS_DIR = Path("results")
FAILURE_LOG = RESULTS_DIR / "failed_experiments.txt"

NUM_CLASSES = 2
CLASS_NAMES = {
  0: "JMA",
  1: "EMA"
}


def load_config(path):
  """Load an experiment configuration file."""

  with Path(path).open("r") as file:
    return yaml.safe_load(file)


def log_failure(model_type, experiment_name, error):
  """Record a failed experiment."""

  RESULTS_DIR.mkdir(parents=True, exist_ok=True)

  with FAILURE_LOG.open("a") as file:
    file.write(f"{model_type} / {experiment_name}\n")
    file.write(f"{type(error).__name__}: {error}\n")
    file.write(traceback.format_exc())
    file.write("\n" + "=" * 80 + "\n")


def run_config(config, device):
  """Run all experiments defined in a configuration."""

  model_type = config["model"]
  training = config["training"]

  epochs = training["epochs"]
  batch_size = training["batch_size"]
  num_workers = training["num_workers"]
  learning_rate = training["learning_rate"]
  weight_decay = training.get("weight_decay", 0.0)
  seed = training["seed"]

  output_dir = RESULTS_DIR / model_type

  print()
  print("=" * 80)
  print(f"MODEL: {model_type.upper()}")
  print("=" * 80)

  for experiment in config["experiments"]:
    experiment_name = experiment["name"]

    model = None
    criterion = None
    optimizer = None
    scaler = None
    train_loader = None
    val_loader = None

    try:
      modality = experiment["modality"]
      fusion = experiment.get("fusion")
      dataset_name = experiment["dataset"]

      if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

      data_dir = DATASETS[dataset_name]

      print()
      print("-" * 80)
      print(f"Experiment: {experiment_name}")
      print(f"Model:      {model_type}")
      print(f"Modality:   {modality}")
      print(f"Fusion:     {fusion}")
      print(f"Dataset:    {dataset_name}")
      print("-" * 80)

      set_seed(seed)

      train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size,
        num_workers
      )

      components = create_experiment_components(
        model_type=model_type,
        modality=modality,
        fusion=fusion,
        num_classes=NUM_CLASSES,
        device=device
      )

      model = components["model"]
      criterion = components["criterion"]
      train_fn = components["train_fn"]
      val_fn = components["val_fn"]

      optimizer = create_optimizer(model, learning_rate, weight_decay)
      scaler = create_scaler(device)

      run_training(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        train_fn=train_fn,
        val_fn=val_fn,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        class_names=CLASS_NAMES,
        epochs=epochs,
        output_dir=output_dir,
        experiment_name=experiment_name
      )

      print()
      print(f"COMPLETED: {model_type} / {experiment_name}")

    except Exception as error:
      print()
      print("!" * 80)
      print(f"FAILED: {model_type} / {experiment_name}")
      print(f"{type(error).__name__}: {error}")
      print("Continuing to next experiment...")
      print("!" * 80)

      log_failure(model_type, experiment_name, error)

    finally:
      del model
      del criterion
      del optimizer
      del scaler
      del train_loader
      del val_loader

      if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
  parser = argparse.ArgumentParser(description="Run snappy object detection experiments")
  parser.add_argument("configs", nargs="+", help="One or more experiment YAML files")
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(f"Device: {device}")

  for config_path in args.configs:
    config_path = Path(config_path)

    try:
      if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

      print()
      print("#" * 80)
      print(f"CONFIG: {config_path}")
      print("#" * 80)

      config = load_config(config_path)
      run_config(config, device)

    except Exception as error:
      print()
      print("!" * 80)
      print(f"FAILED CONFIG: {config_path}")
      print(f"{type(error).__name__}: {error}")
      print("Continuing to next config...")
      print("!" * 80)

      log_failure("config", str(config_path), error)


if __name__ == "__main__":
  main()