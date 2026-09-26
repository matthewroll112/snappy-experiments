# Snappy RGB-D Object Detection Experiments

This project focuses on object detection using RGB and RGB-D data, with experiments comparing different object detection architectures and RGB-D fusion strategies.

## Models

This project currently supports the following models for RGB and RGB-D object detection:
- YOLOv8n
- RT-DETRv2
- EfficientDet-D0

## Project Structure

```bash
├─ configs/              # YAML experiment configurations
├─ experiments/          # Shared experiment runner and model registry
├─ models/               # Model-specific implementations
│   ├─ yolo/
│   ├─ rtdetr/
│   └─ efficientdet/
├─ results/              # Experiment CSV files and best checkpoints
├─ src/                  # Shared project code
│   ├─ data/             # Dataset loading, transforms etc.
│   ├─ evaluation/       # Shared detection evaluation
│   ├─ fusion/           # Shared RGB-D fusion modules
└─ run_experiments.py    # Main experiment entry point
```

## Dataset Format

The experiments assume use of four-page TIFF images containing...
1. Red
2. Green
3. Blue
4. Depth

And that labels use YOLO-style normalized bounding boxes:

```text
class cx cy width height
```

## Environment Setup

>[!IMPORTANT]
>Using a virtual environment such as Conda is highly recommended when installing packages but is not required

### 1. Clone the Repository

```bash
git clone https://github.com/matthewroll112/snappy-experiments.git
cd snappy-experiments
```

### 2. Create a Conda Environment and Activate

```bash
conda create -n snappy-experiments python=3.12
conda activate snappy-experiments
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify PyTorch/CUDA

Both can be checked with:
```bash
python -c "import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())"
```
A different PyTorch build may need to be installed depending on the CUDA version available on the system.

PyTorch installation options can be found at:
https://pytorch.org/get-started/previous-versions/

## Experiment Configs

Experiments are defined using YAML files in 'configs/'

A configuration specifies the model arhitecture, training settings, and experiments to run. e.g.
```yaml
model: yolo

training:
  epochs: 100
  batch_size: 4
  num_workers: 4
  learning_rate: 0.0001
  weight_decay: 0.0
  seed: 13

experiments:
  - name: rgb
    modality: rgb
    fusion: null
    dataset: promptda
  - name: mid_promptda
    modality: rgbd
    fusion: mid
    dataset: promptda
```

## Running Experiments

Experiments are run from the repository root using `run_experiments.py`

A single configuration file can be run with:
```bash
python run_experiments.py configs/yolo/yolo.yaml
```

Multiple config files can also be supplied...
```bash
python run_experiments.py configs/yolo/yolo.yaml configs/rtdetrv2/rtdetrv2.yaml ...
```

## Experiment Outputs

Each experiment produces:
- 1 csv file containing training history
- 1 checkpioint (.pt) file contraining the model with the best validation mAP
