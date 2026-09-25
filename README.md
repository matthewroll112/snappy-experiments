# Snappy RGB-D Object Detection Experiments

This project focuses on object detection using RGB and RGB-D data, with experiments comparing different object detection architectures and RGB-D fusion strategies.

## Project Structure

```bash
├───checkpoints     # Saved model checkpoints
├───configs         # Model and experiment config files
├───data            # Local dataset dir
├───experiments     # Experimentation files
├───models          # Model-specific implementations
├───results         # Experiment results and metrics
└───src             # Shared datasets, transforms, eval, fusion, and utils...
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

## Running Experiments

Experiments are run from the repository root
For example:
```bash
python experiments/rtdetrv2/rgb.py
```
This ensures that paths and imports remain consistent
