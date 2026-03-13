# visionCNN

A research framework for systematically studying the effects of normalization layers on vision model training and out-of-distribution (OOD) robustness. Supports multiple architectures, normalization strategies, datasets, and integrated OOD evaluation — all controlled via YAML configs.

## Features

- **6 normalization layers**: BatchNorm, LayerNorm, GroupNorm, RMSNorm, Derf, NoNorm (+ Weight Standardized Conv)
- **5 architectures**: ResNet, VGG, ViT, ConvNeXtV2, SimpleCNN — all with swappable norm layers
- **OOD evaluation**: ImageNet-R, ImageNet-A, ImageNet-Sketch, ImageNet-C (16 corruptions × 5 severities)
- **Multi-GPU training**: DDP via `torchrun`, SLURM support for clusters
- **Flexible data pipeline**: PyTorch, NVIDIA DALI, RAM-cached backends; GPU-side transforms
- **Experiment tracking**: Weights & Biases integration with local CSV/log fallback

## Project Structure

```
visionCNN/
├── src/                        # Main package (import as `from src.models import ...`)
│   ├── models/
│   │   ├── factory.py          # Model registry and builder
│   │   ├── norms.py            # All normalization layer implementations
│   │   ├── resnet.py           # ResNet with configurable norms
│   │   ├── vgg.py              # VGG with configurable norms
│   │   ├── vit.py              # Vision Transformer with configurable norms
│   │   ├── convnextv2.py       # ConvNeXtV2 variants
│   │   └── simple_cnn.py       # Simple CNN for CIFAR
│   ├── datasets/
│   │   ├── build.py            # Dataset/dataloader factory, augmentation pipeline
│   │   ├── imagenet_ecoset.py  # ImageNet-Ecoset136 subset (136 classes)
│   │   ├── imagenet100.py      # ImageNet-100 subset
│   │   ├── ecoset.py           # Ecoset dataset loader
│   │   ├── ood_datasets.py     # OOD dataset wrappers (R, A, Sketch, C, SIN)
│   │   ├── dali_imagenet.py    # NVIDIA DALI backend
│   │   └── gpu_transforms.py   # GPU-side augmentations
│   ├── evaluation/
│   │   ├── ood_eval.py         # OOD evaluation functions
│   │   ├── class_mapping.py    # Dataset-to-ImageNet synset mapping
│   │   ├── error_consistency.py
│   │   └── shape_bias.py
│   └── utils/
│       ├── distributed.py      # DistributedManager (DDP abstraction)
│       ├── machine.py          # Machine-specific config resolution
│       └── seed.py             # Reproducible seeding
├── configs/
│   ├── machines/               # Per-machine overrides (guppy.yaml, etc.)
│   ├── imagenet_ecoset_norm_experiment/   # ResNet50 norm comparison
│   ├── ecoset_norm_experiment/            # Ecoset norm comparison
│   └── ...                     # Other experiment configs
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── launch_guppy.sh         # Multi-GPU launcher (local)
│   ├── launch_snellius.sh      # SLURM cluster launcher
│   ├── evaluate_ood_full.py    # Post-training OOD evaluation
│   └── ...                     # Analysis and utility scripts
├── logs/                       # Training outputs (checkpoints, logs, history)
├── results/                    # Analysis results
└── pyproject.toml
```

## Installation

```bash
# Create environment
micromamba create -n visioncnn python=3.11
micromamba activate visioncnn

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install timm wandb pyyaml nvidia-dali-cuda120 imagecorruptions

# Install package in editable mode
pip install -e .
```

## Usage

### Training

**Single GPU:**
```bash
python scripts/train.py --config configs/imagenet_ecoset_norm_experiment/batchnorm.yaml
```

**Multi-GPU (local):**
```bash
./scripts/launch_guppy.sh configs/imagenet_ecoset_norm_experiment/batchnorm.yaml
```

**Resume from checkpoint:**
```bash
./scripts/launch_guppy.sh configs/imagenet_ecoset_norm_experiment/batchnorm.yaml \
  --resume logs/resnet50_batchnorm_imagenet_ecoset/last.pt
```

**Override config values from CLI:**
```bash
python scripts/train.py --config configs/batchnorm.yaml \
  --set model.norm_layer=groupnorm training.epochs=100 data.num_workers=8
```

### OOD Evaluation

OOD evaluation runs automatically during training when `ood_eval.enabled: true`. For full post-training evaluation across all corruptions and severities:

```bash
python scripts/evaluate_ood_full.py
```

### Running All Norm Experiments

```bash
./scripts/run_imagenet_ecoset_norm_experiment.sh
```

## Configuration

Configs are YAML files with the following structure:

```yaml
experiment_name: resnet50_batchnorm_imagenet_ecoset

model:
  name: resnet50_custom       # Model architecture
  num_classes: 136
  norm_layer: batchnorm       # Normalization layer

data:
  dataset: imagenet_ecoset136 # Dataset
  img_size: 224
  backend: pytorch            # pytorch | dali | cached
  gpu_transforms: true

training:
  epochs: 50
  batch_size: 64
  lr: 0.001
  optimizer: adamw            # adamw | sgd | adam
  scheduler: cosine           # cosine | step | multistep | none
  label_smoothing: 0.1
  use_amp: true
  channels_last: true
  seed: 42

ood_eval:
  enabled: true
  source_dataset: imagenet_ecoset136
  imagenet_c:
    during_training: true

wandb:
  enabled: true
  project: norm-comparison
```

**Machine overrides** (`configs/machines/*.yaml`) automatically apply based on hostname, setting paths, worker counts, and W&B tags.

## Normalization Layers

| Name | Config value | Description |
|------|-------------|-------------|
| BatchNorm | `batchnorm` | Standard batch normalization |
| LayerNorm | `layernorm` | Per-sample normalization (GroupNorm with 1 group) |
| GroupNorm | `groupnorm` | Group normalization (32 groups) |
| RMSNorm | `rmsnorm` | Root mean square normalization (no mean centering) |
| Derf | `derf` | Dynamic ERF-based normalization using error function |
| NoNorm | `nonorm` | Identity (no-op) for ablation |
| NoNorm + WS | `nonorm_ws` | Weight Standardized Conv without normalization |

## Models

| Name | Config value | Architecture |
|------|-------------|-------------|
| ResNet-18 | `resnet_small` | Custom ResNet-18 with configurable norms |
| ResNet-34 | `resnet_medium` | Custom ResNet-34 with configurable norms |
| ResNet-50 | `resnet50_custom` | Custom ResNet-50 with configurable norms |
| VGG-11 | `vgg_small` | VGG-11 with configurable norms and GAP head |
| VGG-16 | `vgg_medium` | VGG-16 with configurable norms and GAP head |
| ViT-S/16 | `vit_small` | Vision Transformer Small (patch 16) |
| ConvNeXtV2 | `convnext_small` | ConvNeXtV2 Small (ImageNet) |
| SimpleCNN | `simple_cnn` | Basic 3-layer CNN (CIFAR) |

## Datasets

**Training datasets:**
- `imagenet` — Full ImageNet-1K (1000 classes)
- `imagenet_ecoset136` — 136 ImageNet classes overlapping with ecoset
- `imagenet100` — 100-class ImageNet subset
- `ecoset136` — 136 ecoset classes (size-controlled, 1300/class)
- `cifar10`, `cifar100`, `svhn`

**OOD evaluation datasets:**
- ImageNet-R (renditions), ImageNet-A (adversarial), ImageNet-Sketch
- ImageNet-C (16 corruption types × 5 severities)
- Stylized ImageNet (SIN)

## Training Outputs

Each experiment saves to `logs/{experiment_name}/`:
- `best.pt`, `last.pt` — Model checkpoints
- `history.csv` — Per-epoch metrics (train/val loss & accuracy, learning rate)
- `train.log` — Full training log
- `config.yaml` — Resolved configuration
- `ood_predictions/` — Saved OOD predictions (if enabled)
