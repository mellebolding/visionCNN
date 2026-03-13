# CLAUDE.md

## Project Overview

Research framework for studying normalization layer effects on vision model training and OOD robustness. Compares BatchNorm, LayerNorm, GroupNorm, RMSNorm, Derf, and NoNorm across ResNet, VGG, ViT, and ConvNeXtV2 architectures.

## Package Structure

Installed editable (`pip install -e .`). The package name is `src`:
```python
from src.models import factory, norms, resnet, vgg, vit, convnextv2
from src.datasets import build, imagenet_ecoset, ood_datasets
from src.evaluation import ood_eval, class_mapping
from src.utils import distributed, machine, seed
```

## Environment

- Package manager: **micromamba** (not conda)
- Env name: `visioncnn`, Python 3.11, PyTorch 2.5.1+cu121
- Activate: `eval "$(micromamba shell hook -s bash)" && micromamba activate visioncnn`

## Key Commands

```bash
# Training (multi-GPU)
./scripts/launch_guppy.sh configs/imagenet_ecoset_norm_experiment/batchnorm.yaml

# Training (single GPU)
python scripts/train.py --config configs/<config>.yaml

# Full OOD evaluation (post-training)
python scripts/evaluate_ood_full.py

# Run all norm experiments
./scripts/run_imagenet_ecoset_norm_experiment.sh
```

## Config System

- Experiment configs: `configs/<experiment_group>/<norm>.yaml`
- Machine overrides: `configs/machines/guppy.yaml` (auto-detected by hostname)
- CLI overrides: `--set model.norm_layer=groupnorm training.epochs=100`
- Machine config resolves paths and sets defaults; experiment config overrides machine defaults

## Key Files

| Purpose | Path |
|---------|------|
| Training entry point | `scripts/train.py` |
| Model builder | `src/models/factory.py` |
| Norm implementations | `src/models/norms.py` |
| Dataset/dataloader factory | `src/datasets/build.py` |
| OOD evaluation | `src/evaluation/ood_eval.py` |
| Class mapping (OOD filtering) | `src/evaluation/class_mapping.py` |
| DDP abstraction | `src/utils/distributed.py` |
| Machine config resolver | `src/utils/machine.py` |
| Launch script (guppy) | `scripts/launch_guppy.sh` |

## Code Conventions

- Models accept `norm_layer` (string) or `norm_type` depending on architecture; resolved in `factory.py`
- `get_norm_layer(name)` in `norms.py` maps string names to norm classes
- Datasets use a factory pattern: `build_dataset()` and `build_dataloader()` in `datasets/build.py`
- Distributed training abstracted via `DistributedManager` — works for single-GPU, torchrun, and SLURM
- OOD datasets are filtered to only evaluate classes the model was trained on, via `ClassMapping`
- Training outputs go to `logs/{experiment_name}/` (checkpoints, history.csv, train.log)
- GPU transforms (normalization, random erasing) handled by `GPUTransforms` module to reduce CPU bottleneck

## Data Locations (guppy)

- ImageNet: `/export/scratch1/home/melle/datasets/imagenet`
- ImageNet-R: `/export/scratch1/home/melle/datasets/imagenet-r`
- ImageNet-A: `/export/scratch1/home/melle/datasets/imagenet-a`
- ImageNet-Sketch: `/export/scratch1/home/melle/datasets/imagenet-sketch/data/sketch/`
- Ecoset class list: `configs/ecoset_imagenet_classes.txt`

## Verification

To verify changes work end-to-end, run a short training:
```bash
python scripts/train.py --config configs/imagenet_ecoset_norm_experiment/batchnorm.yaml \
  --set training.epochs=1 wandb.enabled=false
```
