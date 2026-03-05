#!/bin/bash
# Run ConvNeXt-Small norm experiment on ImageNet-Ecoset136 (no augmentation).
# Direct comparison to ResNet50 norm experiment (same dataset, different architecture).
#
# Norms: batchnorm, layernorm, groupnorm, rmsnorm, derf
# Usage:
#   ./scripts/run_imagenet_ecoset_convnext_norm_experiment.sh
#   NGPUS=1 ./scripts/run_imagenet_ecoset_convnext_norm_experiment.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NGPUS="${NGPUS:-3}"
CONFIG_DIR="configs/imagenet_ecoset_convnext_norm_experiment"
NORMS=(batchnorm layernorm groupnorm rmsnorm derf)

if [ ! -f "configs/ecoset_imagenet_classes.txt" ]; then
    echo "ERROR: configs/ecoset_imagenet_classes.txt not found."
    echo "Run: python scripts/compute_ecoset_overlap.py"
    exit 1
fi

N_CLASSES=$(grep -v '^#' configs/ecoset_imagenet_classes.txt | grep -c '^n' || true)
echo "============================================"
echo "ImageNet-Ecoset136 ConvNeXt Norm Experiment"
echo "Classes: $N_CLASSES"
echo "GPUs: $NGPUS"
echo "============================================"
echo ""

eval "$(micromamba shell hook -s bash)"
micromamba activate visioncnn

for norm in "${NORMS[@]}"; do
    config="$CONFIG_DIR/${norm}.yaml"
    echo "Starting: ${norm}"
    echo "Config: ${config}"
    echo "--------------------------------------------"

    torchrun --nproc_per_node="$NGPUS" \
        scripts/train.py \
        --config "$config" \
        --set data.root=/export/scratch1/home/melle/datasets/imagenet

    echo ""
done

echo "============================================"
echo "All convnext imagenet_ecoset136 runs complete!"
echo "============================================"
