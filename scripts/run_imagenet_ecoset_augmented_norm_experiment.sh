#!/bin/bash
# Run ResNet50 norm experiment on ImageNet-Ecoset136 WITH max augmentations.
# Direct comparison partner: run_imagenet_ecoset_norm_experiment.sh (no augment).
#
# Augmentations: RandomResizedCrop, HorizontalFlip, Rotation ±15°,
#   ColorJitter (mild), GaussianBlur, RandomGrayscale, RandAugment (n=2, m=9),
#   FourierAmplitudeNoise, RandomErasing, LabelSmoothing=0.1
#
# Usage:
#   ./scripts/run_imagenet_ecoset_augmented_norm_experiment.sh
#   NGPUS=1 ./scripts/run_imagenet_ecoset_augmented_norm_experiment.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NGPUS="${NGPUS:-3}"
CONFIG_DIR="configs/imagenet_ecoset_augmented_norm_experiment"
NORMS=(batchnorm layernorm groupnorm rmsnorm derf)

if [ ! -f "configs/ecoset_imagenet_classes.txt" ]; then
    echo "ERROR: configs/ecoset_imagenet_classes.txt not found."
    echo "Run: python scripts/compute_ecoset_overlap.py"
    exit 1
fi

N_CLASSES=$(grep -v '^#' configs/ecoset_imagenet_classes.txt | grep -c '^n' || true)
echo "============================================"
echo "ImageNet-Ecoset136 Augmented Norm Experiment"
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
echo "All imagenet_ecoset136 augmented runs complete!"
echo "============================================"
