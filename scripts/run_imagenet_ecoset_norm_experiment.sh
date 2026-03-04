#!/bin/bash
# Run ResNet50 norm experiment on ImageNet-Ecoset136 (136 ecoset-overlapping classes).
# Directly comparable to run_ecoset_norm_experiment.sh (same classes, different images).
#
# Requires: configs/ecoset_imagenet_classes.txt (run compute_ecoset_overlap.py first)
# Usage:
#   ./scripts/run_imagenet_ecoset_norm_experiment.sh
#   NGPUS=1 ./scripts/run_imagenet_ecoset_norm_experiment.sh  # single GPU

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NGPUS="${NGPUS:-3}"
CONFIG_DIR="configs/imagenet_ecoset_norm_experiment"
NORMS=(batchnorm layernorm groupnorm rmsnorm derf)

# Check class list exists
if [ ! -f "configs/ecoset_imagenet_classes.txt" ]; then
    echo "ERROR: configs/ecoset_imagenet_classes.txt not found."
    echo "Run: python scripts/compute_ecoset_overlap.py"
    echo "     (requires ecoset to be downloaded at /export/scratch1/home/melle/datasets/ecoset)"
    exit 1
fi

N_CLASSES=$(grep -v '^#' configs/ecoset_imagenet_classes.txt | grep -c '^n' || true)
echo "============================================"
echo "ImageNet-Ecoset136 Norm Experiment"
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
echo "All imagenet_ecoset136 runs complete!"
echo "============================================"
