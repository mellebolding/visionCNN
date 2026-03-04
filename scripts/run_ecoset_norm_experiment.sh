#!/bin/bash
# Run ResNet50 norm experiment on Ecoset136 (size-controlled: 1300 images/class).
# Directly comparable to run_imagenet_ecoset_norm_experiment.sh.
#
# Requires:
#   - configs/ecoset_imagenet_classes.txt (run compute_ecoset_overlap.py first)
#   - Ecoset downloaded at /export/scratch1/home/melle/datasets/ecoset
#     (run scripts/download_ecoset.sh first)
#
# Usage:
#   ./scripts/run_ecoset_norm_experiment.sh
#   ECOSET_ROOT=/path/to/ecoset ./scripts/run_ecoset_norm_experiment.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NGPUS="${NGPUS:-3}"
ECOSET_ROOT="${ECOSET_ROOT:-/export/scratch1/home/melle/datasets/ecoset}"
CONFIG_DIR="configs/ecoset_norm_experiment"
NORMS=(batchnorm layernorm groupnorm rmsnorm derf)

# Check prerequisites
if [ ! -f "configs/ecoset_imagenet_classes.txt" ]; then
    echo "ERROR: configs/ecoset_imagenet_classes.txt not found."
    echo "Run: python scripts/compute_ecoset_overlap.py"
    exit 1
fi

if [ ! -d "$ECOSET_ROOT" ]; then
    echo "ERROR: Ecoset not found at $ECOSET_ROOT"
    echo "Run: ./scripts/download_ecoset.sh"
    exit 1
fi

N_CLASSES=$(grep -v '^#' configs/ecoset_imagenet_classes.txt | grep -c '^n' || true)
echo "============================================"
echo "Ecoset136 Norm Experiment (size-controlled)"
echo "Classes: $N_CLASSES"
echo "Max per class: 1300 (matches ImageNet)"
echo "GPUs: $NGPUS"
echo "Ecoset root: $ECOSET_ROOT"
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
        --set data.root="$ECOSET_ROOT"

    echo ""
done

echo "============================================"
echo "All ecoset136 runs complete!"
echo "============================================"
