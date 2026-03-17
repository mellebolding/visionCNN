#!/bin/bash
# Run ResNet50 norm experiment on ImageNet-Ecoset136 with eco-valid augmentations.
# Runs 5 seeds per norm for error bars. 6 norms including LocalNorm.
# 25 epochs per run with tiered checkpoint schedule (every 1/2/3 epochs).
# OOD eval (ImageNet-C) at every checkpoint.
#
# Usage:
#   ./scripts/run_imagenet_ecoset_ecovalid_norm_experiment.sh
#   NGPUS=1 ./scripts/run_imagenet_ecoset_ecovalid_norm_experiment.sh
#   NORMS="batchnorm localnorm" ./scripts/run_imagenet_ecoset_ecovalid_norm_experiment.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NGPUS="${NGPUS:-3}"
CONFIG_DIR="configs/imagenet_ecoset_norm_experiment_ecovalid"
NORMS=(${NORMS:-batchnorm layernorm groupnorm rmsnorm derf localnorm})
SEEDS=(${SEEDS:-42 43 44 45 46})
IMAGENET_ROOT="${IMAGENET_ROOT:-/export/scratch1/home/melle/datasets/imagenet}"

# Check class list exists
if [ ! -f "configs/ecoset_imagenet_classes.txt" ]; then
    echo "ERROR: configs/ecoset_imagenet_classes.txt not found."
    echo "Run: python scripts/compute_ecoset_overlap.py"
    exit 1
fi

N_CLASSES=$(grep -v '^#' configs/ecoset_imagenet_classes.txt | grep -c '^n' || true)
N_RUNS=$(( ${#NORMS[@]} * ${#SEEDS[@]} ))
echo "============================================"
echo "ImageNet-Ecoset136 Ecovalid Norm Experiment"
echo "Classes: $N_CLASSES"
echo "Norms: ${NORMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $N_RUNS"
echo "GPUs: $NGPUS"
echo "============================================"
echo ""

eval "$(micromamba shell hook -s bash)"
micromamba activate visioncnn

for norm in "${NORMS[@]}"; do
    config="$CONFIG_DIR/${norm}.yaml"
    if [ ! -f "$config" ]; then
        echo "WARNING: Config not found, skipping: $config"
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        run_name="resnet50_${norm}_imagenet_ecoset_ecovalid_seed${seed}"
        echo "Starting: ${norm} seed=${seed}  (${run_name})"
        echo "Config: ${config}"
        echo "--------------------------------------------"

        torchrun --nproc_per_node="$NGPUS" \
            scripts/train.py \
            --config "$config" \
            --set training.seed="$seed" \
                   experiment_name="$run_name" \
                   data.root="$IMAGENET_ROOT"

        echo ""
    done
done

echo "============================================"
echo "All ecovalid runs complete!"
echo "============================================"
