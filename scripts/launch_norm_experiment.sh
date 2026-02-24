#!/bin/bash
# =============================================================================
# Launch all 18 normalization comparison experiments on Snellius.
#
# Submits 6 models x 3 normalization schemes = 18 SLURM jobs.
# Each job uses 1 A100 GPU, 18 CPUs, 120G RAM, 2 hours.
#
# Usage:
#   bash scripts/launch_norm_experiment.sh
#   bash scripts/launch_norm_experiment.sh --dry-run   # print commands without submitting
# =============================================================================

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — commands will be printed but not submitted ==="
fi

CONFIG="configs/norm_experiment.yaml"
PARTITION="gpu_a100"
TIME="02:00:00"

MODELS=("vgg_small" "vgg_medium" "resnet_small" "resnet_medium" "convnext_small" "convnext_medium")
NORMS=("batchnorm" "layernorm" "groupnorm")

echo "=============================================="
echo "Normalization Comparison Experiment"
echo "=============================================="
echo "Config: $CONFIG"
echo "Models: ${MODELS[*]}"
echo "Norms:  ${NORMS[*]}"
echo "Total jobs: $((${#MODELS[@]} * ${#NORMS[@]}))"
echo "=============================================="

JOB_COUNT=0
for MODEL in "${MODELS[@]}"; do
    for NORM in "${NORMS[@]}"; do
        EXP_NAME="${MODEL}_${NORM}"
        JOB_COUNT=$((JOB_COUNT + 1))

        CMD="sbatch \
            --partition=$PARTITION \
            --time=$TIME \
            --job-name=$EXP_NAME \
            scripts/launch_snellius.sh $CONFIG \
            --set model.name=$MODEL model.norm_layer=$NORM experiment_name=$EXP_NAME"

        if $DRY_RUN; then
            echo "[$JOB_COUNT/18] $CMD"
        else
            echo "[$JOB_COUNT/18] Submitting: $EXP_NAME"
            eval $CMD
        fi
    done
done

echo "=============================================="
echo "Submitted $JOB_COUNT jobs."
echo "=============================================="
