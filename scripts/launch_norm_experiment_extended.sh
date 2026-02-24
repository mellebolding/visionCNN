#!/bin/bash
# =============================================================================
# Launch extended normalization experiments on Snellius.
#
# 3 models x 4 norms x (4 batch sizes + 3 learning rates - 1 overlap) = 72 total
# Minus 9 already done (3 models x 3 norms at bs=128, lr=0.001) = 63 new jobs.
#
# Batch size sweep:  bs in {16, 32, 128, 512} at lr=0.001
# LR sweep:          lr in {0.0003, 0.001, 0.003} at bs=128
# (bs=128, lr=0.001 is the overlap point)
#
# Usage:
#   bash scripts/launch_norm_experiment_extended.sh
#   bash scripts/launch_norm_experiment_extended.sh --dry-run
#   bash scripts/launch_norm_experiment_extended.sh --include-existing  # also submit the 9 overlap runs
# =============================================================================

set -e

DRY_RUN=false
INCLUDE_EXISTING=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --include-existing) INCLUDE_EXISTING=true ;;
    esac
done

if $DRY_RUN; then
    echo "=== DRY RUN — commands will be printed but not submitted ==="
fi

CONFIG="configs/norm_experiment_extended.yaml"
PARTITION="gpu_a100"

MODELS=("vgg_medium" "resnet_medium" "convnext_medium")
NORMS=("batchnorm" "layernorm" "groupnorm" "nonorm")

echo "=============================================="
echo "Extended Normalization Experiment"
echo "=============================================="
echo "Config: $CONFIG"
echo "Models: ${MODELS[*]}"
echo "Norms:  ${NORMS[*]}"
echo "Batch size sweep: 16, 32, 128, 512 (at lr=0.001)"
echo "LR sweep: 0.0003, 0.001, 0.003 (at bs=128)"
echo "=============================================="

JOB_COUNT=0
SKIP_COUNT=0

submit_job() {
    local MODEL=$1
    local NORM=$2
    local BS=$3
    local LR=$4
    local TIME=$5

    EXP_NAME="${MODEL}_${NORM}_bs${BS}_lr${LR}"
    JOB_COUNT=$((JOB_COUNT + 1))

    CMD="sbatch \
        --partition=$PARTITION \
        --time=$TIME \
        --job-name=$EXP_NAME \
        scripts/launch_snellius.sh $CONFIG \
        --set model.name=$MODEL model.norm_layer=$NORM \
              training.batch_size=$BS training.lr=$LR \
              experiment_name=$EXP_NAME"

    if $DRY_RUN; then
        echo "[$JOB_COUNT] $CMD"
    else
        echo "[$JOB_COUNT] Submitting: $EXP_NAME (bs=$BS, lr=$LR, time=$TIME)"
        eval $CMD
        # Small delay to avoid SLURM socket timeouts
        sleep 1
    fi
}

# --- Batch size sweep (lr=0.001 fixed) ---
echo ""
echo "--- Batch size sweep (lr=0.001) ---"
for MODEL in "${MODELS[@]}"; do
    for NORM in "${NORMS[@]}"; do
        for BS in 16 32 128 512; do
            LR="0.001"

            # Skip bs=128 lr=0.001 for BN/LN/GN if they were already run
            if [[ "$BS" == "128" && "$NORM" != "nonorm" ]] && ! $INCLUDE_EXISTING; then
                SKIP_COUNT=$((SKIP_COUNT + 1))
                continue
            fi

            # Time limits: smaller batch = more iterations = more time
            case $BS in
                16)  TIME="04:00:00" ;;
                32)  TIME="03:00:00" ;;
                128) TIME="02:00:00" ;;
                512) TIME="02:00:00" ;;
            esac

            submit_job "$MODEL" "$NORM" "$BS" "$LR" "$TIME"
        done
    done
done

# --- LR sweep (bs=128 fixed) ---
echo ""
echo "--- LR sweep (bs=128) ---"
for MODEL in "${MODELS[@]}"; do
    for NORM in "${NORMS[@]}"; do
        for LR in 0.0003 0.003; do
            # lr=0.001 at bs=128 already covered in batch size sweep
            BS=128
            TIME="02:00:00"

            submit_job "$MODEL" "$NORM" "$BS" "$LR" "$TIME"
        done
    done
done

echo ""
echo "=============================================="
echo "Submitted $JOB_COUNT jobs (skipped $SKIP_COUNT existing)."
echo "=============================================="
