#!/bin/bash
# =============================================================================
# Resubmit incomplete/failed experiments.
#
# - 16 timed-out jobs: resume from last.pt with extended time limits
# - 3 OOM jobs (bs=512 VGG-LN/GN, ConvNeXt-LN): resubmit with bs=256
#
# Usage:
#   bash scripts/resubmit_incomplete.sh
#   bash scripts/resubmit_incomplete.sh --dry-run
# =============================================================================

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

CONFIG="configs/norm_experiment_extended.yaml"
PARTITION="gpu_a100"
LOG_DIR="/projects/prjs0771/melle/projects/visionCNN/logs"

JOB_COUNT=0

submit() {
    local EXP_NAME=$1
    local TIME=$2
    local MODEL=$3
    local NORM=$4
    local BS=$5
    local LR=$6
    local RESUME=$7  # path to last.pt or empty

    JOB_COUNT=$((JOB_COUNT + 1))

    CMD="sbatch --partition=$PARTITION --time=$TIME --job-name=$EXP_NAME"
    CMD="$CMD scripts/launch_snellius.sh $CONFIG"
    CMD="$CMD --set model.name=$MODEL model.norm_layer=$NORM"
    CMD="$CMD training.batch_size=$BS training.lr=$LR"
    CMD="$CMD experiment_name=$EXP_NAME"
    if [[ -n "$RESUME" ]]; then
        CMD="$CMD --resume $RESUME"
    fi

    if $DRY_RUN; then
        echo "[$JOB_COUNT] $CMD"
    else
        echo "[$JOB_COUNT] Submitting: $EXP_NAME (resume=${RESUME:+yes})"
        eval $CMD
        sleep 1
    fi
}

echo ""
echo "=== Resuming 16 timed-out experiments ==="
echo ""

# VGG-16 GroupNorm (timed out)
submit vgg_medium_groupnorm_bs128_lr0.0003 03:00:00 vgg_medium groupnorm 128 0.0003 "$LOG_DIR/vgg_medium_groupnorm_bs128_lr0.0003/last.pt"
submit vgg_medium_groupnorm_bs128_lr0.003  03:00:00 vgg_medium groupnorm 128 0.003  "$LOG_DIR/vgg_medium_groupnorm_bs128_lr0.003/last.pt"
submit vgg_medium_groupnorm_bs16_lr0.001   06:00:00 vgg_medium groupnorm 16  0.001  "$LOG_DIR/vgg_medium_groupnorm_bs16_lr0.001/last.pt"
submit vgg_medium_groupnorm_bs32_lr0.001   04:00:00 vgg_medium groupnorm 32  0.001  "$LOG_DIR/vgg_medium_groupnorm_bs32_lr0.001/last.pt"

# VGG-16 LayerNorm (timed out)
submit vgg_medium_layernorm_bs128_lr0.0003 03:00:00 vgg_medium layernorm 128 0.0003 "$LOG_DIR/vgg_medium_layernorm_bs128_lr0.0003/last.pt"
submit vgg_medium_layernorm_bs128_lr0.003  03:00:00 vgg_medium layernorm 128 0.003  "$LOG_DIR/vgg_medium_layernorm_bs128_lr0.003/last.pt"
submit vgg_medium_layernorm_bs16_lr0.001   06:00:00 vgg_medium layernorm 16  0.001  "$LOG_DIR/vgg_medium_layernorm_bs16_lr0.001/last.pt"
submit vgg_medium_layernorm_bs32_lr0.001   04:00:00 vgg_medium layernorm 32  0.001  "$LOG_DIR/vgg_medium_layernorm_bs32_lr0.001/last.pt"

# ConvNeXt-M BatchNorm (timed out)
submit convnext_medium_batchnorm_bs128_lr0.0003 03:00:00 convnext_medium batchnorm 128 0.0003 "$LOG_DIR/convnext_medium_batchnorm_bs128_lr0.0003/last.pt"
submit convnext_medium_batchnorm_bs128_lr0.003  03:00:00 convnext_medium batchnorm 128 0.003  "$LOG_DIR/convnext_medium_batchnorm_bs128_lr0.003/last.pt"

# ConvNeXt-M GroupNorm (timed out)
submit convnext_medium_groupnorm_bs128_lr0.0003 03:00:00 convnext_medium groupnorm 128 0.0003 "$LOG_DIR/convnext_medium_groupnorm_bs128_lr0.0003/last.pt"
submit convnext_medium_groupnorm_bs128_lr0.003  03:00:00 convnext_medium groupnorm 128 0.003  "$LOG_DIR/convnext_medium_groupnorm_bs128_lr0.003/last.pt"
submit convnext_medium_groupnorm_bs512_lr0.001  03:00:00 convnext_medium groupnorm 512 0.001  "$LOG_DIR/convnext_medium_groupnorm_bs512_lr0.001/last.pt"

# ConvNeXt-M LayerNorm (timed out)
submit convnext_medium_layernorm_bs128_lr0.0003 03:00:00 convnext_medium layernorm 128 0.0003 "$LOG_DIR/convnext_medium_layernorm_bs128_lr0.0003/last.pt"
submit convnext_medium_layernorm_bs128_lr0.003  03:00:00 convnext_medium layernorm 128 0.003  "$LOG_DIR/convnext_medium_layernorm_bs128_lr0.003/last.pt"
# convnext_medium_layernorm_bs16_lr0.001 — completed (reached epoch 50)

echo ""
echo "=== Resubmitting 3 OOM experiments with bs=256 ==="
echo "(Original bs=512 OOM'd on A100 40GB)"
echo ""

# These OOM'd at bs=512 on A100 40GB. Resubmit with bs=256 and corrected name.
submit vgg_medium_groupnorm_bs256_lr0.001      02:00:00 vgg_medium     groupnorm 256 0.001 ""
submit vgg_medium_layernorm_bs256_lr0.001      02:00:00 vgg_medium     layernorm 256 0.001 ""
submit convnext_medium_layernorm_bs256_lr0.001 02:00:00 convnext_medium layernorm 256 0.001 ""

echo ""
echo "=============================================="
echo "Submitted $JOB_COUNT jobs total."
echo "=============================================="
