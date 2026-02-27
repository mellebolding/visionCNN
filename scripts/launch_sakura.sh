#!/bin/bash
# =============================================================================
# Local Training Launcher (sakura - RTX 3090)
# =============================================================================
# Launch training on the local machine.
#
# Usage:
#   ./scripts/launch_sakura.sh configs/simple_cnn.yaml
#   ./scripts/launch_sakura.sh configs/resnet18_imagenet.yaml --set training.epochs=1
#
# Environment variables:
#   NGPUS: Number of GPUs to use (default: 1)
#   MASTER_PORT: Port for distributed communication (default: 29500)
# =============================================================================

set -e

# Activate micromamba environment if not already active
if [[ "${CONDA_DEFAULT_ENV}" != "visioncnn" ]]; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate visioncnn
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
CONFIG_FILE="${1:-configs/simple_cnn.yaml}"
shift 2>/dev/null || true

# Check config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Detect number of GPUs
if [[ -z "$NGPUS" ]]; then
    NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [[ $NGPUS -eq 0 ]]; then
        echo "Error: No GPUs detected"
        exit 1
    fi
fi

# Set log directory
LOG_DIR="${LOG_DIR:-logs}"

# Master port
MASTER_PORT=${MASTER_PORT:-29500}

# =============================================================================
# Environment Variables
# =============================================================================

# Machine identifier for config resolution
export VCNN_MACHINE=sakura

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=============================================="
echo "Local Training Launcher (sakura)"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $NGPUS"
echo "Log directory: $LOG_DIR"
echo "Additional args: $@"
echo "=============================================="

torchrun \
    --nproc_per_node=$NGPUS \
    --master_addr=127.0.0.1 \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    --config "$CONFIG_FILE" \
    --log_dir "$LOG_DIR" \
    "$@"

echo "=============================================="
echo "Training complete!"
echo "=============================================="
