#!/bin/bash
# =============================================================================
# Local Multi-GPU Training Launcher
# =============================================================================
# Launch distributed training on a single node with multiple GPUs.
#
# Usage:
#   ./scripts/launch_guppy.sh configs/convnextv2_tiny.yaml
#   ./scripts/launch_guppy.sh configs/convnextv2_tiny.yaml --resume logs/convnextv2_tiny_cifar10/last.pt
#
# Environment variables:
#   NGPUS: Number of GPUs to use (default: all available)
#   MASTER_PORT: Port for distributed communication (default: 29500)
#
# Examples:
#   # Use all available GPUs
#   ./scripts/launch_local.sh configs/convnextv2_tiny.yaml
#
#   # Use specific number of GPUs
#   NGPUS=2 ./scripts/launch_local.sh configs/convnextv2_tiny.yaml
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2 ./scripts/launch_local.sh configs/convnextv2_tiny.yaml
# =============================================================================

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
CONFIG_FILE="${1:-configs/convnextv2_tiny.yaml}"
shift 2>/dev/null || true

# Check config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Detect number of GPUs
if [[ -z "$NGPUS" ]]; then
    NGPUS=$(nvidia-smi -L | wc -l)
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
# Environment Variables to Suppress Warnings
# =============================================================================

# Suppress OMP_NUM_THREADS warning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}


# Force NCCL to use only IPv4 sockets (suppress c10d socket warnings)
export NCCL_SOCKET_FAMILY=IPv4
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1


export MASTER_ADDR=127.0.0.1

# Reduce NCCL verbosity (optional)
export NCCL_DEBUG=WARN

echo "=============================================="
echo "Multi-GPU Training Launcher"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $NGPUS"
echo "Master port: $MASTER_PORT"
echo "Log directory: $LOG_DIR"
echo "OMP threads per process: $OMP_NUM_THREADS"
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
