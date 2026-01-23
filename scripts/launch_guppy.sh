#!/bin/bash
# =============================================================================
# Local Multi-GPU Training Launcher
# =============================================================================
# Launch distributed training on a single node with multiple GPUs.
#
# Usage:
#   ./scripts/launch_local.sh configs/convnextv2_tiny.yaml
#   ./scripts/launch_local.sh configs/convnextv2_tiny.yaml --resume logs/convnextv2_tiny_cifar10/last.pt
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

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Configuration

CONFIG_FILE="${1:-configs/convnextv2_tiny.yaml}"
shift 2>/dev/null || true  # Shift to get remaining args, ignore if no more args

# Check config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Output directory for checkpoints, logs, etc.
LOG_DIR="/logs"
mkdir -p "$LOG_DIR"
LOG_DIR="${LOG_DIR:-logs}"  # Allow override via env var

# Detect number of GPUs
if [[ -z "$NGPUS" ]]; then
    NGPUS=$(nvidia-smi -L | wc -l)
fi

# Master port (pick a random one if not set, to avoid conflicts)
MASTER_PORT=${MASTER_PORT:-29500}

echo "=============================================="
echo "Multi-GPU Training Launcher"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Log directory: $LOG_DIR"
echo "GPUs: $NGPUS"
echo "Master port: $MASTER_PORT"
echo "Additional args: $@"
echo "=============================================="

# Replace <IMAGENET_ROOT> in config if present
torchrun \
    --standalone \
    --nproc_per_node=$NGPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    --config "$CONFIG_FILE" \
    --log_dir "$LOG_DIR" \
    "$@"
