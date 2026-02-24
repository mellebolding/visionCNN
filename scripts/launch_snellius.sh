#!/bin/bash
# =============================================================================
# Snellius SLURM Job Script for Distributed Training
# =============================================================================
# Submit distributed training jobs to Snellius (SURF) with A100 or H100 GPUs.
#
# Usage:
#   sbatch scripts/launch_snellius.sh configs/convnextv2_tiny.yaml
#   sbatch --partition=gpu_h100 scripts/launch_snellius.sh configs/convnextv2_tiny.yaml
#
# To resume training:
#   sbatch scripts/launch_snellius.sh configs/convnextv2_tiny.yaml --resume logs/convnextv2_tiny_cifar10/last.pt
#
# Available partitions:
#   gpu_a100    - A100 GPUs (4 per node)
#   gpu_h100    - H100 GPUs (4 per node)
#
# Modify SBATCH directives below to adjust resources.
# =============================================================================

#SBATCH --job-name=visioncnn
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#SBATCH --account=mcsei4132

# =============================================================================
# Environment Setup
# =============================================================================

# Exit on error
set -e

# Print job info
echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Node list: $SLURM_NODELIST"
echo "=============================================="

# Load CUDA driver userspace libraries
module purge
module load 2023
module load CUDA/12.1.1

# Activate micromamba environment (same env as guppy/sakura)
export MAMBA_ROOT_PREFIX="/projects/prjs0771/melle/micromamba"
eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s bash)"
micromamba activate visioncnn

# Machine identifier for config resolution
export VCNN_MACHINE=snellius

# Set environment variables for distributed training
# Note: MASTER_ADDR and MASTER_PORT are needed for torchrun.
# WORLD_SIZE/RANK/LOCAL_RANK are NOT set here — torchrun manages these
# for its child processes. Setting them from SLURM (ntasks=1) would
# conflict with torchrun's multi-GPU spawning.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# Randomize port per job to avoid conflicts on shared nodes
export MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 10000))}

# Performance optimizations
# OMP_NUM_THREADS controls threads PER PROCESS. With PyTorch DataLoader,
# each num_workers subprocess inherits this. Setting it too high (e.g. 16)
# with 16 workers = 256 threads on 18 CPUs = severe thrashing.
# For PyTorch backend: 1 is optimal (parallelism comes from worker count).
# For DALI backend: DALI has its own thread pool, so 1 is also fine here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# For H100 with NVLink
if [[ "$SLURM_JOB_PARTITION" == *"h100"* ]]; then
    export NCCL_P2P_LEVEL=NVL
fi

# =============================================================================
# Parse Arguments
# =============================================================================

CONFIG_FILE="${1:-configs/convnextv2_tiny.yaml}"
shift 2>/dev/null || true

# Check config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config: $CONFIG_FILE"
echo "Additional args: $@"
echo "=============================================="

# =============================================================================
# Paths
# =============================================================================

# Output directory — machine config provides the default via paths.log_dir
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

# =============================================================================
# Launch Training
# =============================================================================

# For single-node multi-GPU, use torchrun
if [[ $SLURM_NNODES -eq 1 ]]; then
    echo "Single-node training with $SLURM_GPUS_PER_NODE GPUs"
    echo "Outputs will be saved to: $LOG_DIR"
    
    # Replace <IMAGENET_ROOT> in config if present
    srun --ntasks=1 torchrun \
        --standalone \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        scripts/train.py \
        --config "$CONFIG_FILE" \
        --log_dir "$LOG_DIR" \
        "$@"
else
    # For multi-node, use srun with torchrun
    echo "Multi-node training: $SLURM_NNODES nodes, $SLURM_GPUS_PER_NODE GPUs per node"
    echo "Outputs will be saved to: $LOG_DIR"
    
    # Replace <IMAGENET_ROOT> in config if present
    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        scripts/train.py \
        --config "$CONFIG_FILE" \
        --log_dir "$LOG_DIR" \
        "$@"
fi

echo "=============================================="
echo "Training complete!"
echo "=============================================="
