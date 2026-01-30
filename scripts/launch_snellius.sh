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
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=18
#SBATCH --mem=240G
#SBATCH --time=08:00:00
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

# Load modules - use Snellius-optimized PyTorch stack
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate venv with additional packages (timm, wandb, etc.)
source /projects/prjs0771/melle/envs/pytorch21-cuda121/bin/activate

# Ensure all distributed processes use the venv
export PATH="/projects/prjs0771/melle/envs/pytorch21-cuda121/bin:$PATH"
export PYTHONPATH="/projects/prjs0771/melle/envs/pytorch21-cuda121/lib/python3.11/site-packages:$PYTHONPATH"

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Performance optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
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

# Output directory for checkpoints, logs, etc. (project space for Snellius)
LOG_DIR="/projects/prjs0771/melle/projects/visionCNN/logs"
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
