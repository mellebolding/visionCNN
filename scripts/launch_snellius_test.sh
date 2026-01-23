#!/bin/bash
#SBATCH --job-name=visioncnn_test
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#SBATCH --account=mcsei4132

# Quick test script on MIG partition (smaller GPU, faster queue)

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Partition: $SLURM_JOB_PARTITION"
echo "================"

# Load modules (Snellius)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate venv with additional packages
source /projects/prjs0771/melle/envs/pytorch21-cuda121/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Project paths

# Set ImageNet root based on host
PROJECT_DIR=/home/mbolding/visionCNN
LOG_DIR=/projects/prjs0771/melle/projects/visionCNN/logs

cd $PROJECT_DIR

# Quick test: single GPU run on CIFAR-10
# This will use the config defaults but only 1 GPU
python scripts/train.py \
TMP_CONFIG="/tmp/convnextv2_tiny_test.$SLURM_JOB_ID.yaml"
python scripts/train.py \
    --config "$CONFIG_FILE" \
    --log_dir $LOG_DIR
