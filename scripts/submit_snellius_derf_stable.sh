#!/bin/bash
# Submit Derf-stable experiment to Snellius.
# 5 seeds for the stabilized Derf config (lower LR + warmup + AGC).
#
# Usage:
#   ./scripts/submit_snellius_derf_stable.sh
#   SEEDS="42 43" ./scripts/submit_snellius_derf_stable.sh
#   PARTITION=gpu_h100 ./scripts/submit_snellius_derf_stable.sh
#
# Each job: ResNet50, 25 epochs, batch_size=128, 1× A100, ~5h each.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="configs/imagenet_ecoset_norm_experiment_ecovalid/derf_stable.yaml"
SEEDS=(${SEEDS:-42 43 44 45 46})
PARTITION="${PARTITION:-gpu_a100}"
TIME="${TIME:-06:00:00}"
NORM="derf_stable"

mkdir -p logs/slurm

echo "============================================"
echo "Submitting Derf-stable experiment to Snellius"
echo "Config: $CONFIG"
echo "Seeds: ${SEEDS[*]}"
echo "Partition: $PARTITION  Time limit: $TIME"
echo "============================================"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

for seed in "${SEEDS[@]}"; do
    run_name="resnet50_${NORM}_imagenet_ecoset_ecovalid_seed${seed}"
    log_dir="/projects/prjs0771/melle/projects/visionCNN/logs/${run_name}"

    # Skip if a completed run exists
    if [ -f "${log_dir}/history.csv" ]; then
        last_epoch=$(tail -1 "${log_dir}/history.csv" | cut -d',' -f1)
        if [ "${last_epoch}" -ge 25 ] 2>/dev/null; then
            echo "SKIP seed=${seed}: already completed epoch ${last_epoch} (${log_dir})"
            continue
        fi
    fi

    # Skip if already queued or running in SLURM
    if squeue -u "$USER" -o "%j" --noheader 2>/dev/null | grep -qx "${NORM}_s${seed}"; then
        echo "SKIP seed=${seed}: already in queue"
        continue
    fi

    # Auto-resume from last checkpoint if an incomplete run exists
    resume_args=""
    if [ -f "${log_dir}/last.pt" ]; then
        resume_args="--resume ${log_dir}/last.pt"
        echo "  (will resume from ${log_dir}/last.pt)"
    fi

    job_id=$(sbatch \
        --job-name="${NORM}_s${seed}" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --output="logs/slurm/%j_${NORM}_s${seed}.out" \
        --error="logs/slurm/%j_${NORM}_s${seed}.err" \
        scripts/launch_snellius.sh \
        "$CONFIG" \
        --set training.seed="$seed" experiment_name="$run_name" \
        $resume_args \
        | awk '{print $NF}')

    echo "Submitted seed=${seed}: job ${job_id}  (${run_name})"
done

echo ""
echo "============================================"
echo "All jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Logs: logs/slurm/"
echo "============================================"
