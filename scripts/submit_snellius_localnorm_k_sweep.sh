#!/bin/bash
# Submit LocalNorm K-sweep experiment to Snellius.
# 4 K values (K=1,4,8,16) × 5 seeds = 20 independent single-GPU jobs.
# K=2 (existing 'localnorm') is already done in the ecovalid experiment.
#
# Usage:
#   ./scripts/submit_snellius_localnorm_k_sweep.sh
#   NORMS="localnorm_k1 localnorm_k4" SEEDS="42 43" ./scripts/submit_snellius_localnorm_k_sweep.sh
#   PARTITION=gpu_h100 ./scripts/submit_snellius_localnorm_k_sweep.sh
#
# Each job: ResNet50, 25 epochs, batch_size=128, 1× A100, ~5h each.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG_DIR="configs/imagenet_ecoset_norm_experiment_ecovalid"
NORMS=(${NORMS:-localnorm_k1 localnorm_k4 localnorm_k8 localnorm_k16})
SEEDS=(${SEEDS:-42 43 44 45 46})
PARTITION="${PARTITION:-gpu_a100}"
TIME="${TIME:-06:00:00}"

mkdir -p logs/slurm

N_RUNS=$(( ${#NORMS[@]} * ${#SEEDS[@]} ))
echo "============================================"
echo "Submitting LocalNorm K-sweep to Snellius"
echo "Norms: ${NORMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total jobs: $N_RUNS"
echo "Partition: $PARTITION  Time limit: $TIME"
echo "============================================"
echo ""

for norm in "${NORMS[@]}"; do
    config="$CONFIG_DIR/${norm}.yaml"
    if [ ! -f "$config" ]; then
        echo "WARNING: Config not found, skipping: $config"
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        run_name="resnet50_${norm}_imagenet_ecoset_ecovalid_seed${seed}"
        log_dir="/projects/prjs0771/melle/projects/visionCNN/logs/${run_name}"

        # Skip if a completed run exists (history.csv with all 25 epochs present)
        if [ -f "${log_dir}/history.csv" ]; then
            last_epoch=$(tail -1 "${log_dir}/history.csv" | cut -d',' -f1)
            if [ "${last_epoch}" -ge 25 ] 2>/dev/null; then
                echo "SKIP ${norm} seed=${seed}: already completed epoch ${last_epoch} (${log_dir})"
                continue
            fi
        fi

        # Skip if already queued or running in SLURM
        if squeue -u "$USER" -o "%j" --noheader 2>/dev/null | grep -qx "${norm}_s${seed}"; then
            echo "SKIP ${norm} seed=${seed}: already in queue"
            continue
        fi

        # Auto-resume from last checkpoint if an incomplete run exists
        resume_args=""
        if [ -f "${log_dir}/last.pt" ]; then
            resume_args="--resume ${log_dir}/last.pt"
            echo "  (will resume from ${log_dir}/last.pt)"
        fi

        job_id=$(sbatch \
            --job-name="${norm}_s${seed}" \
            --partition="$PARTITION" \
            --time="$TIME" \
            --output="logs/slurm/%j_${norm}_s${seed}.out" \
            --error="logs/slurm/%j_${norm}_s${seed}.err" \
            scripts/launch_snellius.sh \
            "$config" \
            --set training.seed="$seed" experiment_name="$run_name" \
            $resume_args \
            | awk '{print $NF}')

        echo "Submitted ${norm} seed=${seed}: job ${job_id}  (${run_name})"
    done
done

echo ""
echo "============================================"
echo "All $N_RUNS jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Logs: logs/slurm/"
echo "============================================"
