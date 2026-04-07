#!/bin/bash
# =============================================================================
# Submit dynamic-eval OOD evaluation for localnorm and batchnorm checkpoints.
#
# Loads existing trained checkpoints and re-evaluates them with dynamic
# test-time batch statistics (--dynamic_eval flag), saving results to a
# separate _dyn subdirectory so original results are not overwritten.
#
# Usage:
#   ./scripts/submit_snellius_ood_eval_dyn.sh
#   NORMS="batchnorm" SEEDS="42 43" ./scripts/submit_snellius_ood_eval_dyn.sh
# =============================================================================
set -eo pipefail

NORMS=(${NORMS:-batchnorm localnorm})
SEEDS=(${SEEDS:-42 43 44 45 46})
DATASETS_ROOT="/projects/prjs0771/melle/datasets"
LOG_DIR="/projects/prjs0771/melle/projects/visionCNN/logs"
PARTITION="${PARTITION:-gpu_a100}"
SLURM_LOG_DIR="logs/slurm"

mkdir -p "$SLURM_LOG_DIR"

N_RUNS=0
SKIPPED=0

echo "============================================"
echo "Submitting dynamic-eval OOD evaluation jobs"
echo "Norms: ${NORMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Partition: ${PARTITION}"
echo "============================================"

for norm in "${NORMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_name="resnet50_${norm}_imagenet_ecoset_ecovalid_seed${seed}"
        run_dir="${LOG_DIR}/${run_name}"
        checkpoint="${run_dir}/best.pt"
        output_dir="${run_dir}_dyn"

        # Skip if training didn't complete
        if [ ! -f "${run_dir}/history.csv" ]; then
            echo "SKIP ${norm} seed=${seed}: no history.csv at ${run_dir}"
            continue
        fi
        last_epoch=$(tail -1 "${run_dir}/history.csv" | cut -d',' -f1)
        if [ "$last_epoch" -lt 25 ] 2>/dev/null; then
            echo "SKIP ${norm} seed=${seed}: only reached epoch ${last_epoch}"
            continue
        fi

        if [ ! -f "$checkpoint" ]; then
            echo "WARNING: no best.pt for ${norm} seed=${seed}, trying last.pt"
            checkpoint="${run_dir}/last.pt"
            if [ ! -f "$checkpoint" ]; then
                echo "SKIP ${norm} seed=${seed}: no checkpoint found"
                ((SKIPPED++)) || true
                continue
            fi
        fi

        # Skip if already done
        if [ -f "${output_dir}/ood_results.json" ]; then
            echo "SKIP ${norm} seed=${seed}: ${output_dir}/ood_results.json already exists"
            ((SKIPPED++)) || true
            continue
        fi

        job_id=$(sbatch \
            --job-name="dyn_${norm}_s${seed}" \
            --partition="$PARTITION" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=4 \
            --gpus=1 \
            --mem=32G \
            --time="02:00:00" \
            --output="${SLURM_LOG_DIR}/%j_dyn_${norm}_s${seed}.out" \
            --error="${SLURM_LOG_DIR}/%j_dyn_${norm}_s${seed}.err" \
            --wrap="
                MICROMAMBA=/gpfs/work1/0/prjs0771/melle/micromamba/bin/micromamba
                eval \"\$(\$MICROMAMBA shell hook -s bash)\"
                micromamba activate /gpfs/home1/mbolding/.local/share/mamba/envs/visioncnn
                cd /gpfs/home1/mbolding/visionCNN
                python scripts/evaluate_ood_full.py \
                    --checkpoint ${checkpoint} \
                    --datasets imagenet_r imagenet_sketch imagenet_a imagenet_c \
                    --datasets_root ${DATASETS_ROOT} \
                    --output_dir ${output_dir} \
                    --dynamic_eval
            " \
            | awk '{print $NF}')

        echo "SUBMITTED ${norm} seed=${seed}: job ${job_id}  →  ${output_dir}"
        ((N_RUNS++)) || true
    done
done

echo ""
echo "============================================"
echo "Submitted ${N_RUNS} dynamic-eval jobs (${SKIPPED} skipped)."
echo "Results will be saved to: \${run_dir}_dyn/ood_results.json"
echo "Monitor with: squeue -u \$USER"
echo "Logs: ${SLURM_LOG_DIR}/"
echo "============================================"
