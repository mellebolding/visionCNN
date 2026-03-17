#!/bin/bash
# =============================================================================
# Submit post-training OOD evaluation for all ecovalid norm experiment runs.
#
# Usage:
#   ./scripts/submit_snellius_ood_eval.sh
#   NORMS="batchnorm localnorm" SEEDS="42 43" ./scripts/submit_snellius_ood_eval.sh
# =============================================================================
set -eo pipefail

NORMS=(${NORMS:-batchnorm layernorm groupnorm rmsnorm derf localnorm})
SEEDS=(${SEEDS:-42 43 44 45 46})
DATASETS="imagenet_c"
DATASETS_ROOT="/projects/prjs0771/melle/datasets"
LOG_DIR="/projects/prjs0771/melle/projects/visionCNN/logs"
PARTITION="${PARTITION:-gpu_a100}"
SLURM_LOG_DIR="logs/slurm"

mkdir -p "$SLURM_LOG_DIR"

N_RUNS=0
SKIPPED=0

echo "============================================"
echo "Submitting OOD evaluation jobs"
echo "Norms: ${NORMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Datasets: ${DATASETS}"
echo "Partition: ${PARTITION}"
echo "============================================"

for norm in "${NORMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_name="resnet50_${norm}_imagenet_ecoset_ecovalid_seed${seed}"
        run_dir="${LOG_DIR}/${run_name}"
        checkpoint="${run_dir}/best.pt"
        ood_results="${run_dir}/ood_results.json"

        # Skip if training didn't complete
        if [ ! -f "${run_dir}/history.csv" ]; then
            echo "SKIP ${norm} seed=${seed}: no history.csv"
            continue
        fi
        last_epoch=$(tail -1 "${run_dir}/history.csv" | cut -d',' -f1)
        if [ "$last_epoch" -lt 25 ] 2>/dev/null; then
            echo "SKIP ${norm} seed=${seed}: only reached epoch ${last_epoch}"
            continue
        fi

        # Skip if OOD eval already done
        if [ -f "$ood_results" ]; then
            echo "SKIP ${norm} seed=${seed}: ood_results.json already exists"
            ((SKIPPED++)) || true
            continue
        fi

        if [ ! -f "$checkpoint" ]; then
            echo "WARNING: no best.pt for ${norm} seed=${seed}, using last.pt"
            checkpoint="${run_dir}/last.pt"
        fi

        job_id=$(sbatch \
            --job-name="ood_${norm}_s${seed}" \
            --partition="$PARTITION" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=4 \
            --gpus=1 \
            --mem=32G \
            --time=00:45:00 \
            --output="${SLURM_LOG_DIR}/%j_ood_${norm}_s${seed}.out" \
            --error="${SLURM_LOG_DIR}/%j_ood_${norm}_s${seed}.err" \
            --wrap="
                eval \"\$(micromamba shell hook -s bash)\" && micromamba activate visioncnn
                cd /gpfs/home1/mbolding/visionCNN
                python scripts/evaluate_ood_full.py \
                    --checkpoint ${checkpoint} \
                    --datasets ${DATASETS} \
                    --datasets_root ${DATASETS_ROOT} \
                    --output_dir ${run_dir}
            " \
            | awk '{print $NF}')

        echo "SUBMITTED ${norm} seed=${seed}: job ${job_id}"
        ((N_RUNS++)) || true
    done
done

echo ""
echo "============================================"
echo "Submitted ${N_RUNS} OOD eval jobs (${SKIPPED} skipped)."
echo "Monitor with: squeue -u \$USER"
echo "Logs: ${SLURM_LOG_DIR}/"
echo "============================================"
