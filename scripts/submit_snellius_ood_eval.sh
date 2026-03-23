#!/bin/bash
# =============================================================================
# Submit post-training OOD evaluation for all ecovalid norm experiment runs.
#
# Evaluates imagenet_r, imagenet_a, imagenet_sketch, and imagenet_c (full,
# 15 corruptions × 5 severities). For runs that already have ood_results.json
# but are missing imagenet_c keys, submits an imagenet_c-only job.
#
# Usage:
#   ./scripts/submit_snellius_ood_eval.sh
#   NORMS="batchnorm localnorm" SEEDS="42 43" ./scripts/submit_snellius_ood_eval.sh
# =============================================================================
set -eo pipefail

NORMS=(${NORMS:-batchnorm layernorm groupnorm rmsnorm derf localnorm})
SEEDS=(${SEEDS:-42 43 44 45 46})
DATASETS_STANDARD="imagenet_r imagenet_a imagenet_sketch"
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
echo "Datasets: ${DATASETS_STANDARD} imagenet_c"
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

        if [ ! -f "$checkpoint" ]; then
            echo "WARNING: no best.pt for ${norm} seed=${seed}, using last.pt"
            checkpoint="${run_dir}/last.pt"
        fi

        # Determine which datasets to run
        if [ ! -f "$ood_results" ]; then
            # Full evaluation: standard OOD + imagenet_c
            datasets="${DATASETS_STANDARD} imagenet_c"
            time_limit="02:00:00"
            job_suffix="ood_${norm}_s${seed}"
        elif ! python3 -c "import json,sys; d=json.load(open('${ood_results}')); sys.exit(0 if any('imagenet_c' in k for k in d) else 1)" 2>/dev/null; then
            # ood_results.json exists but missing imagenet_c — run C-only
            datasets="imagenet_c"
            time_limit="01:30:00"
            job_suffix="imc_${norm}_s${seed}"
            echo "  (existing ood_results.json found, appending imagenet_c)"
        else
            echo "SKIP ${norm} seed=${seed}: ood_results.json already has imagenet_c"
            ((SKIPPED++)) || true
            continue
        fi

        job_id=$(sbatch \
            --job-name="$job_suffix" \
            --partition="$PARTITION" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=4 \
            --gpus=1 \
            --mem=32G \
            --time="${time_limit}" \
            --output="${SLURM_LOG_DIR}/%j_${job_suffix}.out" \
            --error="${SLURM_LOG_DIR}/%j_${job_suffix}.err" \
            --wrap="
                MICROMAMBA=/gpfs/work1/0/prjs0771/melle/micromamba/bin/micromamba
                eval \"\$(\$MICROMAMBA shell hook -s bash)\"
                micromamba activate /gpfs/home1/mbolding/.local/share/mamba/envs/visioncnn
                cd /gpfs/home1/mbolding/visionCNN
                python scripts/evaluate_ood_full.py \
                    --checkpoint ${checkpoint} \
                    --datasets ${datasets} \
                    --datasets_root ${DATASETS_ROOT} \
                    --output_dir ${run_dir}
            " \
            | awk '{print $NF}')

        echo "SUBMITTED ${norm} seed=${seed} [${datasets}]: job ${job_id}"
        ((N_RUNS++)) || true
    done
done

echo ""
echo "============================================"
echo "Submitted ${N_RUNS} OOD eval jobs (${SKIPPED} skipped)."
echo "Monitor with: squeue -u \$USER"
echo "Logs: ${SLURM_LOG_DIR}/"
echo "============================================"
