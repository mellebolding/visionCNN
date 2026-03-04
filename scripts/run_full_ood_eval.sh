#!/bin/bash
# Run full OOD evaluation on all 5 norm experiment checkpoints,
# then compute cross-model error consistency.
#
# Usage:
#   ./scripts/run_full_ood_eval.sh
#   ./scripts/run_full_ood_eval.sh --datasets imagenet_r imagenet_a imagenet_c
#
# Requires: OOD datasets downloaded (see scripts/setup_ood_datasets.sh)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate environment
eval "$(micromamba shell hook -s bash)"
micromamba activate visioncnn

# Norm variants
NORMS=(batchnorm layernorm groupnorm rmsnorm derf)
LOG_DIR="logs"
DATASETS_ROOT="/export/scratch1/home/melle/datasets"

# Default datasets (override with CLI args)
DATASETS="${@:---datasets all}"

echo "============================================"
echo "Full OOD Evaluation — Norm Experiment"
echo "============================================"
echo ""

# Phase 1: Run full OOD eval on each checkpoint
for norm in "${NORMS[@]}"; do
    run_dir="${LOG_DIR}/resnet50_${norm}_noaug"
    checkpoint="${run_dir}/best.pt"

    if [ ! -f "$checkpoint" ]; then
        echo "SKIP: ${norm} — no checkpoint at ${checkpoint}"
        continue
    fi

    echo "--------------------------------------------"
    echo "Evaluating: ${norm}"
    echo "Checkpoint: ${checkpoint}"
    echo "--------------------------------------------"

    python scripts/evaluate_ood_full.py \
        --checkpoint "$checkpoint" \
        $DATASETS \
        --datasets_root "$DATASETS_ROOT" \
        --save_predictions \
        --batch_size 64 \
        --num_workers 8

    echo ""
done

# Phase 2: Error consistency analysis (needs predictions from at least 2 runs)
echo "============================================"
echo "Error Consistency Analysis"
echo "============================================"

RUN_DIRS=()
for norm in "${NORMS[@]}"; do
    run_dir="${LOG_DIR}/resnet50_${norm}_noaug"
    pred_dir="${run_dir}/predictions"
    if [ -d "$pred_dir" ] && [ "$(ls -A "$pred_dir" 2>/dev/null)" ]; then
        RUN_DIRS+=("$run_dir")
    fi
done

if [ ${#RUN_DIRS[@]} -ge 2 ]; then
    echo "Running error consistency on ${#RUN_DIRS[@]} models..."
    python scripts/analyze_error_consistency.py \
        --runs "${RUN_DIRS[@]}" \
        --output_dir "${LOG_DIR}/error_consistency"
else
    echo "SKIP: Need at least 2 models with predictions, found ${#RUN_DIRS[@]}"
fi

echo ""
echo "============================================"
echo "All done!"
echo "============================================"
