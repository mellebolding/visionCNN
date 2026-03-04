#!/bin/bash
# Full OOD evaluation for the ecoset136 norm experiment.
# Evaluates all 5 checkpoints, then runs cross-model error consistency.
#
# Usage:
#   ./scripts/run_full_ood_eval_ecoset.sh
#   ./scripts/run_full_ood_eval_ecoset.sh --datasets imagenet_r imagenet_a imagenet_c

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

eval "$(micromamba shell hook -s bash)"
micromamba activate visioncnn

NORMS=(batchnorm layernorm groupnorm rmsnorm derf)
LOG_DIR="logs"
DATASETS_ROOT="/export/scratch1/home/melle/datasets"
DATASETS="${@:---datasets all}"

echo "============================================"
echo "Full OOD Evaluation — Ecoset136"
echo "============================================"
echo ""

for norm in "${NORMS[@]}"; do
    run_dir="${LOG_DIR}/resnet50_${norm}_ecoset"
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

# Error consistency analysis
echo "============================================"
echo "Error Consistency Analysis"
echo "============================================"

RUN_DIRS=()
for norm in "${NORMS[@]}"; do
    pred_dir="${LOG_DIR}/resnet50_${norm}_ecoset/predictions"
    if [ -d "$pred_dir" ] && [ "$(ls -A "$pred_dir" 2>/dev/null)" ]; then
        RUN_DIRS+=("${LOG_DIR}/resnet50_${norm}_ecoset")
    fi
done

if [ ${#RUN_DIRS[@]} -ge 2 ]; then
    echo "Running error consistency on ${#RUN_DIRS[@]} models..."
    python scripts/analyze_error_consistency.py \
        --runs "${RUN_DIRS[@]}" \
        --output_dir "${LOG_DIR}/error_consistency_ecoset"
else
    echo "SKIP: Need at least 2 models with predictions, found ${#RUN_DIRS[@]}"
fi

echo ""
echo "============================================"
echo "All done!"
echo "============================================"
