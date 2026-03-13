#!/bin/bash
# Run Derf VP (variance-preserving) experiments for all 3 architectures.
# Compares fixed Derf against original broken Derf and other norms.
#
# Usage:
#   ./scripts/run_derf_vp_experiment.sh

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

CONFIGS=(
    configs/imagenet_ecoset_derf_vp_experiment/resnet50.yaml
    configs/imagenet_ecoset_derf_vp_experiment/convnext_small.yaml
    configs/imagenet_ecoset_derf_vp_experiment/vit_small.yaml
)

for cfg in "${CONFIGS[@]}"; do
    echo "=============================="
    echo "Launching: $cfg"
    echo "=============================="
    ./scripts/launch_guppy.sh "$cfg"
    echo ""
done
