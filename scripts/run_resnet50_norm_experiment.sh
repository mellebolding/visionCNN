#!/bin/bash
# Run all ResNet50 normalization experiments sequentially.
# Each uses a different MASTER_PORT to avoid conflicts.
#
# Usage:
#   ./scripts/run_resnet50_norm_experiment.sh
#
# To run a subset, pass norm names as arguments:
#   ./scripts/run_resnet50_norm_experiment.sh batchnorm groupnorm

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs/resnet50_norm_experiment"
LAUNCH="$SCRIPT_DIR/launch_guppy.sh"

# Default: all norms
ALL_NORMS="batchnorm layernorm groupnorm rmsnorm derf"
NORMS="${@:-$ALL_NORMS}"

BASE_PORT=${MASTER_PORT:-29500}
PORT=$BASE_PORT

for norm in $NORMS; do
    config="$CONFIG_DIR/${norm}.yaml"
    if [ ! -f "$config" ]; then
        echo "ERROR: Config not found: $config"
        continue
    fi

    echo "============================================"
    echo "Starting: ResNet50 + $norm"
    echo "Config:   $config"
    echo "Port:     $PORT"
    echo "============================================"

    MASTER_PORT=$PORT "$LAUNCH" "$config" "$@"

    PORT=$((PORT + 1))
    echo ""
    echo "Finished: ResNet50 + $norm"
    echo ""
done

echo "All experiments complete!"
