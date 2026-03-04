#!/bin/bash
# Download OOD evaluation datasets for the normalization experiment.
#
# Usage:
#   ./scripts/setup_ood_datasets.sh [DATASETS_ROOT]
#
# Default: /export/scratch1/home/melle/datasets

set -e

DATASETS_ROOT="${1:-/export/scratch1/home/melle/datasets}"
echo "Datasets root: $DATASETS_ROOT"
mkdir -p "$DATASETS_ROOT"

# --- ImageNet-R (Renditions) ---
if [ ! -d "$DATASETS_ROOT/imagenet-r" ]; then
    echo "Downloading ImageNet-R..."
    wget -q --show-progress -P "$DATASETS_ROOT" \
        https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
    echo "Extracting ImageNet-R..."
    tar -xf "$DATASETS_ROOT/imagenet-r.tar" -C "$DATASETS_ROOT"
    rm "$DATASETS_ROOT/imagenet-r.tar"
    echo "ImageNet-R ready at $DATASETS_ROOT/imagenet-r/"
else
    echo "ImageNet-R already exists at $DATASETS_ROOT/imagenet-r/"
fi

# --- ImageNet-A (Adversarial) ---
if [ ! -d "$DATASETS_ROOT/imagenet-a" ]; then
    echo "Downloading ImageNet-A..."
    wget -q --show-progress -P "$DATASETS_ROOT" \
        https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
    echo "Extracting ImageNet-A..."
    tar -xf "$DATASETS_ROOT/imagenet-a.tar" -C "$DATASETS_ROOT"
    rm "$DATASETS_ROOT/imagenet-a.tar"
    echo "ImageNet-A ready at $DATASETS_ROOT/imagenet-a/"
else
    echo "ImageNet-A already exists at $DATASETS_ROOT/imagenet-a/"
fi

# --- ImageNet-Sketch ---
if [ ! -d "$DATASETS_ROOT/imagenet-sketch" ]; then
    echo "Downloading ImageNet-Sketch..."
    # ImageNet-Sketch is hosted on Google Drive / HuggingFace
    # Using HuggingFace download via Python
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='songweig/imagenet_sketch',
    repo_type='dataset',
    local_dir='$DATASETS_ROOT/imagenet-sketch',
    ignore_patterns=['*.md', '*.txt', '.gitattributes'],
)
print('ImageNet-Sketch downloaded successfully')
" 2>/dev/null || {
    echo "HuggingFace download failed. Trying kaggle..."
    echo "Please download manually from: https://www.kaggle.com/datasets/wanghaohan/imagenetsketch"
    echo "Extract to: $DATASETS_ROOT/imagenet-sketch/"
}
else
    echo "ImageNet-Sketch already exists at $DATASETS_ROOT/imagenet-sketch/"
fi

# --- Python dependencies ---
echo ""
echo "Installing Python dependencies..."
pip install -q imagecorruptions 2>/dev/null || echo "Warning: imagecorruptions install failed"

echo ""
echo "=== Setup complete ==="
echo "Datasets:"
for d in imagenet-r imagenet-a imagenet-sketch; do
    if [ -d "$DATASETS_ROOT/$d" ]; then
        n=$(find "$DATASETS_ROOT/$d" -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
        echo "  $d: $n images"
    else
        echo "  $d: NOT FOUND"
    fi
done
