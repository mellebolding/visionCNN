#!/bin/bash
# Download and extract ecoset from the University of Osnabrück.
#
# The ZIP is password-protected. The password is published in the official
# HuggingFace loading script (kietzmannlab/ecoset/ecoset.py, base64-encoded).
# Downloading and using ecoset implies acceptance of the CC BY NC SA 2.0 license:
#   https://codeocean.com/capsule/9570390
#
# After download, the script auto-runs compute_ecoset_overlap.py to
# (re-)generate configs/ecoset_imagenet_classes.txt.
#
# Usage:
#   ./scripts/download_ecoset.sh [DATASETS_ROOT]
#
# Default: /export/scratch1/home/melle/datasets

set -eo pipefail

DATASETS_ROOT="${1:-/export/scratch1/home/melle/datasets}"
ECOSET_DIR="$DATASETS_ROOT/ecoset"
ZIP_URL="https://files.ikw.uni-osnabrueck.de/ml/ecoset/ecoset.zip"
ZIP_PATH="$DATASETS_ROOT/ecoset.zip"
# Password from kietzmannlab/ecoset/ecoset.py (base64: ZWNvc2V0X21zamtr)
ZIP_PASS="ecoset_msjkk"

echo "============================================"
echo "Ecoset Download"
echo "Target: $ECOSET_DIR"
echo "============================================"

# Check if already fully downloaded
if [ -d "$ECOSET_DIR" ]; then
    n=$(find "$ECOSET_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
    if [ "$n" -gt 1000 ]; then
        echo "Ecoset already present at $ECOSET_DIR ($n image files)"
        exit 0
    fi
fi

mkdir -p "$ECOSET_DIR"

# Download ZIP if not already present
if [ ! -f "$ZIP_PATH" ]; then
    echo ""
    echo "Downloading (~154 GB, ~30 min at 80 MB/s)..."
    echo "  $ZIP_URL"
    echo ""
    wget --show-progress -O "$ZIP_PATH" "$ZIP_URL"
    echo "Download complete."
else
    echo ""
    echo "ZIP already present at $ZIP_PATH, skipping download."
fi

# Extract
echo ""
echo "Extracting to $ECOSET_DIR ..."
# The ZIP root contains train/, val/, test/ — extract into a temp dir then move,
# because unzip can't remap the top-level folder name.
TMP_DIR="$DATASETS_ROOT/_ecoset_extract_tmp"
mkdir -p "$TMP_DIR"
unzip -P "$ZIP_PASS" -q "$ZIP_PATH" -d "$TMP_DIR"
# Move splits into ecoset/
for split in train val test; do
    if [ -d "$TMP_DIR/$split" ]; then
        mv "$TMP_DIR/$split" "$ECOSET_DIR/$split"
    fi
done
# Move any remaining files (e.g. ecoset.py)
find "$TMP_DIR" -maxdepth 1 -not -type d -exec mv {} "$ECOSET_DIR/" \;
rmdir "$TMP_DIR" 2>/dev/null || rm -rf "$TMP_DIR"
echo "Extraction complete."

# Clean up zip
rm -f "$ZIP_PATH"

# Activate env for overlap computation
eval "$(micromamba shell hook -s bash)"
micromamba activate visioncnn

echo ""
echo "Running ecoset overlap computation..."
cd "$(dirname "$0")/.."
python scripts/compute_ecoset_overlap.py \
    --ecoset_root "$ECOSET_DIR" \
    --imagenet_root "$DATASETS_ROOT/imagenet/train" \
    --output configs/ecoset_imagenet_classes.txt

echo ""
echo "============================================"
echo "Setup complete!"
echo "  Ecoset: $ECOSET_DIR"
echo "  Class list: configs/ecoset_imagenet_classes.txt"
echo "============================================"
