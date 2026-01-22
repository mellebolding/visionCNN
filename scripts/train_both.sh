#!/bin/bash
# cd /export/scratch1/home/melle/visionCNN && bash scripts/train_both.sh

# Train all models sequentially

echo "=========================================="
echo "Training SimpleCNN (100% data)..."
echo "=========================================="
python scripts/train.py --config configs/simple_cnn_strong_aug.yaml

echo "=========================================="
echo "Training SimpleCNN (50% data)..."
echo "=========================================="
python scripts/train.py --config configs/simple_cnn_strong_aug_50pct.yaml

echo "=========================================="
echo "Training SimpleCNN (10% data)..."
echo "=========================================="
python scripts/train.py --config configs/simple_cnn_strong_aug_10pct.yaml

echo "=========================================="
echo "Training ConvNeXtV2 (100% data)..."
echo "=========================================="
python scripts/train.py --config configs/convnextv2_tiny_strong_aug.yaml

echo "=========================================="
echo "Training ConvNeXtV2 (50% data)..."
echo "=========================================="
python scripts/train.py --config configs/convnextv2_tiny_strong_aug_50pct.yaml

echo "=========================================="
echo "Training ConvNeXtV2 (10% data)..."
echo "=========================================="
python scripts/train.py --config configs/convnextv2_tiny_strong_aug_10pct.yaml

echo "=========================================="
echo "All training complete!"
echo "=========================================="
echo "Run 'python scripts/analyze.py summary' to see results"
