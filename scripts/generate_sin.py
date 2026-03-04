#!/usr/bin/env python3
"""Generate Stylized ImageNet (SIN) for a given class subset.

Uses AdaIN style transfer (Huang & Belongie, 2017) to create texture-shifted
versions of ImageNet images. Each image gets a random artistic style applied,
removing texture cues while preserving shape.

Usage:
    # Generate SIN for ImageNet-100 val set
    python scripts/generate_sin.py \
        --source_root /export/scratch1/home/melle/datasets/imagenet/val \
        --source_dataset imagenet100 \
        --output_root /export/scratch1/home/melle/datasets/sin-imagenet100 \
        --style_dir /export/scratch1/home/melle/datasets/styles

    # Generate for train set too (optional, much larger)
    python scripts/generate_sin.py \
        --source_root /export/scratch1/home/melle/datasets/imagenet/train \
        --source_dataset imagenet100 \
        --output_root /export/scratch1/home/melle/datasets/sin-imagenet100-train \
        --style_dir /export/scratch1/home/melle/datasets/styles

Dependencies:
    pip install Pillow torchvision
    Download style images (e.g., from Kaggle "painter by numbers" or WikiArt)
"""
import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------- AdaIN Network Components ----------

class VGGEncoder(nn.Module):
    """VGG19 encoder up to relu4_1 for AdaIN style transfer."""

    def __init__(self):
        super().__init__()
        # VGG19 layers up to relu4_1
        self.layers = nn.Sequential(
            nn.Conv2d(3, 3, 1),        # 0
            nn.ReflectionPad2d(1),      # 1
            nn.Conv2d(3, 64, 3),        # 2
            nn.ReLU(inplace=True),      # 3  relu1_1
            nn.ReflectionPad2d(1),      # 4
            nn.Conv2d(64, 64, 3),       # 5
            nn.ReLU(inplace=True),      # 6  relu1_2
            nn.MaxPool2d(2, 2),         # 7
            nn.ReflectionPad2d(1),      # 8
            nn.Conv2d(64, 128, 3),      # 9
            nn.ReLU(inplace=True),      # 10 relu2_1
            nn.ReflectionPad2d(1),      # 11
            nn.Conv2d(128, 128, 3),     # 12
            nn.ReLU(inplace=True),      # 13 relu2_2
            nn.MaxPool2d(2, 2),         # 14
            nn.ReflectionPad2d(1),      # 15
            nn.Conv2d(128, 256, 3),     # 16
            nn.ReLU(inplace=True),      # 17 relu3_1
            nn.ReflectionPad2d(1),      # 18
            nn.Conv2d(256, 256, 3),     # 19
            nn.ReLU(inplace=True),      # 20 relu3_2
            nn.ReflectionPad2d(1),      # 21
            nn.Conv2d(256, 256, 3),     # 22
            nn.ReLU(inplace=True),      # 23 relu3_3
            nn.ReflectionPad2d(1),      # 24
            nn.Conv2d(256, 256, 3),     # 25
            nn.ReLU(inplace=True),      # 26 relu3_4
            nn.MaxPool2d(2, 2),         # 27
            nn.ReflectionPad2d(1),      # 28
            nn.Conv2d(256, 512, 3),     # 29
            nn.ReLU(inplace=True),      # 30 relu4_1
        )

    def forward(self, x):
        return self.layers(x)


class AdaINDecoder(nn.Module):
    """Decoder that mirrors VGG encoder (relu4_1 back to image)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.layers(x)


def adaptive_instance_normalization(content_feat, style_feat):
    """AdaIN: normalize content features to have style's mean and std."""
    size = content_feat.size()
    style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    style_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-6
    content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    content_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-6

    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


def style_transfer(encoder, decoder, content, style, alpha=1.0):
    """Perform AdaIN style transfer."""
    content_feat = encoder(content)
    style_feat = encoder(style)
    t = adaptive_instance_normalization(content_feat, style_feat)
    t = alpha * t + (1 - alpha) * content_feat
    return decoder(t)


def download_pretrained_weights(cache_dir: str):
    """Download pretrained AdaIN encoder/decoder weights."""
    os.makedirs(cache_dir, exist_ok=True)
    encoder_path = os.path.join(cache_dir, "vgg_normalised.pth")
    decoder_path = os.path.join(cache_dir, "decoder.pth")

    base_url = "https://huggingface.co/fx-long/AdaIN-style-transfer/resolve/main"

    if not os.path.exists(encoder_path):
        print(f"Downloading VGG encoder weights...")
        torch.hub.download_url_to_file(f"{base_url}/vgg_normalised.pth", encoder_path)

    if not os.path.exists(decoder_path):
        print(f"Downloading AdaIN decoder weights...")
        torch.hub.download_url_to_file(f"{base_url}/decoder.pth", decoder_path)

    return encoder_path, decoder_path


def get_class_dirs(source_root: str, source_dataset: str) -> list[str]:
    """Get class directories to process based on source dataset."""
    if source_dataset == "imagenet100":
        from src.datasets.imagenet100 import get_imagenet100_classes
        # get_imagenet100_classes needs the train root to determine the 100 classes
        # But we can also just check which synsets exist in source_root
        all_dirs = sorted(os.listdir(source_root))
        imagenet_train = source_root.replace("/val", "/train")
        if os.path.isdir(imagenet_train):
            classes = get_imagenet100_classes(imagenet_train)
        else:
            # Fallback: assume source already filtered
            classes = all_dirs
        return [d for d in all_dirs if d in set(classes)]
    else:
        # Full dataset — all directories
        return sorted([d for d in os.listdir(source_root)
                       if os.path.isdir(os.path.join(source_root, d))])


def load_style_images(style_dir: str, max_styles: int = 100) -> list[str]:
    """Load paths to style images."""
    if not os.path.isdir(style_dir):
        print(f"Warning: Style directory {style_dir} not found")
        print("Please download style images (e.g., from WikiArt or Kaggle 'painter by numbers')")
        return []

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for f in os.listdir(style_dir):
        if os.path.splitext(f)[1].lower() in extensions:
            paths.append(os.path.join(style_dir, f))
    random.shuffle(paths)
    return paths[:max_styles]


def main():
    parser = argparse.ArgumentParser(description="Generate Stylized ImageNet")
    parser.add_argument("--source_root", type=str, required=True,
                        help="Source ImageNet directory (e.g., datasets/imagenet/val)")
    parser.add_argument("--source_dataset", type=str, default="imagenet100",
                        choices=["imagenet100", "imagenet", "ecoset"],
                        help="Which class subset to use")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory for stylized images")
    parser.add_argument("--style_dir", type=str, required=True,
                        help="Directory containing style images")
    parser.add_argument("--weights_cache", type=str,
                        default=os.path.expanduser("~/.cache/adain"),
                        help="Cache dir for pretrained weights")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Style strength (0=content, 1=full style)")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Processing resolution (default 512)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load style images
    style_paths = load_style_images(args.style_dir)
    if not style_paths:
        print("No style images found. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(style_paths)} style images")

    # Download and load pretrained weights
    encoder_path, decoder_path = download_pretrained_weights(args.weights_cache)

    encoder = VGGEncoder()
    encoder.layers.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
    encoder = encoder.to(device).eval()

    decoder = AdaINDecoder()
    decoder.layers.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
    decoder = decoder.to(device).eval()

    # Get class directories
    class_dirs = get_class_dirs(args.source_root, args.source_dataset)
    print(f"Processing {len(class_dirs)} classes from {args.source_dataset}")

    # Image transforms
    content_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])
    style_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    total_images = 0
    for cls_dir in tqdm(class_dirs, desc="Classes"):
        src_dir = os.path.join(args.source_root, cls_dir)
        out_dir = os.path.join(args.output_root, cls_dir)
        os.makedirs(out_dir, exist_ok=True)

        images = sorted([f for f in os.listdir(src_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPEG"))])

        for img_name in images:
            out_path = os.path.join(out_dir, img_name)
            if os.path.exists(out_path):
                continue

            try:
                # Load content image
                content_img = Image.open(os.path.join(src_dir, img_name)).convert("RGB")
                content_tensor = content_transform(content_img).unsqueeze(0).to(device)

                # Random style
                style_path = random.choice(style_paths)
                style_img = Image.open(style_path).convert("RGB")
                style_tensor = style_transform(style_img).unsqueeze(0).to(device)

                # Style transfer
                with torch.no_grad():
                    output = style_transfer(encoder, decoder, content_tensor, style_tensor,
                                            alpha=args.alpha)

                # Save
                output = output.clamp(0, 1).squeeze(0).cpu()
                output_img = transforms.ToPILImage()(output)
                output_img.save(out_path, quality=95)
                total_images += 1

            except Exception as e:
                print(f"Warning: Failed to process {img_name}: {e}")
                continue

    print(f"\nDone! Generated {total_images} stylized images in {args.output_root}")


if __name__ == "__main__":
    main()
