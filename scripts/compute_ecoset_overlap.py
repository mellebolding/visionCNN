#!/usr/bin/env python3
"""Compute the ImageNet synsets that overlap with ecoset categories.

Ecoset uses PLAIN ENGLISH folder names in NNNN_label format (e.g. 0001_bee,
0045_volcano). The overlap is found by matching ecoset labels against the
primary name of each ImageNet synset (the first comma-separated alternative
in the timm/ImageNet class name list).

This yields exactly 136 synsets — confirming the commonly cited ecoset-ImageNet
overlap count.

Output format (tab-separated):
    synset_id<TAB>ecoset_label
    e.g. n02206856<TAB>bee

Usage:
    # Use pre-computed cache (if ecoset not downloaded):
    python scripts/compute_ecoset_overlap.py

    # Recompute from downloaded ecoset (also rescans imagenet):
    python scripts/compute_ecoset_overlap.py \\
        --ecoset_root /export/scratch1/home/melle/datasets/ecoset \\
        --imagenet_root /export/scratch1/home/melle/datasets/imagenet/train
"""
import argparse
import os
import sys
from pathlib import Path


# ---------- Ecoset label list ----------
# All 565 basic-level categories in ecoset (from kietzmannlab/ecoset on HuggingFace).
# Each label corresponds to one folder NNNN_label in the ecoset directory tree.
ECOSET_LABELS = [
    'cymbals', 'bison', 'lemonade', 'crib', 'chestnut', 'mosquito', 'aloe',
    'extinguisher', 'onion', 'starfish', 'basket', 'jar', 'snail', 'mushroom',
    'coffin', 'joystick', 'raspberry', 'gearshift', 'tyrannosaurus', 'stadium',
    'telescope', 'blueberry', 'hippo', 'cannabis', 'hairbrush', 'river',
    'artichoke', 'wallet', 'city', 'bee', 'rifle', 'boar', 'bib', 'envelope',
    'silverfish', 'shower', 'curtain', 'pinwheel', 'guillotine', 'snowplow',
    'hut', 'jukebox', 'gecko', 'marshmallow', 'lobster', 'flashlight',
    'breadfruit', 'cow', 'spoon', 'blender', 'croissant', 'greenhouse',
    'church', 'antenna', 'monkey', 'zucchini', 'snake', 'manatee', 'child',
    'table', 'winterberry', 'sloth', 'cannon', 'baguette', 'persimmon',
    'candelabra', 'necklace', 'flag', 'geyser', 'thermos', 'tweezers',
    'chandelier', 'kebab', 'mailbox', 'steamroller', 'crayon', 'lawnmower',
    'pomegranate', 'fire', 'violin', 'matchstick', 'train', 'hamster',
    'bobsleigh', 'boat', 'bullet', 'forklift', 'clock', 'saltshaker',
    'anteater', 'crowbar', 'lightbulb', 'pier', 'muffin', 'paintbrush',
    'crawfish', 'bench', 'nectarine', 'eyedropper', 'backpack', 'goat',
    'hotplate', 'fishnet', 'robot', 'rice', 'shovel', 'candle', 'blimp',
    'bridge', 'mountain', 'coleslaw', 'stagecoach', 'waterfall', 'ladle',
    'radiator', 'drain', 'tray', 'house', 'key', 'skunk', 'lake', 'earpiece',
    'gazebo', 'blackberry', 'groundhog', 'paperclip', 'cookie', 'milk', 'rug',
    'thermostat', 'milkshake', 'scoreboard', 'bean', 'giraffe', 'antelope',
    'newsstand', 'camcorder', 'sawmill', 'balloon', 'ladder', 'videotape',
    'microphone', 'coin', 'hay', 'moth', 'octopus', 'honeycomb', 'wrench',
    'cane', 'bobcat', 'banner', 'newspaper', 'reef', 'worm', 'cucumber',
    'beach', 'couch', 'streetlamp', 'rhino', 'ceiling', 'cupcake', 'hourglass',
    'caterpillar', 'tamale', 'asparagus', 'flower', 'frog', 'dog', 'knife',
    'lamp', 'walnut', 'grape', 'scone', 'peanut', 'ferret', 'kettle',
    'elephant', 'oscilloscope', 'weasel', 'guava', 'gramophone', 'stove',
    'bamboo', 'chicken', 'guacamole', 'toolbox', 'tractor', 'tiger',
    'butterfly', 'coffeepot', 'bus', 'meteorite', 'fish', 'graveyard',
    'blowtorch', 'grapefruit', 'cat', 'jellyfish', 'carousel', 'wheat',
    'tadpole', 'kazoo', 'raccoon', 'typewriter', 'scissors', 'pothole',
    'earring', 'drawers', 'cup', 'warthog', 'wall', 'lighthouse', 'burrito',
    'cassette', 'nacho', 'sink', 'seashell', 'bed', 'noodles', 'woman',
    'rabbit', 'fence', 'pistachio', 'pencil', 'hotdog', 'ball', 'ship',
    'strawberry', 'pan', 'custard', 'dolphin', 'tent', 'bun', 'tortilla',
    'tumbleweed', 'playground', 'scallion', 'anchor', 'hare', 'waterspout',
    'dough', 'burner', 'kale', 'razor', 'chocolate', 'doughnut', 'squeegee',
    'bandage', 'beaver', 'refrigerator', 'cork', 'anvil', 'microchip',
    'banana', 'thumbtack', 'chair', 'sharpener', 'bird', 'castle', 'wand',
    'doormat', 'celery', 'steak', 'ant', 'apple', 'cave', 'scaffolding',
    'bell', 'towel', 'mantis', 'thimble', 'bowl', 'chess', 'pickle',
    'lollypop', 'leek', 'barrel', 'dollhouse', 'tapioca', 'spareribs', 'fig',
    'apricot', 'strongbox', 'brownie', 'beaker', 'manhole', 'piano', 'whale',
    'hammer', 'dishrag', 'pecan', 'highlighter', 'pretzel', 'earwig',
    'cogwheel', 'trashcan', 'syringe', 'turnip', 'pear', 'lettuce', 'hedgehog',
    'guardrail', 'bubble', 'pineapple', 'burlap', 'moon', 'spider', 'fern',
    'binoculars', 'gravel', 'plum', 'scorpion', 'cube', 'squirrel', 'book',
    'crouton', 'bag', 'lantern', 'parsley', 'jaguar', 'thyme', 'oyster',
    'kumquat', 'chinchilla', 'cherry', 'umbrella', 'bicycle', 'eggbeater',
    'pig', 'kitchen', 'fondue', 'treadmill', 'casket', 'papaya', 'beetle',
    'shredder', 'grasshopper', 'anthill', 'chili', 'bottle', 'calculator',
    'gondola', 'pizza', 'compass', 'mop', 'hamburger', 'chipmunk', 'bagel',
    'outhouse', 'pliers', 'wolf', 'matchbook', 'corn', 'salamander', 'lasagna',
    'stethoscope', 'eggroll', 'avocado', 'eggplant', 'mouse', 'walrus',
    'sprinkler', 'glass', 'cauldron', 'parsnip', 'canoe', 'pancake', 'koala',
    'deer', 'chalk', 'urinal', 'toilet', 'cabbage', 'platypus', 'lizard',
    'leopard', 'cake', 'hammock', 'defibrillator', 'sundial', 'beet',
    'popcorn', 'spinach', 'cauliflower', 'canyon', 'spacecraft', 'teapot',
    'tunnel', 'porcupine', 'jail', 'spearmint', 'dustpan', 'calipers',
    'toast', 'drum', 'phone', 'wire', 'alligator', 'vase', 'motorcycle',
    'toothpick', 'coconut', 'lion', 'turtle', 'cheetah', 'bugle', 'casino',
    'fountain', 'pie', 'bread', 'meatball', 'windmill', 'gun', 'projector',
    'chameleon', 'tomato', 'nutmeg', 'plate', 'bulldozer', 'camel', 'sphinx',
    'mall', 'hanger', 'ukulele', 'wheelbarrow', 'ring', 'dildo', 'loudspeaker',
    'odometer', 'ruler', 'mousetrap', 'breadbox', 'parachute', 'bolt',
    'bracelet', 'library', 'otter', 'airplane', 'pea', 'tongs', 'cactus',
    'knot', 'shrimp', 'computer', 'sheep', 'television', 'melon', 'kangaroo',
    'helicopter', 'birdcage', 'pumpkin', 'dishwasher', 'crocodile', 'stairs',
    'garlic', 'barnacle', 'crate', 'lime', 'axe', 'hairpin', 'egg', 'emerald',
    'candy', 'stegosaurus', 'broom', 'mistletoe', 'submarine', 'fireworks',
    'peach', 'ape', 'chalkboard', 'bumblebee', 'potato', 'battery', 'guitar',
    'opossum', 'volcano', 'llama', 'ashtray', 'sieve', 'coliseum', 'cinnamon',
    'moose', 'tree', 'donkey', 'wasp', 'corkscrew', 'gargoyle', 'taco',
    'macadamia', 'camera', 'mandolin', 'kite', 'cranberry', 'thermometer',
    'tofu', 'closet', 'hovercraft', 'escalator', 'horseshoe', 'wristwatch',
    'lemon', 'sushi', 'rat', 'rainbow', 'pillow', 'radish', 'granola', 'okra',
    'pastry', 'mango', 'dragonfly', 'flashbulb', 'chalice', 'acorn',
    'birdhouse', 'gooseberry', 'locker', 'padlock', 'missile', 'clarinet',
    'panda', 'iceberg', 'road', 'flea', 'hazelnut', 'cockroach', 'needle',
    'omelet', 'desert', 'condom', 'graffiti', 'iguana', 'bucket', 'photocopier',
    'blanket', 'microscope', 'horse', 'nest', 'screwdriver', 'toaster', 'car',
    'doll', 'salsa', 'man', 'zebra', 'stapler', 'grate', 'truck', 'bear',
    'carrot', 'auditorium', 'cashew', 'shield', 'crown', 'altar', 'pudding',
    'cheese', 'rhubarb', 'broccoli', 'tower', 'cumin', 'elevator', 'wheelchair',
    'flyswatter',
]


def get_ecoset_labels_from_dir(ecoset_root: str) -> list[str]:
    """Extract ecoset category labels from the downloaded directory.

    Ecoset folders use NNNN_label format, e.g.:
        0001_bee/
        0002_volcano/
        0045_tree_frog/   (multi-word: underscore → space)
    """
    labels = []
    # Ecoset may have train/ subdir or be flat
    search_root = ecoset_root
    for candidate_root in [ecoset_root, os.path.join(ecoset_root, "train")]:
        if os.path.isdir(candidate_root):
            for name in os.listdir(candidate_root):
                if not os.path.isdir(os.path.join(candidate_root, name)):
                    continue
                parts = name.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    label = parts[1].replace("_", " ")
                    labels.append(label)
            if labels:
                search_root = candidate_root
                break
    return sorted(set(labels))


def get_imagenet_synsets(imagenet_root: str) -> set[str]:
    """Get all synset folder names from ImageNet train directory."""
    if not os.path.isdir(imagenet_root):
        return set()
    return {
        name for name in os.listdir(imagenet_root)
        if os.path.isdir(os.path.join(imagenet_root, name))
        and name.startswith("n") and len(name) == 9 and name[1:].isdigit()
    }


def compute_overlap(
    ecoset_labels: list[str],
    imagenet_root: str,
) -> list[tuple[str, str]]:
    """Match ecoset labels to ImageNet synsets via primary class name.

    For each ecoset label, looks for an ImageNet synset whose *primary name*
    (the part before the first comma in the timm description) exactly matches
    the label (case-insensitive).

    Returns:
        Sorted list of (synset_id, ecoset_label) pairs.
    """
    try:
        from timm.data import ImageNetInfo
    except ImportError:
        print("ERROR: timm not installed. Run: pip install timm")
        sys.exit(1)

    info = ImageNetInfo()
    train_synsets = sorted(get_imagenet_synsets(imagenet_root))

    if not train_synsets:
        print(f"ERROR: No ImageNet synsets found at {imagenet_root}")
        sys.exit(1)

    # Build: primary_name (lowercase) -> synset_id
    primary_to_synset: dict[str, str] = {}
    for idx, synset in enumerate(train_synsets):
        primary = info.index_to_description(idx).split(",")[0].lower().strip()
        primary_to_synset[primary] = synset

    ecoset_set = {lbl.lower().strip() for lbl in ecoset_labels}

    matches: list[tuple[str, str]] = []
    for label in ecoset_labels:
        label_lower = label.lower().strip()
        if label_lower in primary_to_synset:
            matches.append((primary_to_synset[label_lower], label_lower))

    return sorted(matches, key=lambda x: x[0])  # sort by synset ID


def save_overlap(matches: list[tuple[str, str]], output_path: str) -> None:
    """Write synset-label pairs to a tab-separated file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# ImageNet synsets that overlap with ecoset categories\n")
        f.write("# Generated by scripts/compute_ecoset_overlap.py\n")
        f.write(f"# {len(matches)} synsets\n")
        f.write("# Format: synset_id<TAB>ecoset_label\n")
        for synset, label in matches:
            f.write(f"{synset}\t{label}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute ImageNet-Ecoset synset overlap")
    parser.add_argument("--ecoset_root", type=str,
                        default="/export/scratch1/home/melle/datasets/ecoset",
                        help="Path to ecoset dataset (optional — uses built-in label list if absent)")
    parser.add_argument("--imagenet_root", type=str,
                        default="/export/scratch1/home/melle/datasets/imagenet/train",
                        help="Path to ImageNet train directory")
    parser.add_argument("--output", type=str,
                        default="configs/ecoset_imagenet_classes.txt",
                        help="Output file for the sorted synset-label list")
    args = parser.parse_args()

    # Get ecoset labels: from directory if available, else from built-in list
    if os.path.isdir(args.ecoset_root):
        print(f"Scanning ecoset directory: {args.ecoset_root}")
        ecoset_labels = get_ecoset_labels_from_dir(args.ecoset_root)
        print(f"Found {len(ecoset_labels)} unique labels in ecoset directory")
    else:
        print(f"Ecoset directory not found at {args.ecoset_root}")
        print("Using built-in ecoset label list (565 labels)")
        ecoset_labels = ECOSET_LABELS

    print(f"Scanning ImageNet: {args.imagenet_root}")
    print("Matching ecoset labels to ImageNet primary class names...")

    matches = compute_overlap(ecoset_labels, args.imagenet_root)
    print(f"\nOverlap: {len(matches)} synsets (ecoset labels that match an ImageNet primary class name)")

    # Show results
    print("\nAll matches:")
    for synset, label in matches:
        print(f"  {synset}  {label}")

    save_overlap(matches, args.output)
    print(f"\nSaved {len(matches)} synsets to {args.output}")
    print("You can now use dataset: imagenet_ecoset136 in training configs.")


if __name__ == "__main__":
    main()
