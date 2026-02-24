"""Model factory for building models from config."""
import torchvision.models as tv_models

from .convnextv2 import (
    convnextv2_tiny, convnextv2_base, convnextv2_small,
    convnext_small, convnext_medium,
)
from .simple_cnn import simple_cnn
from .vgg import vgg_small, vgg_medium
from .resnet import resnet_small, resnet_medium


def resnet18(num_classes: int = 1000, pretrained: bool = False, **kwargs):
    """Create ResNet18 model (torchvision)."""
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    if num_classes != 1000:
        model.fc = __import__('torch').nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet34(num_classes: int = 1000, pretrained: bool = False, **kwargs):
    """Create ResNet34 model (torchvision)."""
    weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet34(weights=weights)
    if num_classes != 1000:
        model.fc = __import__('torch').nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet50(num_classes: int = 1000, pretrained: bool = False, **kwargs):
    """Create ResNet50 model (torchvision)."""
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.resnet50(weights=weights)
    if num_classes != 1000:
        model.fc = __import__('torch').nn.Linear(model.fc.in_features, num_classes)
    return model


# Models that accept norm_layer (string) as a kwarg
_NORM_AWARE_MODELS = {
    "vgg_small", "vgg_medium",
    "resnet_small", "resnet_medium",
}

# Models that accept norm_type (string) as a kwarg (ConvNeXtV2-based)
_NORM_TYPE_MODELS = {
    "convnextv2_tiny", "convnextv2_base", "convnextv2_small",
    "convnext_small", "convnext_medium",
}

MODEL_REGISTRY = {
    # ConvNeXtV2 - CIFAR
    "convnextv2_tiny": convnextv2_tiny,
    "convnextv2_base": convnextv2_base,
    "convnextv2_small": convnextv2_small,
    # ConvNeXtV2 - ImageNet
    "convnext_small": convnext_small,
    "convnext_medium": convnext_medium,
    # Simple CNN
    "simple_cnn": simple_cnn,
    # Torchvision ResNets (no configurable norm)
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    # Custom ResNets (configurable norm)
    "resnet_small": resnet_small,
    "resnet_medium": resnet_medium,
    # Custom VGGs (configurable norm)
    "vgg_small": vgg_small,
    "vgg_medium": vgg_medium,
}


def build_model(cfg: dict):
    """Build a model from config.

    Args:
        cfg: Configuration dictionary with model.name, model.num_classes,
             and optionally model.norm_layer.

    Returns:
        nn.Module: Instantiated model
    """
    name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    norm_layer = cfg["model"].get("norm_layer", "batchnorm")

    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    model_fn = MODEL_REGISTRY[name]

    if name in _NORM_AWARE_MODELS:
        return model_fn(num_classes=num_classes, norm_layer=norm_layer)
    elif name in _NORM_TYPE_MODELS:
        return model_fn(num_classes=num_classes, norm_type=norm_layer)
    else:
        return model_fn(num_classes=num_classes)


def register_model(name: str, model_fn):
    """Register a new model to the factory."""
    MODEL_REGISTRY[name] = model_fn


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
