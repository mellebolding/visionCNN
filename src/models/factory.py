"""Model factory for building models from config."""
import torchvision.models as tv_models

from .convnextv2 import convnextv2_tiny, convnextv2_base
from .simple_cnn import simple_cnn


def resnet18(num_classes: int = 1000, pretrained: bool = False):
    """Create ResNet18 model."""
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    if num_classes != 1000:
        model.fc = __import__('torch').nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet34(num_classes: int = 1000, pretrained: bool = False):
    """Create ResNet34 model."""
    weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet34(weights=weights)
    if num_classes != 1000:
        model.fc = __import__('torch').nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet50(num_classes: int = 1000, pretrained: bool = False):
    """Create ResNet50 model."""
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.resnet50(weights=weights)
    if num_classes != 1000:
        model.fc = __import__('torch').nn.Linear(model.fc.in_features, num_classes)
    return model


MODEL_REGISTRY = {
    "convnextv2_tiny": convnextv2_tiny,
    "convnextv2_base": convnextv2_base,
    "simple_cnn": simple_cnn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def build_model(cfg: dict):
    """Build a model from config.
    
    Args:
        cfg: Configuration dictionary with model.name and model.num_classes
        
    Returns:
        nn.Module: Instantiated model
    """
    name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    
    model_fn = MODEL_REGISTRY[name]
    return model_fn(num_classes=num_classes)


def register_model(name: str, model_fn):
    """Register a new model to the factory.
    
    Args:
        name: Model name to register
        model_fn: Function that returns a model instance
    """
    MODEL_REGISTRY[name] = model_fn


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
