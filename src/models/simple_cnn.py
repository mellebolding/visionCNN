"""Simple CNN model variants for CIFAR-style datasets."""
from .simple_cnn_impl import SimpleCNN


def simple_cnn(num_classes: int = 10, **kwargs) -> SimpleCNN:
    """Simple CNN baseline - balanced model for CIFAR10/100 (32x32 images).
    
    Parameters: ~1.5M
    Architecture: 3 conv layers [64, 128, 256] + 2 FC layers [512, num_classes]
    """
    return SimpleCNN(
        num_classes=num_classes,
        channels=[64, 128, 256],
        fc_dim=512,
        dropout=0.3,
        **kwargs
    )


def simple_cnn_small(num_classes: int = 10, **kwargs) -> SimpleCNN:
    """Simple CNN Small - lightweight variant.
    
    Parameters: ~0.4M
    Architecture: 3 conv layers [32, 64, 128] + 2 FC layers [256, num_classes]
    """
    return SimpleCNN(
        num_classes=num_classes,
        channels=[32, 64, 128],
        fc_dim=256,
        dropout=0.3,
        **kwargs
    )


def simple_cnn_large(num_classes: int = 10, **kwargs) -> SimpleCNN:
    """Simple CNN Large - more capacity variant.
    
    Parameters: ~3.8M
    Architecture: 3 conv layers [96, 192, 384] + 2 FC layers [768, num_classes]
    """
    return SimpleCNN(
        num_classes=num_classes,
        channels=[96, 192, 384],
        fc_dim=768,
        dropout=0.3,
        **kwargs
    )
