"""Utilities for setting random seeds for reproducibility."""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ['set_seed']