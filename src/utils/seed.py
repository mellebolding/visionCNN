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
    # For max speed, set deterministic=False and benchmark=True
    # For reproducibility, set deterministic=True and benchmark=False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


__all__ = ['set_seed']