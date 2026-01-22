"""Simple CNN implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """5-layer CNN with batch normalization and dropout.
    
    Args:
        num_classes: Number of output classes
        channels: List of channel dimensions for conv layers [conv1, conv2, conv3]
        fc_dim: Dimension of first fully connected layer
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        channels: list[int] = [64, 128, 256], 
        fc_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, channels[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # After 3 pooling layers: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(channels[2] * 4 * 4, fc_dim)
        self.bn4 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        x = self.fc2(x)
        return x
