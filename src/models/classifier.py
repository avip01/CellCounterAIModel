"""Simple model wrapper (placeholder)."""
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(128 * 128 * 3, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.net(x)
