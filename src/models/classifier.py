import torch
import torch.nn as nn

class Classifier(nn.Module):
    """Small configurable CNN for 128x128 crop classification."""
    def __init__(self, in_channels=3, num_classes=1, width=32, dropout=0.2):
        super().__init__()
        # small stack of Conv -> BN -> ReLU -> Pool blocks
        # final linear layer outputs a single logit for binary classification
        raise NotImplementedError

    @staticmethod
    def from_config(cfg: dict):
        return Classifier(in_channels=3, num_classes=1, width=cfg.get('width', 32))
