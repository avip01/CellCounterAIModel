"""Dataset and dataloader utilities (placeholder)."""
from torch.utils.data import Dataset
from typing import Any


class CellDataset(Dataset):
    def __init__(self, manifest_path: str, transform=None):
        self.manifest_path = manifest_path
        self.transform = transform
        # ... load manifest
        self.items = []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx) -> Any:
        # return (image, label) placeholder
        return None
