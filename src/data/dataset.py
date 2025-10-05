import torch
from torch.utils.data import Dataset
from typing import List, Dict

class CropDataset(Dataset):
    """
    Loads crops listed in manifest.csv and returns (image_tensor, label)
    Expects manifest with: image_path,label,source_image,...
    """
    def __init__(self, manifest_path: str, transforms=None):
        # load manifest into memory, build index
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        # read image, apply transforms, return tensor and label
        pass
