from __future__ import annotations
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 
try:
    from src.data.augmentations import get_train_transforms, get_val_transforms
except ImportError:
    print("Could not import from src.data.augmentations, trying relative import")
    from .augmentations import get_train_transforms, get_val_transforms


IMG_EXTS = {'.png','.jpg','.jpeg','.tif','.tiff'}


class CellDataset(Dataset):
    def __init__(self, data_root: str|Path, split_file: str|Path, transform=None):
        """
        We now accept a 'transform' argument.
        """
        root = Path(data_root)
        self.img_dir = root / 'raw'
        self.msk_dir = root / 'marked'
        with open(split_file, 'r') as f:
            names = [ln.strip() for ln in f if ln.strip()]
        
        self.items = [self.img_dir / n for n in names]
        
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ip = self.items[idx]
        mp = self.msk_dir / ip.name

        img = cv2.imread(str(ip))
        if img is None:
            print(f"Warning: Could not read image {ip}. Skipping.")
            return self[np.random.randint(0, len(self))] 

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mp}. Skipping.")
            return self[np.random.randint(0, len(self))]
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            x = augmented['image']
            y = augmented['mask']
        else:
            x = torch.from_numpy(img.transpose(2,0,1) / 255.0).float()
            y = torch.from_numpy(mask).float()

        y = y.unsqueeze(0) 

        return x, y, ip.name 


def make_loaders(data_root: str|Path, splits_dir: str|Path, batch_size=2, num_workers=0):
    import os
    sf_train = Path(splits_dir)/'train.txt'
    sf_val = Path(splits_dir)/'val.txt'
    assert sf_train.exists() and sf_val.exists(), f"Missing split files in {splits_dir}"

    train_tfm = get_train_transforms()
    val_tfm = get_val_transforms()
    train_set = CellDataset(data_root, sf_train, transform=train_tfm)
    val_set = CellDataset(data_root, sf_val, transform=val_tfm)
    
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )