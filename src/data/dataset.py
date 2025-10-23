"""
Classification Dataset loader.
Located at: src/data/dataset.py
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .augmentations import get_train_transforms, get_val_transforms

IMG_EXTS = {'.png','.jpg','.jpeg','.tif','.tiff'}

def get_all_image_paths(data_root: Path) -> list[tuple[Path, int]]:
    """
    Finds all images and assigns them a label (1 for pos, 0 for neg).
    """
    data_root = Path(data_root)
    processed_dir = data_root / 'processed'
    
    items = []

    neg_dir = processed_dir / 'negatives' / 'crops'
    if not neg_dir.exists():
        print(f"Warning: Directory not found: {neg_dir}")
    else:
        print(f"Searching for negatives in: {neg_dir}")
        for ext in IMG_EXTS:
            for p in neg_dir.glob(f"*{ext}"):
                items.append((p, 0)) 

    pos_dir_base = processed_dir / 'positives' / 'crops'
    if not pos_dir_base.exists():
        print(f"Warning: Directory not found: {pos_dir_base}")
    else:
        print(f"Searching for positives in: {pos_dir_base} (and subfolders)")
        for ext in IMG_EXTS:
            for p in pos_dir_base.rglob(f"*{ext}"):
                 items.append((p, 1))

    print(f"Found {len(items)} total source images.")
    return items


class CellClassificationDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform=None):
        """
        A dataset for cell CLASSIFICATION.
        
        Args:
            items (list): A list of (path, label) tuples for this split.
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}. Returning a random item.")
            rand_idx = np.random.randint(0, len(self))
            return self[rand_idx]
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img, label_tensor.unsqueeze(0)

def make_loaders(data_root: str|Path, batch_size=32, num_workers=0):
    """
    Finds all images, creates 70/15/15 splits, and returns DataLoaders.
    """
    all_items = get_all_image_paths(Path(data_root))
        
    if not all_items:
        raise FileNotFoundError("No images found. Check your paths in dataset.py.")
    
    labels = [label for _, label in all_items]
    if len(np.unique(labels)) < 2:
        print("Warning: Your dataset only contains one class (all positive or all negative).")
        print("Splitting will not be stratified.")
        stratify = None
    else:
        stratify = labels

    train_items, temp_items = train_test_split(
        all_items, 
        test_size=0.30, 
        random_state=42, 
        stratify=stratify
    )
    
    temp_labels = [label for _, label in temp_items]
    if len(np.unique(temp_labels)) < 2:
        stratify_temp = None
    else:
        stratify_temp = temp_labels

    val_items, test_items = train_test_split(
        temp_items, 
        test_size=0.50, 
        random_state=42, 
        stratify=stratify_temp
    )

    print(f"Data split: {len(train_items)} train, {len(val_items)} val, {len(test_items)} test")
    
    train_tfm = get_train_transforms()
    val_tfm = get_val_transforms()

    train_set = CellClassificationDataset(train_items, transform=train_tfm)
    val_set = CellClassificationDataset(val_items, transform=val_tfm)
    test_set = CellClassificationDataset(test_items, transform=val_tfm)

    if len(train_set) == 0 or len(val_set) == 0:
        print("ERROR: One of the datasets is empty. Check your image paths.")
        return None, None, None

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader