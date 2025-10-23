from __future__ import annotations
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


IMG_EXTS = {'.png','.jpg','.jpeg','.tif','.tiff'}


class ToTensor:
def __call__(self, img):
arr = np.array(img, dtype=np.float32)
if arr.ndim == 2:
arr = arr[..., None]
arr = arr.transpose(2, 0, 1) / 255.0
return torch.from_numpy(arr)


class Normalize:
def __init__(self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
self.mean = torch.tensor(mean)[:, None, None]
self.std = torch.tensor(std)[:, None, None]
def __call__(self, x):
if x.shape[0] == 1:
m, s = self.mean[0], self.std[0]
return (x - m) / s
return (x - self.mean) / self.std


class ToMaskTensor:
def __call__(self, mask_img):
arr = np.array(mask_img, dtype=np.float32)
if arr.ndim == 3:
arr = arr[..., 0]
arr = (arr > 127).astype(np.float32)
return torch.from_numpy(arr)[None, ...] # (1,H,W)


class CellDataset(Dataset):
def __init__(self, data_root: str|Path, split_file: str|Path):
root = Path(data_root)
self.img_dir = root / 'raw'
self.msk_dir = root / 'marked'
with open(split_file, 'r') as f:
names = [ln.strip() for ln in f if ln.strip()]
self.items = [self.img_dir / n for n in names]
self.to_tensor = ToTensor(); self.norm = Normalize(); self.to_mask = ToMaskTensor()


def __len__(self):
return len(self.items)


def __getitem__(self, idx):
ip = self.items[idx]
mp = self.msk_dir / ip.name
img = Image.open(ip).convert('RGB')
mask = Image.open(mp)
x = self.norm(self.to_tensor(img))
y = self.to_mask(mask)
return x, y, ip.name # base name used for CSV later




def make_loaders(data_root: str|Path, splits_dir: str|Path, batch_size=2, num_workers=0):
import os
sf_train = Path(splits_dir)/'train.txt'
sf_val = Path(splits_dir)/'val.txt'
assert sf_train.exists() and sf_val.exists(), f"Missing split files in {splits_dir}"
train_set = CellDataset(data_root, sf_train)
val_set = CellDataset(data_root, sf_val)
return (
DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
)