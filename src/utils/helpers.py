import os, random
import numpy as np
import torch


def seed_everything(seed: int = 42):
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
os.makedirs(path, exist_ok=True)


def device():
return torch.device('cuda' if torch.cuda.is_available() else 'cpu')