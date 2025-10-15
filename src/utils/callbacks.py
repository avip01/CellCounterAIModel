import torch, os


class EarlyStopping:
def __init__(self, patience=5, min_delta=0.0):
self.patience = patience
self.min_delta = min_delta
self.best = None
self.count = 0
self.stop = False
def step(self, value):
if self.best is None or value < self.best - self.min_delta:
self.best = value
self.count = 0
else:
self.count += 1
if self.count >= self.patience:
self.stop = True


class ModelCheckpoint:
def __init__(self, outdir: str, filename="best.pth"):
self.path = os.path.join(outdir, filename)
os.makedirs(outdir, exist_ok=True)
self.best = None
def step(self, metric_value, model):
if self.best is None or metric_value < self.best:
self.best = metric_value
torch.save(model.state_dict(), self.path)
return True
return False