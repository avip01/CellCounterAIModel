import torch


def _bin(x):
return (torch.sigmoid(x) > 0.5).float()


def iou(logits, target, eps=1e-6):
pred = _bin(logits)
inter = (pred*target).sum(dim=(1,2,3))
union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
return ((inter + eps)/(union + eps)).mean().item()


def dice(logits, target, eps=1e-6):
pred = _bin(logits)
inter = (pred*target).sum(dim=(1,2,3))
denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
return ((2*inter + eps)/(denom + eps)).mean().item()