from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import make_loaders
from src.models.unet_small import UNetSmall
from src.utils.metrics import iou, dice
from src.utils.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.helpers import seed_everything, ensure_dir, device




def train_epoch(model, loader, criterion, optimzr, dev):
model.train(); total=0.0
for x, y, _ in loader:
x, y = x.to(dev), y.to(dev)
optimzr.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward(); optimzr.step()
total += loss.item()*x.size(0)
return total/len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, dev):
model.eval(); total=0.0; miou=0.0; mdice=0.0; n=0
for x, y, _ in loader:
x, y = x.to(dev), y.to(dev)
logits = model(x)
loss = criterion(logits, y)
b = x.size(0)
total += loss.item()*b
miou += iou(logits, y)*b
mdice += dice(logits, y)*b
n += b
return total/len(loader.dataset), miou/n, mdice/n




def main():
ap = argparse.ArgumentParser()
ap.add_argument('--data_root', type=str, required=True)
ap.add_argument('--splits_dir', type=str, required=True)
ap.add_argument('--outdir', type=str, default='experiments/exp_0001')
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--batch_size', type=int, default=2)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--patience', type=int, default=5)
args = ap.parse_args()


seed_everything(42)
ensure_dir(args.outdir)
dev = device()


train_loader, val_loader = make_loaders(args.data_root, args.splits_dir, args.batch_size)
model = UNetSmall().to(dev)
criterion = nn.BCEWithLogitsLoss()
optimzr = optim.Adam(model.parameters(), lr=args.lr)
early = EarlyStopping(patience=args.patience)
ckpt = ModelCheckpoint(args.outdir, 'best.pth')


for epoch in range(args.epochs):
tr_loss = train_epoch(model, train_loader, criterion, optimzr, dev)
val_loss, val_iou, val_dice = eval_epoch(model, val_loader, criterion, dev)
ckpt.step(val_loss, model)
early.step(val_loss)
print(f"Epoch {epoch+1}/{args.epochs} | train {tr_loss:.4f} | val {val_loss:.4f} | IoU {val_iou:.3f} | Dice {val_dice:.3f}")
if early.stop:
print('Early stopping.')
break


if __name__ == '__main__':
main()