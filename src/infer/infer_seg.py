from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.measure import label, regionprops
from src.models.unet_small import UNetSmall
from src.data.dataset import ToTensor, Normalize


@torch.no_grad()
def predict_prob(model, img):
x = Normalize()(ToTensor()(img)).unsqueeze(0)
logits = model(x)
prob = torch.sigmoid(logits)[0,0].cpu().numpy()
return prob




def write_csv(rows, out_csv):
with open(out_csv, 'w', newline='') as f:
w = csv.writer(f)
w.writerow(['image_name','class','x','y'])
w.writerows(rows)




def main():
ap = argparse.ArgumentParser()
ap.add_argument('--data_root', type=str, required=True)
ap.add_argument('--weights', type=str, required=True)
ap.add_argument('--out_csv', type=str, default='experiments/exp_0001/centroids.csv')
ap.add_argument('--default_class', type=str, default='cell')
ap.add_argument('--prob_thresh', type=float, default=0.5)
ap.add_argument('--min_area', type=int, default=5)
args = ap.parse_args()


model = UNetSmall()
model.load_state_dict(torch.load(args.weights, map_location='cpu'))
model.eval()


img_dir = Path(args.data_root)/'raw'
rows = []
for p in sorted(img_dir.iterdir()):
if p.suffix.lower() not in ['.png','.jpg','.jpeg','.tif','.tiff']:
continue
img = Image.open(p).convert('RGB')
prob = predict_prob(model, img)
mask = (prob > args.prob_thresh).astype(np.uint8)
lbl = label(mask)
for r in regionprops(lbl):
if r.area < args.min_area:
continue
y, x = r.centroid
rows.append([p.name, args.default_class, float(x), float(y)])
write_csv(rows, args.out_csv)
print(f"Wrote {len(rows)} detections → {args.out_csv}")


if __name__ == '__main__':
main()