from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw




def draw_points_mask(h, w, xs, ys, r=4):
mask = Image.new('L', (w, h), 0)
d = ImageDraw.Draw(mask)
for x, y in zip(xs, ys):
d.ellipse((x-r, y-r, x+r, y+r), fill=255)
return np.array(mask)




def main():
ap = argparse.ArgumentParser()
ap.add_argument('--points_csv', type=str, default='data/manifests/points.csv')
ap.add_argument('--img_root', type=str, default='data/raw')
ap.add_argument('--out_dir', type=str, default='data/marked')
ap.add_argument('--radius', type=int, default=4)
args = ap.parse_args()


out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.points_csv)
df.columns = [c.strip().lower() for c in df.columns]
assert {'image_path','x','y'}.issubset(df.columns), "points.csv must have image_path,x,y"


for img_rel, g in df.groupby('image_path'):
p = Path(args.img_root) / Path(img_rel).name
im = Image.open(p).convert('RGB')
w, h = im.size
xs = g['x'].values; ys = g['y'].values
m = draw_points_mask(h, w, xs, ys, r=args.radius)
Image.fromarray(m).save(out / Path(img_rel).name)
print('Masks written to', str(out))


if __name__ == '__main__':
main()