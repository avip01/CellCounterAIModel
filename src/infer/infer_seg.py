from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.measure import label, regionprops
from src.models.unet_small import UNetSmall
try:
    from src.data.augmentations import get_val_transforms
except ImportError:
    print("Could not import from src.data.augmentations, trying relative import")
    from ..data.augmentations import get_val_transforms

@torch.no_grad()
def predict_prob(model, img, transform, dev):
    """
    Updated to use the albumentations transform pipeline.
    'img' is a PIL Image.
    """
    img_np = np.array(img)
    
    augmented = transform(image=img_np)
    x = augmented['image']
    
    x = x.unsqueeze(0).to(dev)

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

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")
    
    model = UNetSmall()
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.to(dev)
    model.eval()
    transform = get_val_transforms()

    img_dir = Path(args.data_root)/'raw'
    rows = []
    print(f"Processing images in {img_dir}...")
    
    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(list(img_dir.glob(f"*{ext}")))
    
    for p in sorted(img_paths):
        img = Image.open(p).convert('RGB')
        prob = predict_prob(model, img, transform, dev)
        
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
    if 'IMG_EXTS' not in globals():
        IMG_EXTS = {'.png','.jpg','.jpeg','.tif','.tiff'}
    main()