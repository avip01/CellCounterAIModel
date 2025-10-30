import argparse, os
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn
from scipy.ndimage import maximum_filter, label, center_of_mass

# ---------- model helpers ----------
def strip_prefix(state, prefix):
    if all(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state

def build_resnet18(num_classes=1):
    import torchvision.models as tvm
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device)
    state = next((ckpt.get(k) for k in ("state_dict","model","net","weights")
                  if isinstance(ckpt.get(k), dict)), ckpt)
    for pref in ("module.","model.","net."):
        if any(k.startswith(pref) for k in state.keys()):
            state = strip_prefix(state, pref)
    m = build_resnet18(1).to(device)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Sliding-window inference with local-maxima selection")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", default="experiments/quickcheck")
    ap.add_argument("--window", type=int, default=224)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--peak_radius", type=int, default=18)
    ap.add_argument("--nms_radius", type=int, default=36)
    ap.add_argument("--circle_radius", type=int, default=10)
    ap.add_argument("--thickness", type=int, default=-1)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--topk", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    print(f"[debug] image: {H}x{W}")

    model = load_model(args.weights, device)

    # ---- sliding window passes ----
    xs, ys, probs = [], [], []
    for y in range(0, H - args.window + 1, args.stride):
        for x in range(0, W - args.window + 1, args.stride):
            crop = img[y:y+args.window, x:x+args.window, :]
            arr = (crop[:, :, ::-1].astype(np.float32)/255.).transpose(2,0,1)[None]  # RGB tensor, no norm
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(arr).to(device))).item()
            xs.append(x + args.window//2)
            ys.append(y + args.window//2)
            probs.append(p)

    xs, ys, probs = np.array(xs), np.array(ys), np.array(probs)
    print(f"[debug] windows={len(probs)}  prob min={probs.min():.4f}  max={probs.max():.4f}")

    # ---- build probability map ----
    grid_h = (H - args.window)//args.stride + 1
    grid_w = (W - args.window)//args.stride + 1
    prob_map = probs.reshape(grid_h, grid_w)
    prob_map = cv2.resize(prob_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # ---- find local peaks ----
    peaks_mask = (prob_map == maximum_filter(prob_map, size=args.peak_radius))
    labeled, n = label(peaks_mask)
    coms = np.array(center_of_mass(prob_map, labeled, range(1, n+1)))
    if len(coms)==0:
        print("[OK] No peaks found"); return
    peak_xy = np.fliplr(coms)  # (x,y)
    peak_scores = prob_map[coms[:,0].astype(int), coms[:,1].astype(int)]
    pts = np.column_stack([peak_xy, peak_scores])

    # ---- threshold / top-k ----
    if args.topk and args.topk < len(pts):
        k = np.argsort(-pts[:,2])[:args.topk]
        pts = pts[k]
    else:
        pts = pts[pts[:,2] >= args.threshold]

    # ---- radius-based NMS ----
    def nms_radius(points_xy, scores, radius):
        if len(points_xy)==0: return []
        order = np.argsort(-scores)
        keep, used = [], np.zeros(len(points_xy), bool)
        rr = radius*radius
        for i in order:
            if used[i]: continue
            keep.append(i)
            dx = points_xy[:,0]-points_xy[i,0]; dy = points_xy[:,1]-points_xy[i,1]
            used |= (dx*dx+dy*dy)<=rr; used[i]=True
        return keep
    if len(pts):
        keep = nms_radius(pts[:, :2], pts[:,2], args.nms_radius)
        pts = pts[keep]

    # ---- save results ----
    stem = Path(args.image).stem
    csv_path = os.path.join(args.out_dir, f"{stem}_peaks.csv")
    np.savetxt(csv_path, pts, fmt=["%.1f","%.1f","%.4f"], delimiter=",",
               header="x,y,prob", comments="")
    print(f"[OK] CSV → {csv_path} (detections={len(pts)})")

    overlay = img.copy()
    for x,y,_ in pts:
        cv2.circle(overlay, (int(x),int(y)), args.circle_radius, (255,255,255), args.thickness)
    out_img = os.path.join(args.out_dir, f"{stem}_peaks_overlay.png")
    cv2.imwrite(out_img, overlay)
    print(f"[OK] Overlay → {out_img}")

if __name__ == "__main__":
    main()
