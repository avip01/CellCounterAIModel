# infer_single_overlay.py — candidate-based overlay (yellow -> classify -> draw)
import argparse, os
from pathlib import Path
from typing import List, Tuple, Any, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn


# ---------- utils ----------
def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_tensor_bchw_uint8(crops: List[np.ndarray], color_order: str) -> torch.Tensor:
    arr = np.stack(crops, axis=0)  # N,H,W,C (BGR from cv2)
    if color_order.lower() == "rgb":
        arr = arr[:, :, :, ::-1]   # BGR->RGB
    arr = arr.astype(np.float32) / 255.0
    arr = arr.transpose(0, 3, 1, 2)  # N,C,H,W
    return torch.from_numpy(arr)


def strip_prefix_if_present(state_dict, prefix: str):
    keys = list(state_dict.keys())
    if keys and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def build_resnet18(num_classes: int) -> nn.Module:
    import torchvision.models as tvm
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def extract_state_dict(ckpt: Any, checkpoint_key: Optional[str]):
    if isinstance(ckpt, dict):
        if checkpoint_key and checkpoint_key in ckpt and isinstance(ckpt[checkpoint_key], dict):
            return ckpt[checkpoint_key]
        for k in ("state_dict", "model_state", "model", "net", "weights"):
            v = ckpt.get(k)
            if isinstance(v, dict) and v and isinstance(next(iter(v.values())), torch.Tensor):
                return v
        if ckpt and isinstance(next(iter(ckpt.values())), torch.Tensor):
            return ckpt
    if hasattr(ckpt, "keys") and ckpt and isinstance(next(iter(ckpt.values())), torch.Tensor):
        return ckpt
    return {}


def load_model(weights_path: str, device: torch.device, num_classes: int) -> nn.Module:
    ckpt = torch.load(weights_path, map_location=device)
    state = extract_state_dict(ckpt, None)
    model = build_resnet18(num_classes=num_classes).to(device)
    for pref in ("module.", "model.", "net."):
        if any(k.startswith(pref) for k in state.keys()):
            state = strip_prefix_if_present(state, pref)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("[debug] model.fc weight shape:", tuple(model.fc.weight.shape))
    return model


# ---------- candidate extraction (yellow) ----------
def find_yellow_candidates(img_bgr: np.ndarray,
                           h_lo=15, h_hi=40,
                           s_lo=80, v_lo=80,
                           min_area=5, max_area=500) -> List[Tuple[int, int]]:
    """Return list of (x,y) centers likely to be yellow puncta."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.medianBlur(mask, 3)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: List[Tuple[int, int]] = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    return centers


def center_crop(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    r = size // 2
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(w, x + r), min(h, y + r)
    crop = img[y0:y1, x0:x1, :]
    if crop.shape[0] != size or crop.shape[1] != size:
        pad_top = (size - crop.shape[0]) // 2
        pad_bottom = size - crop.shape[0] - pad_top
        pad_left = (size - crop.shape[1]) // 2
        pad_right = size - crop.shape[1] - pad_left
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_REFLECT_101)
    return crop


# ---------- main ----------
def main():
    import sys
    ap = argparse.ArgumentParser("Candidate-based overlay (yellow->classify->circles)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", default="experiments/quickcheck")
    ap.add_argument("--window", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--nms_radius", type=int, default=16)
    ap.add_argument("--circle_radius", type=int, default=8)
    ap.add_argument("--thickness", type=int, default=-1)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    # preprocessing
    ap.add_argument("--color_order", choices=["rgb", "bgr"], default="rgb")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--mean", default="0.485,0.456,0.406")
    ap.add_argument("--std", default="0.229,0.224,0.225")
    # yellow detection params
    ap.add_argument("--h_lo", type=int, default=15)
    ap.add_argument("--h_hi", type=int, default=40)
    ap.add_argument("--s_lo", type=int, default=80)
    ap.add_argument("--v_lo", type=int, default=80)
    ap.add_argument("--min_area", type=int, default=5)
    ap.add_argument("--max_area", type=int, default=500)

    args = ap.parse_args()
    print("[startup]", sys.executable)
    print("[args]", vars(args))
    sys.stdout.flush()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    img = load_image_bgr(args.image)
    H, W = img.shape[:2]
    print(f"[debug] image loaded: {H}x{W}")

    # 1) candidates
    cand_xy = find_yellow_candidates(
        img, h_lo=args.h_lo, h_hi=args.h_hi, s_lo=args.s_lo, v_lo=args.v_lo,
        min_area=args.min_area, max_area=args.max_area
    )
    print(f"[debug] yellow candidates: {len(cand_xy)}")

    if len(cand_xy) == 0:
        out_path = os.path.join(args.out_dir, f"{Path(args.image).stem}_overlay.png")
        cv2.imwrite(out_path, img)
        print(f"[OK] Saved (no candidates) → {out_path}")
        return

    # 2) classify
    model = load_model(args.weights, device=device, num_classes=1)
    crops = [center_crop(img, x, y, args.window) for (x, y) in cand_xy]
    x_tensor = to_tensor_bchw_uint8(crops, args.color_order).to(device)

    if args.normalize:
        mean = torch.tensor(tuple(float(v) for v in args.mean.split(",")), device=device).view(1, 3, 1, 1)
        std = torch.tensor(tuple(float(v) for v in args.std.split(",")), device=device).view(1, 3, 1, 1)
        x_tensor = (x_tensor - mean) / std

    with torch.no_grad():
        logits = model(x_tensor).squeeze()
        if not torch.is_floating_point(logits):
            logits = logits.float()
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    probs = np.asarray(probs, dtype=np.float32)
    print("[debug] prob stats on candidates:",
          "min", float(probs.min()), "max", float(probs.max()), "mean", float(probs.mean()))

    keep = probs >= args.threshold
    sel_xy = np.array(cand_xy, dtype=np.int32)[keep]
    sel_sc = probs[keep]

    # NMS
    def nms_radius(points: np.ndarray, scores: np.ndarray, radius: int) -> List[int]:
        if len(points) == 0:
            return []
        order = np.argsort(-scores)
        keep_idx, used = [], np.zeros(len(points), dtype=bool)
        rr = radius * radius
        for i in order:
            if used[i]:
                continue
            keep_idx.append(i)
            dx = points[:, 0] - points[i, 0]
            dy = points[:, 1] - points[i, 1]
            used |= (dx * dx + dy * dy) <= rr
            used[i] = True
        return keep_idx

    if len(sel_xy) and args.nms_radius > 0:
        k = nms_radius(sel_xy, sel_sc, args.nms_radius)
        sel_xy = sel_xy[k]

    # 3) draw overlay
    overlay = img.copy()
    for (x, y) in sel_xy:
        cv2.circle(overlay, (int(x), int(y)), args.circle_radius, (255, 255, 255), args.thickness)

    out_path = os.path.join(args.out_dir, f"{Path(args.image).stem}_overlay.png")
    cv2.imwrite(out_path, overlay)
    print(f"[OK] Saved overlay → {out_path}  (detections: {len(sel_xy)})")


# ---------- script entry ----------
if __name__ == "__main__":
    main()
