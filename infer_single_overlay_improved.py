# infer_single_overlay_improved.py — candidate-based overlay with edge+green+yellow gate
# Pipeline: HSV (yellow) → edge/green/yellow gate (DotDotGoose) → classify crops → NMS → (optional) snap-to-yellow → draw

import argparse, os
from pathlib import Path
from typing import List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


# -------------------- utils --------------------
def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_tensor_bchw_uint8(crops: List[np.ndarray], color_order: str) -> torch.Tensor:
    # crops are BGR (cv2)
    arr = np.stack(crops, axis=0)  # N,H,W,C
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


# -------------------- HSV candidate extraction --------------------
def find_yellow_candidates(img_bgr: np.ndarray,
                           h_lo=15, h_hi=40,
                           s_lo=80, v_lo=80,
                           min_area=5, max_area=500) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Returns candidate centers and the binary HSV mask used to find them.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # cleanup pipeline
    mask = cv2.medianBlur(mask, 5)
    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    kernel_medium = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

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
    return centers, mask


# -------------------- Edge + Green + Yellow gate (DotDotGoose) --------------------
def _float_rgb(img_bgr: np.ndarray):
    img = img_bgr.astype(np.float32) / 255.0
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    return R, G, B


def build_edge_green_yellow_gate(
    img_bgr: np.ndarray,
    edge_band_px=5, canny_lo=10, canny_hi=40,
    green_k=0.25,
    tR=0.55, tG=0.55, tB=0.35, delta_rg=0.20, rg_ratio=0.75,
    min_cc_area=6, max_cc_area=200,
    keep_border=False, border_px=8,
    debug=False, debug_prefix="debug"
):
    """
    Returns (cand_mask, debug_dict).
    cand_mask=True only where all 3 conditions hold:
      - on red edge band (laminin),
      - green positive (CD19),
      - yellow co-mix (R&G high, B low, R≈G).
    """
    H, W = img_bgr.shape[:2]
    R, G, B = _float_rgb(img_bgr)

    # 1) Red edge band
    red_u8 = np.clip(R * 255, 0, 255).astype(np.uint8)
    red_blur = cv2.GaussianBlur(red_u8, (0, 0), 1.0)
    edges = cv2.Canny(red_blur, canny_lo, canny_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_band_px + 1, 2 * edge_band_px + 1))
    edge_band = cv2.dilate(edges, kernel) > 0

    # 2) Green positivity (adaptive)
    g_blur = cv2.GaussianBlur(G, (0, 0), 1.0)
    g_thr = float(np.mean(g_blur) + green_k * np.std(g_blur))
    green_mask = (g_blur > g_thr)

    # 3) Yellow co-mix
    yellow = (R > tR) & (G > tG) & (B < tB)
    yellow &= (np.abs(R - G) < delta_rg)
    rg_min = np.minimum(R, G) + 1e-6
    rg_max = np.maximum(R, G) + 1e-6
    yellow &= (rg_min / rg_max) > rg_ratio

    cand = edge_band & green_mask & yellow

    # Remove border if requested
    if not keep_border:
        inner = np.zeros_like(cand, dtype=bool)
        inner[border_px:H - border_px, border_px:W - border_px] = True
        cand &= inner

    # Clean small speckles and fill small gaps
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, k3)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k3).astype(bool)

    # Area filter on connected components
    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    good = np.zeros_like(cand, dtype=bool)
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_cc_area <= a <= max_cc_area:
            good |= (lab == i)

    if debug:
        def out(img, name):
            cv2.imwrite(f"{debug_prefix}_{name}.png", (255 * img.astype(np.uint8)))
        out(edge_band, "edgeband")
        out(green_mask, "green")
        out(yellow, "yellow")
        out(good, "gatemask")

    return good, {"edge_band": edge_band, "green": green_mask, "yellow": yellow}


# crop around a point with reflect padding to fixed size
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
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    return crop


# -------------------- optional snap-to-yellow refinement --------------------
def snap_to_yellow_centroids(img_bgr, pts_xy, search_r=14,
                             h_lo=18, h_hi=42, s_lo=150, v_lo=170,
                             min_area=8, max_area=220):
    """
    For each (x,y), find closest yellow-blob centroid within search_r using an HSV mask
    and keep only points that can snap to a valid blob (area in [min_area, max_area]).
    """
    if len(pts_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)

    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))
    mask = cv2.medianBlur(mask, 3)

    snapped = []
    for (x, y) in pts_xy.astype(int):
        x0, y0 = max(0, x - search_r), max(0, y - search_r)
        x1, y1 = min(W, x + search_r + 1), min(H, y + search_r + 1)
        sub = mask[y0:y1, x0:x1]

        num, lab, stats, centroids = cv2.connectedComponentsWithStats(sub, connectivity=8)
        best_d2, best_xy = None, None
        cx0, cy0 = x - x0, y - y0
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            cx, cy = centroids[i]
            d2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
            if (best_d2 is None) or (d2 < best_d2):
                best_d2, best_xy = d2, (x0 + cx, y0 + cy)

        if best_xy is not None:
            snapped.append(best_xy)

    return np.array(snapped, dtype=np.float32)


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser("Candidate-based overlay with DotDotGoose gate")
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
    ap.add_argument("--normalize", action="store_true")  # NOTE: you trained WITHOUT normalization
    ap.add_argument("--mean", default="0.485,0.456,0.406")
    ap.add_argument("--std", default="0.229,0.224,0.225")
    # yellow detection params
    ap.add_argument("--h_lo", type=int, default=15)
    ap.add_argument("--h_hi", type=int, default=40)
    ap.add_argument("--s_lo", type=int, default=80)
    ap.add_argument("--v_lo", type=int, default=80)
    ap.add_argument("--min_area", type=int, default=5)
    ap.add_argument("--max_area", type=int, default=500)
    # snapping refinement
    ap.add_argument("--snap", action="store_true", help="snap detections to local yellow centroids")
    ap.add_argument("--snap_search_r", type=int, default=14)
    ap.add_argument("--snap_h_lo", type=int, default=18)
    ap.add_argument("--snap_h_hi", type=int, default=42)
    ap.add_argument("--snap_s_lo", type=int, default=150)
    ap.add_argument("--snap_v_lo", type=int, default=170)
    ap.add_argument("--snap_min_area", type=int, default=8)
    ap.add_argument("--snap_max_area", type=int, default=220)

    # ------------ NEW: edge + co-localization gate ------------
    ap.add_argument("--use_edge_gate", action="store_true")
    ap.add_argument("--edge_band_px", type=int, default=5)
    ap.add_argument("--canny_lo", type=int, default=10)
    ap.add_argument("--canny_hi", type=int, default=40)

    ap.add_argument("--green_k", type=float, default=0.25)

    ap.add_argument("--tR", type=float, default=0.55)
    ap.add_argument("--tG", type=float, default=0.55)
    ap.add_argument("--tB", type=float, default=0.35)
    ap.add_argument("--delta_rg", type=float, default=0.20)
    ap.add_argument("--rg_ratio", type=float, default=0.75)

    ap.add_argument("--min_cc_area", type=int, default=6)
    ap.add_argument("--max_cc_area", type=int, default=200)

    ap.add_argument("--keep_border", action="store_true")
    ap.add_argument("--border_px", type=int, default=8)

    ap.add_argument("--debug_masks", action="store_true")
    # -----------------------------------------------------------

    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    img = load_image_bgr(args.image)
    H, W = img.shape[:2]
    print(f"[debug] image loaded: {H}x{W}")

    # 1) candidates via HSV
    cand_xy, mask = find_yellow_candidates(
        img, h_lo=args.h_lo, h_hi=args.h_hi, s_lo=args.s_lo, v_lo=args.v_lo,
        min_area=args.min_area, max_area=args.max_area
    )
    print(f"[debug] yellow candidates (HSV): {len(cand_xy)}")

    # 1b) Edge+co-localization gate (DotDotGoose rule)
    gate_mask = None
    if args.use_edge_gate:
        stem = Path(args.image).stem
        dbg_prefix = os.path.join(args.out_dir, f"{stem}")
        gate_mask, _dbg = build_edge_green_yellow_gate(
            img_bgr=img,
            edge_band_px=args.edge_band_px, canny_lo=args.canny_lo, canny_hi=args.canny_hi,
            green_k=args.green_k,
            tR=args.tR, tG=args.tG, tB=args.tB, delta_rg=args.delta_rg, rg_ratio=args.rg_ratio,
            min_cc_area=args.min_cc_area, max_cc_area=args.max_cc_area,
            keep_border=args.keep_border, border_px=args.border_px,
            debug=args.debug_masks, debug_prefix=f"{dbg_prefix}"
        )
        if len(cand_xy):
            cand_xy = [(x, y) for (x, y) in cand_xy if gate_mask[int(y), int(x)]]
        print(f"[debug] candidates after edge-gate: {len(cand_xy)}")

    # ---- debug saves for visibility ----
    stem = Path(args.image).stem
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_yellowmask.png"), mask)
    cand_vis = img.copy()
    for (x, y) in cand_xy:
        cv2.circle(cand_vis, (int(x), int(y)), 6, (0, 255, 255), -1)
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_candidates.png"), cand_vis)

    if len(cand_xy) == 0:
        out_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
        cv2.imwrite(out_path, img)
        print(f"[OK] Saved (no candidates) → {out_path}")
        return

    # 2) classify crops
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
    print("[debug] prob stats on candidates: min", float(probs.min()),
          "max", float(probs.max()), "mean", float(probs.mean()))

    keep = probs >= args.threshold
    sel_xy = np.array(cand_xy, dtype=np.int32)[keep]
    sel_sc = probs[keep]

    # small radius NMS to avoid near-duplicates
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
        sel_sc = sel_sc[k]

    # 2b) optional snapping to local yellow centroids
    if len(sel_xy) and args.snap:
        snapped_xy = snap_to_yellow_centroids(
            img, sel_xy,
            search_r=args.snap_search_r,
            h_lo=args.snap_h_lo, h_hi=args.snap_h_hi,
            s_lo=args.snap_s_lo, v_lo=args.snap_v_lo,
            min_area=args.snap_min_area, max_area=args.snap_max_area
        )
        sel_xy = snapped_xy.astype(np.int32)
        print(f"[debug] after snap: {len(sel_xy)}")

    # 3) draw overlay
    overlay = img.copy()
    for (x, y) in sel_xy:
        cv2.circle(overlay, (int(x), int(y)), args.circle_radius, (255, 255, 255), args.thickness)

    out_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
    cv2.imwrite(out_path, overlay)
    print(f"[OK] Saved overlay → {out_path}  (detections: {len(sel_xy)})")


if __name__ == "__main__":
    main()
