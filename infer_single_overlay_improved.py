# infer_single_overlay_improved.py — robust candidate union + gated detection with eval
# Pipeline: candidates (gate ∪ hsv) → CNN (224+norm) → NMS → snap → NMS → overlay + CSVs (+ optional PR/F1)

import argparse, os, csv
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
    """Return centroid candidates from HSV yellow mask and the mask image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=2)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)

    num, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centers: List[Tuple[int, int]] = []
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_area <= a <= max_area:
            cx, cy = cent[i]
            centers.append((int(round(cx)), int(round(cy))))
    return centers, mask

# -------------------- Edge + Green + Yellow gate --------------------
def _float_rgb(img_bgr: np.ndarray):
    img = img_bgr.astype(np.float32) / 255.0
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    return R, G, B

def build_edge_green_yellow_gate(
    img_bgr: np.ndarray,
    edge_band_px=5, canny_lo=10, canny_hi=40,
    green_k=0.30,          # slightly stricter default
    tR=0.58, tG=0.58, tB=0.34,
    delta_rg=0.18, rg_ratio=0.82,
    min_cc_area=8, max_cc_area=220,
    keep_border=True, border_px=6,
    debug=False, debug_prefix="debug"
):
    """
    cand=True where:
      - on red edge band (laminin),
      - green positive (adaptive),
      - yellow co-mix (R&G high, B low, R≈G).
    Returns a cleaned mask already area-filtered to [min_cc_area, max_cc_area].
    """
    H, W = img_bgr.shape[:2]
    R, G, B = _float_rgb(img_bgr)

    red_u8 = np.clip(R * 255, 0, 255).astype(np.uint8)
    red_blur = cv2.GaussianBlur(red_u8, (0, 0), 1.0)
    edges = cv2.Canny(red_blur, canny_lo, canny_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_band_px + 1, 2 * edge_band_px + 1))
    edge_band = cv2.dilate(edges, kernel) > 0

    g_blur = cv2.GaussianBlur(G, (0, 0), 1.0)
    g_thr = float(np.mean(g_blur) + green_k * np.std(g_blur))
    green_mask = (g_blur > g_thr)

    yellow = (R > tR) & (G > tG) & (B < tB)
    yellow &= (np.abs(R - G) < delta_rg)
    rg_min = np.minimum(R, G) + 1e-6
    rg_max = np.maximum(R, G) + 1e-6
    yellow &= (rg_min / rg_max) > rg_ratio

    cand = edge_band & green_mask & yellow

    if keep_border:
        inner = np.zeros_like(cand, dtype=bool)
        inner[border_px:H - border_px, border_px:W - border_px] = True
        cand &= inner

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, k3)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k3).astype(bool)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    good = np.zeros_like(cand, dtype=bool)
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_cc_area <= a <= max_cc_area:
            good |= (lab == i)

    if debug:
        def out(img, name): cv2.imwrite(f"{debug_prefix}_{name}.png", (255 * img.astype(np.uint8)))
        out(edge_band, "edgeband"); out(green_mask, "green"); out(yellow, "yellow"); out(good, "gatemask")

    return good, {"edge_band": edge_band, "green": green_mask, "yellow": yellow}

# -------------------- crops, NMS & snap --------------------
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

def nms_radius(points: np.ndarray, scores: np.ndarray, radius: int) -> List[int]:
    if len(points) == 0: return []
    order = np.argsort(-scores)
    keep_idx, used = [], np.zeros(len(points), dtype=bool)
    rr = radius * radius
    for i in order:
        if used[i]: continue
        keep_idx.append(i)
        dx = points[:, 0] - points[i, 0]
        dy = points[:, 1] - points[i, 1]
        used |= (dx * dx + dy * dy) <= rr
        used[i] = True
    return keep_idx

def snap_to_yellow_centroids(img_bgr,
                             pts_xy: np.ndarray,
                             search_r=14,
                             h_lo=18, h_hi=42, s_lo=150, v_lo=170,
                             min_area=8, max_area=220):
    """Return snapped_xy (K,2) and indices keep_idx into pts_xy that survived snapping."""
    if len(pts_xy) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))
    mask = cv2.medianBlur(mask, 3)

    snapped, keep_idx = [], []
    for i, (x, y) in enumerate(pts_xy.astype(int)):
        x0, y0 = max(0, x - search_r), max(0, y - search_r)
        x1, y1 = min(W, x + search_r + 1), min(H, y + search_r + 1)
        sub = mask[y0:y1, x0:x1]
        num, lab, stats, centroids = cv2.connectedComponentsWithStats(sub, connectivity=8)
        best_d2, best_xy = None, None
        cx0, cy0 = x - x0, y - y0
        for j in range(1, num):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area: continue
            cx, cy = centroids[j]
            d2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
            if (best_d2 is None) or (d2 < best_d2):
                best_d2, best_xy = d2, (x0 + cx, y0 + cy)
        if best_xy is not None:
            snapped.append(best_xy); keep_idx.append(i)

    if len(snapped) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.array(snapped, dtype=np.float32), np.array(keep_idx, dtype=np.int32)

# -------------------- CSV helpers --------------------
def save_detections_csv(csv_path: str, coords_xy: np.ndarray, scores: np.ndarray):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "score"])
        if len(coords_xy):
            for (x, y), s in zip(coords_xy, scores):
                w.writerow([int(x), int(y), float(s)])

def save_artifacts_csv(csv_path: str, rows: List[Tuple[str, str]]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artifact_type", "path"])
        for t, p in rows:
            w.writerow([t, os.path.abspath(p)])

# -------------------- evaluation (optional) --------------------
def load_truth_coords(path: str) -> np.ndarray:
    pts = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == 0 and any(k in line.lower() for k in ["x", "y"]):
                continue  # header
            parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            if len(parts) < 2: continue
            # allow formats: x,y  or  x,y,score  or  <name>,<...>,x,y,(score)
            try:
                # try last two as x,y
                x = float(parts[-2]); y = float(parts[-1])
            except:
                continue
            pts.append((x, y))
    return np.array(pts, dtype=np.float32)

def prf1(dets: np.ndarray, gts: np.ndarray, tol_px: float = 10.0) -> Tuple[float, float, float]:
    if len(gts) == 0 and len(dets) == 0:
        return 1.0, 1.0, 1.0
    if len(gts) == 0:
        return 0.0, 1.0, 0.0
    if len(dets) == 0:
        return 1.0, 0.0, 0.0
    # greedy bipartite matching by distance
    dets = dets.astype(np.float32)
    gts  = gts.astype(np.float32)
    used_gt = np.zeros(len(gts), dtype=bool)
    tp = 0
    for d in dets:
        d2 = np.sum((gts - d)**2, axis=1)
        j = int(np.argmin(d2))
        if not used_gt[j] and np.sqrt(d2[j]) <= tol_px:
            used_gt[j] = True
            tp += 1
    fp = len(dets) - tp
    fn = len(gts) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser("Microscopy candidate union + gated detection (224+norm)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", default="experiments/quickcheck")

    # ***** match training *****
    ap.add_argument("--window", type=int, default=224)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--mean", default="0.485,0.456,0.406")
    ap.add_argument("--std",  default="0.229,0.224,0.225")
    ap.add_argument("--color_order", choices=["rgb", "bgr"], default="rgb")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # classifier + post-processing
    ap.add_argument("--threshold", type=float, default=0.90)   # tuned for your score band
    ap.add_argument("--nms_radius", type=int, default=16)
    ap.add_argument("--circle_radius", type=int, default=8)
    ap.add_argument("--thickness", type=int, default=-1)

    # HSV params
    ap.add_argument("--h_lo", type=int, default=15)
    ap.add_argument("--h_hi", type=int, default=40)
    ap.add_argument("--s_lo", type=int, default=80)
    ap.add_argument("--v_lo", type=int, default=80)
    ap.add_argument("--min_area", type=int, default=6)
    ap.add_argument("--max_area", type=int, default=500)

    # Gate params (slightly stricter defaults)
    ap.add_argument("--use_edge_gate", action="store_true")
    ap.add_argument("--edge_band_px", type=int, default=5)
    ap.add_argument("--canny_lo", type=int, default=10)
    ap.add_argument("--canny_hi", type=int, default=40)
    ap.add_argument("--green_k", type=float, default=0.30)
    ap.add_argument("--tR", type=float, default=0.58)
    ap.add_argument("--tG", type=float, default=0.58)
    ap.add_argument("--tB", type=float, default=0.34)
    ap.add_argument("--delta_rg", type=float, default=0.18)
    ap.add_argument("--rg_ratio", type=float, default=0.82)
    ap.add_argument("--min_cc_area_gate", type=int, default=8)
    ap.add_argument("--max_cc_area_gate", type=int, default=220)
    ap.add_argument("--keep_border", action="store_true", default=True)
    ap.add_argument("--border_px", type=int, default=6)
    ap.add_argument("--debug_masks", action="store_true")

    # Candidate strategy (we'll always do union: gate ∪ hsv)
    ap.add_argument("--candidate_union_radius", type=int, default=8)

    # Snap
    ap.add_argument("--snap", action="store_true", default=True)
    ap.add_argument("--snap_search_r", type=int, default=14)
    ap.add_argument("--snap_h_lo", type=int, default=18)
    ap.add_argument("--snap_h_hi", type=int, default=48)
    ap.add_argument("--snap_s_lo", type=int, default=140)
    ap.add_argument("--snap_v_lo", type=int, default=165)
    ap.add_argument("--snap_min_area", type=int, default=6)
    ap.add_argument("--snap_max_area", type=int, default=240)

    # Optional evaluation
    ap.add_argument("--truth_csv", type=str, default=None)
    ap.add_argument("--match_tol_px", type=float, default=10.0)

    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    artifacts: List[Tuple[str, str]] = []
    img = load_image_bgr(args.image)
    H, W = img.shape[:2]
    print(f"[debug] image loaded: {H}x{W}")
    stem = Path(args.image).stem

    # Gate
    gate_mask = None
    if args.use_edge_gate:
        dbg_prefix = os.path.join(args.out_dir, f"{stem}")
        gate_mask, _ = build_edge_green_yellow_gate(
            img_bgr=img,
            edge_band_px=args.edge_band_px, canny_lo=args.canny_lo, canny_hi=args.canny_hi,
            green_k=args.green_k,
            tR=args.tR, tG=args.tG, tB=args.tB, delta_rg=args.delta_rg, rg_ratio=args.rg_ratio,
            min_cc_area=args.min_cc_area_gate, max_cc_area=args.max_cc_area_gate,
            keep_border=args.keep_border, border_px=args.border_px,
            debug=args.debug_masks, debug_prefix=dbg_prefix
        )
        if args.debug_masks:
            for suf in ("edgeband","green","yellow","gatemask"):
                artifacts.append(("debug_mask", f"{dbg_prefix}_{suf}.png"))

    # Candidates: gate ∪ hsv with de-dupe
    cand_xy: List[Tuple[int,int]] = []
    if gate_mask is not None:
        num, lab, stats, cent = cv2.connectedComponentsWithStats(gate_mask.astype(np.uint8), connectivity=8)
        for i in range(1, num):
            a = stats[i, cv2.CC_STAT_AREA]
            if args.min_area <= a <= args.max_area:
                cx, cy = cent[i]
                cand_xy.append((int(round(cx)), int(round(cy))))
        print(f"[stage] gate centroids: {len(cand_xy)}")

    hsv_pts, yellow_mask = find_yellow_candidates(
        img, h_lo=args.h_lo, h_hi=args.h_hi, s_lo=args.s_lo, v_lo=args.v_lo,
        min_area=args.min_area, max_area=args.max_area
    )
    print(f"[stage] hsv centroids: {len(hsv_pts)}")

    # union + radius de-dup
    all_pts = cand_xy + hsv_pts
    if len(all_pts):
        P = np.array(all_pts, dtype=np.int32)
        ones = np.ones(len(P), dtype=np.float32)
        keep = nms_radius(P, ones, args.candidate_union_radius)
        cand_xy = [tuple(map(int, P[i])) for i in keep]
    else:
        cand_xy = []
    print(f"[stage] total candidates after union+dedupe: {len(cand_xy)}")

    if yellow_mask is not None:
        ypath = os.path.join(args.out_dir, f"{stem}_yellowmask.png")
        cv2.imwrite(ypath, yellow_mask); artifacts.append(("yellowmask", ypath))

    # Visualize candidates
    cand_vis = img.copy()
    for (x,y) in cand_xy: cv2.circle(cand_vis, (int(x),int(y)), 6, (0,255,255), -1)
    cpath = os.path.join(args.out_dir, f"{stem}_candidates.png")
    cv2.imwrite(cpath, cand_vis); artifacts.append(("candidates_vis", cpath))

    det_csv = os.path.join(args.out_dir, f"{stem}_detections.csv")
    art_csv = os.path.join(args.out_dir, f"{stem}_artifacts.csv")

    if len(cand_xy) == 0:
        out_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
        cv2.imwrite(out_path, img); artifacts.append(("overlay", out_path))
        save_detections_csv(det_csv, np.empty((0,2),dtype=np.int32), np.empty((0,),dtype=np.float32))
        artifacts.append(("csv", det_csv)); save_artifacts_csv(art_csv, artifacts)
        print("[OK] Saved (no candidates) and CSVs.");
        if args.truth_csv:
            gts = load_truth_coords(args.truth_csv)
            p,r,f = prf1(np.empty((0,2)), gts, tol_px=args.match_tol_px)
            print(f"[eval] P={p:.3f} R={r:.3f} F1={f:.3f}")
        return

    # ----- classify (224 + normalization) -----
    model = load_model(args.weights, device=device, num_classes=1)
    crops = [center_crop(img, x, y, args.window) for (x, y) in cand_xy]
    x_tensor = to_tensor_bchw_uint8(crops, args.color_order).to(device)
    if args.normalize:
        mean = torch.tensor(tuple(float(v) for v in args.mean.split(",")), device=device).view(1, 3, 1, 1)
        std  = torch.tensor(tuple(float(v) for v in args.std.split(",")),  device=device).view(1, 3, 1, 1)
        x_tensor = (x_tensor - mean) / std

    with torch.no_grad():
        logits = model(x_tensor).squeeze()
        if not torch.is_floating_point(logits): logits = logits.float()
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    print("[stage] prob stats – min {:.3f} / max {:.3f} / mean {:.3f}".format(probs.min(), probs.max(), probs.mean()))

    keep = probs >= args.threshold
    sel_xy = np.array(cand_xy, dtype=np.int32)[keep]
    sel_sc = probs[keep]
    print(f"[stage] kept after threshold {args.threshold}: {len(sel_xy)}")

    # NMS → SNAP (safe) → NMS
    if len(sel_xy) and args.nms_radius > 0:
        k = nms_radius(sel_xy, sel_sc, args.nms_radius)
        sel_xy, sel_sc = sel_xy[k], sel_sc[k]
        print(f"[stage] after pre-snap NMS (r={args.nms_radius}): {len(sel_xy)}")

    if len(sel_xy) and args.snap:
        pre_xy, pre_sc = sel_xy.copy(), sel_sc.copy()
        snapped_xy, keep_idx = snap_to_yellow_centroids(
            img_bgr=img, pts_xy=sel_xy, search_r=args.snap_search_r,
            h_lo=args.snap_h_lo, h_hi=args.snap_h_hi,
            s_lo=args.snap_s_lo, v_lo=args.snap_v_lo,
            min_area=args.snap_min_area, max_area=args.snap_max_area
        )
        sel_xy = snapped_xy.astype(np.int32); sel_sc = sel_sc[keep_idx]
        print(f"[stage] after snap kept: {len(sel_xy)}")
        if len(sel_xy) == 0:
            print("[warn] snap removed all detections; reverting to pre-snap")
            sel_xy, sel_sc = pre_xy, pre_sc

    if len(sel_xy) and args.nms_radius > 0:
        k = nms_radius(sel_xy, sel_sc, args.nms_radius)
        sel_xy, sel_sc = sel_xy[k], sel_sc[k]
        print(f"[stage] after post-snap NMS (r={args.nms_radius}): {len(sel_xy)}")

    # 3) draw + CSVs
    overlay = img.copy()
    for (x,y) in sel_xy: cv2.circle(overlay, (int(x), int(y)), args.circle_radius, (255,255,255), args.thickness)
    out_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
    cv2.imwrite(out_path, overlay); artifacts.append(("overlay", out_path))
    save_detections_csv(det_csv, sel_xy, sel_sc); artifacts.append(("csv", det_csv))
    save_artifacts_csv(art_csv, artifacts)
    print(f"[OK] Saved overlay → {out_path} (detections: {len(sel_xy)})")
    print(f"[OK] Wrote CSVs: {det_csv}, {art_csv}")

    # 4) optional evaluation
    if args.truth_csv:
        gts = load_truth_coords(args.truth_csv)
        p, r, f = prf1(sel_xy.astype(np.float32), gts, tol_px=args.match_tol_px)
        print(f"[eval] tol={args.match_tol_px:.1f}px  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

if __name__ == "__main__":
    main()
