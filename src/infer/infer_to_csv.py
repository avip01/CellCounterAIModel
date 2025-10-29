# src/infer/infer_to_csv.py
"""
Score ALL candidates and write detections.csv

Supports these candidate schemas:
  A) image_name,x_px,y_px,crop_path[,biomarker_id]     # <-- your new manifest
  B) image_name,x,y,crop_path[,biomarker_id]            # variant naming
  C) filepath,label,group_id,source[,x,y]               # legacy training-style

Output columns:
  image_name,biomarker_id,x_px,y_px,probability
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Match your training transforms
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# If your model is multi-class, set these:
BINARY = True         # False if multi-class
TARGET_CLASS = 1      # used only if BINARY=False


def build_model():
    # Make this match training exactly (swap to your mobilenet if needed)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1 if BINARY else 2)
    return m


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    # allow {"model": state_dict} or raw state_dict
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)
    return model


def preprocess():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def _resolve_crop_path(raw_field: str | Path, crops_root: Path) -> Path:
    raw = Path(str(raw_field).strip())
    if raw.is_absolute() or str(raw).startswith(("data/", "data\\")):
        return raw
    return crops_root / raw


def _parse_schema(fields: list[str]) -> str:
    f = {c.lower() for c in fields}
    if {"image_name", "crop_path"}.issubset(f) and ({"x_px","y_px"}.issubset(f) or {"x","y"}.issubset(f)):
        return "A"  # new manifest (prefers x_px/y_px)
    if {"filepath","group_id"}.issubset(f):
        return "C"  # legacy
    raise ValueError(f"Unrecognized schema. Columns={fields}")


def main():
    ap = argparse.ArgumentParser(description="Score ALL candidates -> detections.csv")
    ap.add_argument("--candidates", default="data/manifests/candidate_manifest.csv",
                    help="CSV with candidates.")
    ap.add_argument("--crops_root", default=".",
                    help="Root folder for candidate crops.")
    ap.add_argument("--weights", default="models/cell_classifier_best.pth",
                    help=".pth weights from training")
    ap.add_argument("--out_dir", default="experiments/exp_0001",
                    help="Where detections.csv is written")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = Path(args.candidates)
    crops_root = Path(args.crops_root)

    if not candidates_csv.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_csv}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model = load_checkpoint(model, args.weights, device)
    model.eval()
    tfm = preprocess()

    # Read candidates
    rows = []
    with candidates_csv.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames or []
        schema = _parse_schema(fields)

        for r in rdr:
            if schema == "A":
                image_name = r.get("image_name","").strip()
                crop_field = r.get("crop_path","").strip()
                # prefer x_px,y_px; fallback to x,y
                x_str = (r.get("x_px") or r.get("x") or "").strip()
                y_str = (r.get("y_px") or r.get("y") or "").strip()
                biomarker = r.get("biomarker_id")
            else:  # schema C (legacy)
                image_name = (r.get("group_id") or "unknown").strip()
                crop_field = (r.get("filepath") or "").strip()
                x_str = (r.get("x") or "").strip()
                y_str = (r.get("y") or "").strip()
                # map label->biomarker if present, else leave None
                biomarker = r.get("label")

            # coords
            try:
                x_val = float(x_str) if x_str != "" else None
                y_val = float(y_str) if y_str != "" else None
            except Exception:
                x_val, y_val = None, None

            # biomarker_id
            try:
                biomarker_id = int(biomarker) if biomarker not in (None, "", "None") else None
            except Exception:
                biomarker_id = None

            rows.append({
                "image_name": image_name,
                "crop_field": crop_field,
                "x_px": x_val,
                "y_px": y_val,
                "biomarker_id": biomarker_id,
            })

    print(f"Loaded {len(rows)} candidate rows from {candidates_csv}")

    # Score & write
    out_csv = out_dir / "detections.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name","biomarker_id","x_px","y_px","probability"])
        writer.writeheader()

        missing = 0
        for r in rows:
            img_name = r["image_name"]
            biomarker_id = r["biomarker_id"]
            # if not provided, default positives=1 (you can change this)
            if biomarker_id is None:
                biomarker_id = 1

            # coords as strings in output (blank if None)
            x_out = f"{r['x_px']:.2f}" if r["x_px"] is not None else ""
            y_out = f"{r['y_px']:.2f}" if r["y_px"] is not None else ""

            prob_str = ""
            crop_fp = _resolve_crop_path(r["crop_field"], crops_root)

            if crop_fp.exists():
                try:
                    img = Image.open(crop_fp).convert("RGB")
                    ximg = tfm(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(ximg)
                        if BINARY:
                            if logits.ndim == 2 and logits.size(1) == 1:
                                prob = torch.sigmoid(logits[:, 0])
                            elif logits.ndim == 1:
                                prob = torch.sigmoid(logits)
                            else:
                                # fallback: take first logit
                                prob = torch.sigmoid(logits.view(-1)[0:1])
                        else:
                            prob = torch.softmax(logits, dim=-1)[:, TARGET_CLASS]
                        p = float(prob.squeeze().detach().cpu().item())
                        prob_str = f"{p:.6f}"
                except Exception:
                    pass
            else:
                missing += 1

            writer.writerow({
                "image_name": img_name,
                "biomarker_id": biomarker_id,
                "x_px": x_out,
                "y_px": y_out,
                "probability": prob_str,
            })

    if missing:
        print(f"[warn] {missing} crop files listed in the manifest were not found on disk.")
    print(f"[OK] Wrote detections → {out_csv}")


if __name__ == "__main__":
    main()
