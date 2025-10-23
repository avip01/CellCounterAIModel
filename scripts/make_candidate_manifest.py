# scripts/make_candidate_manifest.py
# Build a unified candidate manifest from your current structure:
# - data/processed/positives/*/<... numeric chip files ...>          -> label=1, group_id inferred
# - data/processed/negatives/crops/<... numeric chip files ...>      -> label=0, group_id mapped via negatives/summary.csv if possible
#
# Output:
# - data/manifests/candidate_manifest.csv with columns: filepath,label,group_id,source
#
# Run from repo root:
#   python scripts\make_candidate_manifest.py

from pathlib import Path
from collections import Counter
import pandas as pd
import re

# ---------- Config ----------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# Accept ONLY pure numeric filenames like 0000001234.png or 42.tif (reject overlays with letters)
IMG_NUMERIC_RGX = re.compile(r"^\d+\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)

# ---------- Helpers ----------
def posix(p: Path) -> str:
    return p.as_posix()

def looks_like_chip(fname: str) -> bool:
    """Accept ONLY numeric filenames; reject overlays/anything with letters."""
    return IMG_NUMERIC_RGX.fullmatch(fname) is not None

def find_group_id_from_path(p: Path) -> str:
    """
    Robustly infer a group_id like '1A', '2B', etc. from any part of the file path.
    Handles naming variants: '1a RGB', '1A-RGB', '1a_rgb', nested '.../1A/...', etc.
    """
    for part in [str(x) for x in p.parts][::-1]:
        m = re.search(r"([0-9]+[A-Za-z])", part, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""

def main():
    # Script is in scripts/, so ROOT is repo root
    ROOT = Path(__file__).resolve().parent.parent

    POS_DIR = ROOT / "data" / "processed" / "positives"
    NEG_DIR = ROOT / "data" / "processed" / "negatives"
    NEG_CROPS = NEG_DIR / "crops"
    MANIFESTS_DIR = ROOT / "data" / "manifests"
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH = MANIFESTS_DIR / "candidate_manifest.csv"

    rows = []

    # -------------------------
    # POSITIVES (label=1): include ONLY numeric-named images (ignore overlays)
    # -------------------------
    if POS_DIR.exists():
        group_dirs = [d for d in POS_DIR.iterdir() if d.is_dir()]
        for group_dir in sorted(group_dirs):
            # Collect any image under this group folder
            imgs = [p for p in group_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
            if not imgs:
                continue
            for img in imgs:
                if not img.is_file():
                    continue
                if not looks_like_chip(img.name):  # numeric-only chips
                    continue
                gid = find_group_id_from_path(img)
                rows.append({
                    "filepath": posix(img.relative_to(ROOT)),
                    "label": 1,
                    "group_id": gid,
                    "source": "candidate"
                })
    else:
        print(f"[WARN] Positives directory not found: {POS_DIR}")

    # -------------------------
    # NEGATIVES (label=0): try to map group_id via negatives/summary.csv by base filename
    # -------------------------
    neg_group_map = {}
    neg_summary_path = NEG_DIR / "summary.csv"
    if neg_summary_path.exists():
        try:
            df_neg_sum = pd.read_csv(neg_summary_path, dtype=str).fillna("")
            file_col = next((c for c in ["filepath","file","chip name","chip","path","filename"] if c in df_neg_sum.columns), None)
            group_col = next((c for c in ["group_id","group","survey id","survey_id","slide","image","origin_group"] if c in df_neg_sum.columns), None)
            if file_col and group_col:
                # map base filename -> group
                df_neg_sum["__base"] = df_neg_sum[file_col].map(lambda s: Path(str(s)).name.lower())
                neg_group_map = df_neg_sum.set_index("__base")[group_col].str.strip().to_dict()
            else:
                print("[INFO] negatives/summary.csv found but columns not recognized; negatives will have empty group_id.")
        except Exception as e:
            print(f"[INFO] Could not read negatives summary: {e}. Negatives will have empty group_id.")

    if NEG_CROPS.exists():
        for img in NEG_CROPS.rglob("*"):
            if not img.is_file():
                continue
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if not looks_like_chip(img.name):  # numeric-only chips
                continue
            base = img.name.lower()
            gid = neg_group_map.get(base, "")
            rows.append({
                "filepath": posix(img.relative_to(ROOT)),
                "label": 0,
                "group_id": gid if isinstance(gid, str) else str(gid),
                "source": "candidate"
            })
    else:
        print(f"[WARN] Negatives crops directory not found: {NEG_CROPS}")

    # -------------------------
    # Build DataFrame, drop dups, verify existence, normalize, save
    # -------------------------
    df = pd.DataFrame(rows, columns=["filepath","label","group_id","source"]).drop_duplicates(subset=["filepath"]).reset_index(drop=True)

    # keep only files that actually exist on disk
    exists_mask = df["filepath"].map(lambda p: (ROOT / p).exists())
    missing = df.loc[~exists_mask, "filepath"].tolist()
    if missing:
        print(f"[WARN] {len(missing)} paths not found; dropping (showing first 10):")
        for m in missing[:10]:
            print("  -", m)
    df = df.loc[exists_mask].reset_index(drop=True)

    # normalize types / blanks
    df["label"] = df["label"].astype(int)
    df["group_id"] = df["group_id"].astype(str).replace(["nan","NaN","None"], "").fillna("")
    df["source"] = df["source"].astype(str)

    # save
    df.to_csv(OUT_PATH, index=False)

    # -------------------------
    # Report
    # -------------------------
    total = len(df)
    pos_ct = int((df["label"] == 1).sum())
    neg_ct = total - pos_ct

    pos_groups = Counter(df.loc[df["label"] == 1, "group_id"])
    uniq_groups = sorted(g for g in set(df["group_id"]) if g)

    print(f"\nWritten: {OUT_PATH}")
    print(f"Total rows: {total:,} | Positives: {pos_ct:,} | Negatives: {neg_ct:,}")
    print("Positive counts by group (top 25):")
    for gid, cnt in pos_groups.most_common(25):
        print(f"  {gid or '<none>'}: {cnt}")
    print(f"\nUnique group_ids detected (non-empty): {len(uniq_groups)} -> {uniq_groups}")

    # Helpful hint if any positives came out with blank group_id
    blank_pos = int(((df["label"] == 1) & (df["group_id"] == "")).sum())
    if blank_pos > 0:
        print(f"[NOTE] {blank_pos} positive rows have empty group_id. Check folder naming for those groups.")

if __name__ == "__main__":
    main()
