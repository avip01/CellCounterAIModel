# scripts/make_candidate_manifest.py
# Build candidate_manifest.csv with real coordinates from summary.csv / cords.csv
# Layout (per-user):
#   Positives: data/processed/positives/crops/<GROUP>/(summary.csv|cords.csv) + chips in same folder
#   Negatives: data/processed/negatives/(summary.csv|cords.csv) at root (chips under negatives/ or negatives/crops/)
#
# Output:
#   data/manifests/candidate_manifest.csv with columns:
#     image_name, biomarker_id, crop_path, x_px, y_px
#
# Run:
#   (.venv) python scripts/make_candidate_manifest.py

from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent

POS_CROPS_ROOT = ROOT / "data" / "processed" / "positives" / "crops"
NEG_ROOT       = ROOT / "data" / "processed" / "negatives"

OUT = ROOT / "data" / "manifests" / "candidate_manifest.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def clean_chip_name(val: str) -> str:
    """Normalize chip name from Windows/absolute path to repo-relative-ish POSIX-like."""
    if not isinstance(val, str):
        return ""
    s = val.strip().replace("\\", "/")
    # Drop drive letter like C:/...
    m = re.match(r"^[A-Za-z]:/(.*)$", s)
    return m.group(1) if m else s


def try_find_chip(chip_name: str, search_roots: list[Path]) -> Path:
    """Try to locate the chip file under a list of roots, tolerant to ext mismatches."""
    if not chip_name:
        return Path()

    norm = clean_chip_name(chip_name)
    # 1) Try as ROOT-relative
    p = (ROOT / norm)
    if p.exists() and p.is_file():
        return p

    # 2) Try as absolute path (already normalized)
    p_abs = Path(norm)
    if p_abs.is_absolute() and p_abs.exists() and p_abs.is_file():
        return p_abs

    # 3) Try under provided roots (exact name)
    name_only = Path(norm).name
    for base in search_roots:
        cand = base / name_only
        if cand.exists() and cand.is_file():
            return cand

    # 4) Try alt extensions in provided roots
    stem = Path(norm).stem
    for base in search_roots:
        for ext in IMG_EXTS:
            cand = base / f"{stem}{ext}"
            if cand.exists() and cand.is_file():
                return cand

    # 5) Last resort: recursive search by stem.*
    for base in search_roots:
        hits = list(base.rglob(stem + ".*"))
        for h in hits:
            if h.is_file():
                return h

    return Path()


def load_coords_csv(folder: Path):
    """Prefer summary.csv, fallback to cords.csv."""
    for name in ("summary.csv", "cords.csv"):
        f = folder / name
        if f.exists():
            try:
                return pd.read_csv(f, dtype=str).fillna(""), f
            except Exception as e:
                print(f"[WARN] Could not read {f}: {e}")
    return None, None


def get_col(df: pd.DataFrame, choices: list[str]):
    choices_lower = [c.lower() for c in df.columns]
    for want in choices:
        for i, have in enumerate(choices_lower):
            if have == want.lower():
                return df.columns[i]
    return None


def extract_columns(df: pd.DataFrame):
    col_image = get_col(df, ["image", "slide", "parent"])
    col_x     = get_col(df, ["x", "x_px", "xcoord", "x_center"])
    col_y     = get_col(df, ["y", "y_px", "ycoord", "y_center"])
    col_chip  = get_col(df, ["chip name", "chip", "filename", "file", "filepath"])
    return col_image, col_x, col_y, col_chip


def process_positives(rows: list[dict]):
    """data/processed/positives/crops/<GROUP>/coords + chips in same group folder."""
    if not POS_CROPS_ROOT.exists():
        print(f"[WARN] Positives crops root missing: {POS_CROPS_ROOT}")
        return

    group_dirs = [d for d in POS_CROPS_ROOT.iterdir() if d.is_dir()]
    if not group_dirs:
        print(f"[WARN] No group folders under {POS_CROPS_ROOT}")
        return

    for gdir in sorted(group_dirs):
        df, src = load_coords_csv(gdir)
        if df is None:
            print(f"[WARN] No summary/cords.csv in {gdir}")
            continue

        col_image, col_x, col_y, col_chip = extract_columns(df)
        if not all([col_image, col_x, col_y, col_chip]):
            print(f"[WARN] {src} missing required columns (need image/x/y/chip)")
            continue

        for _, r in df.iterrows():
            chip_name = str(r[col_chip]).strip()
            if not chip_name:
                continue

            crop_path = try_find_chip(chip_name, [gdir])
            if not (crop_path and crop_path.exists() and crop_path.suffix.lower() in IMG_EXTS):
                print(f"[WARN] Missing crop for {chip_name} in {gdir.name}")
                continue

            try:
                x_px = float(str(r[col_x]).strip())
                y_px = float(str(r[col_y]).strip())
            except ValueError:
                print(f"[WARN] Bad x/y in {src}: {r[col_x]}, {r[col_y]}")
                continue

            image_name = Path(str(r[col_image]).strip()).name
            if image_name.lower().endswith(tuple(ext.lstrip(".") for ext in IMG_EXTS)):
                image_name = Path(image_name).stem
            else:
                image_name = Path(image_name).stem  # safe default

            rows.append({
                "image_name": image_name,
                "biomarker_id": 1,
                "crop_path": crop_path.relative_to(ROOT).as_posix(),
                "x_px": f"{x_px:.2f}",
                "y_px": f"{y_px:.2f}",
            })


def process_negatives(rows: list[dict]):
    """
    data/processed/negatives at ROOT has summary/cords.csv.
    Chips may live under negatives/ or negatives/crops/.
    """
    if not NEG_ROOT.exists():
        print(f"[WARN] Negatives root missing: {NEG_ROOT}")
        return

    df, src = load_coords_csv(NEG_ROOT)
    if df is None:
        print(f"[INFO] No negatives summary/cords.csv; skipping negatives.")
        return

    col_image, col_x, col_y, col_chip = extract_columns(df)
    if not all([col_image, col_x, col_y, col_chip]):
        print(f"[WARN] {src} missing required columns (need image/x/y/chip) for negatives.")
        return

    # Search roots for negatives
    search_roots = [NEG_ROOT]
    crops_dir = NEG_ROOT / "crops"
    if crops_dir.exists():
        search_roots.insert(0, crops_dir)

    for _, r in df.iterrows():
        chip_name = str(r[col_chip]).strip()
        if not chip_name:
            continue

        crop_path = try_find_chip(chip_name, search_roots)
        if not (crop_path and crop_path.exists() and crop_path.suffix.lower() in IMG_EXTS):
            print(f"[WARN] Missing negative crop for {chip_name}")
            continue

        try:
            x_px = float(str(r[col_x]).strip())
            y_px = float(str(r[col_y]).strip())
        except ValueError:
            print(f"[WARN] Bad x/y in {src}: {r[col_x]}, {r[col_y]}")
            continue

        image_name = Path(str(r[col_image]).strip()).name
        image_name = Path(image_name).stem

        rows.append({
            "image_name": image_name,
            "biomarker_id": 0,
            "crop_path": crop_path.relative_to(ROOT).as_posix(),
            "x_px": f"{x_px:.2f}",
            "y_px": f"{y_px:.2f}",
        })


def main():
    rows: list[dict] = []

    process_positives(rows)
    process_negatives(rows)

    df = pd.DataFrame(rows, columns=["image_name", "biomarker_id", "crop_path", "x_px", "y_px"]).drop_duplicates()
    df.to_csv(OUT, index=False)

    # Quick QA summary
    n = len(df)
    n_pos = (df["biomarker_id"] == 1).sum()
    n_neg = n - n_pos
    print(f"[OK] Wrote {n:,} candidates → {OUT}")
    print(f"      Positives: {n_pos:,} | Negatives: {n_neg:,}")
    if n == 0:
        print("[HINT] If zero, check the folder paths above and that CSV column names match (image/x/y/chip).")


if __name__ == "__main__":
    main()
