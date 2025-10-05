"""IO helpers for images and manifests."""
from typing import List


def save_manifest(manifest: List[dict], path: str):
    import csv

    if not manifest:
        return
    keys = manifest[0].keys()
    with open(path, "w", newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(manifest)
