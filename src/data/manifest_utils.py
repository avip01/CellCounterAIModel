"""Helpers to read/write manifests and splits."""
import csv
from typing import List


def read_manifest(path: str) -> List[dict]:
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        return list(rdr)
