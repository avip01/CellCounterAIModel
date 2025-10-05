"""Small app-level wrapper for running inference."""
from typing import List


def run_batch(image_paths: List[str]):
    """Run inference on a batch of images (placeholder).

    Returns a list of dicts with image_id and count.
    """
    results = []
    for p in image_paths:
        # placeholder result
        results.append({"image": p, "count": 0})
    return results
