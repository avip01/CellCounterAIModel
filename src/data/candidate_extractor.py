"""
Deterministic OpenCV-based candidate detector.
Single responsibility: from big image -> list of bounding boxes.
"""

from typing import List, Tuple
import numpy as np

BBox = Tuple[int, int, int, int]  # xmin, ymin, w, h

def find_candidates_bgr(image_bgr: np.ndarray, cfg: dict) -> List[BBox]:
    """
    Find yellow connected components and return tight bounding boxes.
    - image_bgr: OpenCV BGR image (unmarked)
    - cfg: dict with HSV thresholds, min_area, morph sizes
    """
    # Implementation detail: convert to HSV, threshold, morphology, connectedComponents
    raise NotImplementedError
