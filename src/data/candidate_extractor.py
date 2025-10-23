import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Dict, Tuple

BBox = Tuple[int, int, int, int]  # xmin, ymin, w, h
Candidate = Dict  # {'bbox': (x,y,w,h), 'area': int, 'mask': np.ndarray}



def find_candidates_bgr(
    image_bgr,
    hsv_yellow_lower,
    hsv_yellow_upper,
    min_area: int,
    morph_kernel_size: int,
) -> List[Candidate]:
    """
    Find yellow cell components in a BGR image.

    Returns a list of candidates, each with:
        'bbox': (xmin, ymin, w, h)
        'area': contour area (px)
        'mask': binary mask (same as image) where component pixels are 255
    """
    if image_bgr is None: #something is wrong with the image file
        return []

    # Color thresholding in HSV space
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_yellow_lower, dtype=np.uint8)
    upper = np.array(hsv_yellow_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of connected components
    contours, i = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #I need a blank argument after contours for this to work

    # Extract the candidates from the image
    candidates: List[Candidate] = []
    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        comp_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(comp_mask, [cnt], -1, color=255, thickness=-1)
        candidates.append({"bbox": (x, y, w, h), "area": area, "mask": comp_mask})

    return candidates


def draw_candidates_on_image(image_bgr: np.ndarray, candidates: List[Candidate], color=(0, 255, 255)):
    """
    Return a copy of image_bgr with bounding boxes and small index labels drawn.
    color is BGR tuple (default: yellow-ish).
    """
    out = image_bgr.copy()
    for idx, c in enumerate(candidates):
        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"{idx}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    # DEBUG / TEST, THIS CODE SHOULD NEVER ACTUALLY BE RAN
    import sys
    from pathlib import Path

    img_path = Path("data/raw/1a RGB.tif")
    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print("Failed to read image with OpenCV.")
        sys.exit(1)

   # parameter tuning to use in the make_manifest.py file.
    candidates = find_candidates_bgr(img, min_area=20, morph_kernel_size=5,
                                     hsv_yellow_lower=(20, 40, 40), hsv_yellow_upper=(35, 255, 255))

    out = draw_candidates_on_image(img, candidates, color=(0, 255, 255))

    # Ngl I just used AI to write this because I was having errors
    try:
        cv2.namedWindow("candidates_debug", cv2.WINDOW_NORMAL)
        cv2.imshow("candidates_debug", out)
        print("Press any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"cv2.imshow failed ({e}). Falling back to saving + matplotlib.")
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12,8))
        plt.imshow(out_rgb)
        plt.title(f"Candidates: {img_path.name}")
        plt.axis("off")
        plt.show(block=True)
        fallback = Path.cwd() / f"{img_path.stem}_candidates_debug.png"
        cv2.imwrite(str(fallback), out)
        print(f"Saved debug image to: {fallback}")