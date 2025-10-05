"""Small CV helpers (padding, resize, NMS wrapper)."""
import cv2


def resize_keep_aspect(image, size):
    return cv2.resize(image, (size, size))
