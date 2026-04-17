import cv2
import numpy as np


def histogram_equalization(image):
    """Apply standard histogram equalization (grayscale)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.equalizeHist(gray)


def clahe_equalization(image, clip_limit=2.0, tile_grid=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def get_histogram(image):
    """Return histogram data for a grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten()
