import cv2
import numpy as np


def _to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image


def sobel_edges(image):
    """Detect edges using Sobel operator."""
    gray = _to_gray(image)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(np.clip(magnitude, 0, 255))


def laplacian_edges(image):
    """Detect edges using Laplacian operator."""
    gray = _to_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(lap), 0, 255))


def canny_edges(image, low=50, high=150):
    """Detect edges using Canny algorithm."""
    gray = _to_gray(image)
    return cv2.Canny(gray, low, high)
