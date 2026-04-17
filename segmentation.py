import cv2
import numpy as np


def _to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image


def otsu_threshold(image):
    """Apply Otsu's thresholding for automatic segmentation."""
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_val, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, thresh_val


def adaptive_threshold(image, block_size=11, C=2):
    """Apply adaptive thresholding for uneven lighting."""
    gray = _to_gray(image)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )


def count_objects(binary_image):
    """Count distinct objects (connected components) in a binary image."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)
    # Exclude background (label 0)
    count = num_labels - 1
    return count, labels, stats
