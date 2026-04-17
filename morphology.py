import cv2
import numpy as np


def _kernel(size=5):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def erode(image, ksize=5):
    return cv2.erode(image, _kernel(ksize), iterations=1)


def dilate(image, ksize=5):
    return cv2.dilate(image, _kernel(ksize), iterations=1)


def opening(image, ksize=5):
    """Erosion followed by dilation — removes small noise."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, _kernel(ksize))


def closing(image, ksize=5):
    """Dilation followed by erosion — fills small holes."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, _kernel(ksize))
