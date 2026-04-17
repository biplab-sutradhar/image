import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, prob=0.05):
    """Add salt-and-pepper noise to an image."""
    noisy = image.copy()
    rng = np.random.default_rng()
    mask = rng.random(image.shape[:2])
    noisy[mask < prob / 2] = 0
    noisy[mask > 1 - prob / 2] = 255
    return noisy


def remove_gaussian_noise(image, ksize=5):
    """Apply Gaussian blur to reduce Gaussian noise."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def remove_salt_pepper_noise(image, ksize=5):
    """Apply median filter to reduce salt-and-pepper noise."""
    return cv2.medianBlur(image, ksize)
