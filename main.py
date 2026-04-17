import sys
import os
import urllib.request
import cv2
import numpy as np

from pipeline import run

SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
SAMPLE_PATH = "sample_images/lenna.png"


def download_sample():
    """Download the standard Lenna test image if no image is provided."""
    os.makedirs("sample_images", exist_ok=True)
    print(f"Downloading sample image...")
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    print(f"Saved to {SAMPLE_PATH}")
    return SAMPLE_PATH


def create_synthetic_image():
    """Create a synthetic test image with shapes if download fails."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 180
    cv2.circle(img, (150, 150), 80, (60, 60, 200), -1)
    cv2.rectangle(img, (280, 100), (430, 250), (200, 60, 60), -1)
    cv2.ellipse(img, (256, 380), (120, 70), 0, 0, 360, (60, 180, 60), -1)
    cv2.putText(img, "Test Image", (140, 490), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2)
    path = "sample_images/synthetic.png"
    os.makedirs("sample_images", exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Created synthetic test image: {path}")
    return path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        try:
            image_path = download_sample()
        except Exception:
            print("Download failed, using synthetic image.")
            image_path = create_synthetic_image()

    run(image_path, output_dir="output")
