import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def _to_rgb(image):
    """Convert BGR to RGB for matplotlib display."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_stage(title, images, labels, cmap_list=None, save_path=None):
    """Display a row of images for a pipeline stage."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (img, label) in enumerate(zip(images, labels)):
        cmap = cmap_list[i] if cmap_list else ("gray" if len(img.shape) == 2 else None)
        axes[i].imshow(_to_rgb(img) if len(img.shape) == 3 else img, cmap=cmap)
        axes[i].set_title(label, fontsize=11)
        axes[i].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close()


def show_histograms(original_hist, enhanced_hist, save_path=None):
    """Compare histograms before and after enhancement."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(original_hist, color="steelblue")
    axes[0].set_title("Original Histogram")
    axes[0].set_xlim([0, 256])
    axes[1].plot(enhanced_hist, color="darkorange")
    axes[1].set_title("After CLAHE")
    axes[1].set_xlim([0, 256])
    plt.suptitle("Histogram Comparison", fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close()


def print_metrics(stage, psnr_val, ssim_val):
    print(f"  [{stage}] PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}")
