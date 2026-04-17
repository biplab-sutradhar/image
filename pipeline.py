import cv2
import numpy as np
import os

from stages import noise, enhancement, edges, morphology, segmentation
from metrics import compute_psnr, compute_ssim
from visualizer import show_stage, show_histograms, print_metrics


def run(image_path, output_dir="output"):
    print(f"\n=== Handwritten Notes Digitization Pipeline ===")
    print(f"Input: {image_path}\n")

    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # ── Stage 1: Noise Removal (camera grain from phone photos) ──────────────
    print("Stage 1: Noise Removal")
    noisy_gauss = noise.add_gaussian_noise(original, sigma=20)
    noisy_sp = noise.add_salt_pepper_noise(original, prob=0.04)
    denoised_gauss = noise.remove_gaussian_noise(noisy_gauss, ksize=5)
    denoised_median = noise.remove_salt_pepper_noise(noisy_sp, ksize=5)

    show_stage(
        "Stage 1 — Noise Removal (simulating phone camera grain)",
        [original, noisy_gauss, denoised_gauss, noisy_sp, denoised_median],
        ["Original", "Gaussian Noise", "Gaussian Filtered", "Salt & Pepper", "Median Filtered"],
        save_path=os.path.join(output_dir, "stage1_noise.png"),
    )
    print_metrics("Gaussian denoise", compute_psnr(original, denoised_gauss), compute_ssim(original, denoised_gauss))
    print_metrics("Median denoise  ", compute_psnr(original, denoised_median), compute_ssim(original, denoised_median))

    # Use the median-denoised image as input for remaining stages
    denoised = denoised_median

    # ── Stage 2: Contrast Enhancement (fix uneven lighting / shadows) ─────────
    print("\nStage 2: Contrast Enhancement")
    he = enhancement.histogram_equalization(denoised)
    clahe = enhancement.clahe_equalization(denoised, clip_limit=3.0, tile_grid=(8, 8))
    orig_hist = enhancement.get_histogram(denoised)
    clahe_hist = enhancement.get_histogram(clahe)

    show_stage(
        "Stage 2 — Contrast Enhancement (fixing uneven lighting on paper)",
        [denoised, he, clahe],
        ["Denoised", "Histogram Equalization", "CLAHE (used next)"],
        cmap_list=[None, "gray", "gray"],
        save_path=os.path.join(output_dir, "stage2_enhancement.png"),
    )
    show_histograms(orig_hist, clahe_hist, save_path=os.path.join(output_dir, "stage2_histograms.png"))
    print_metrics("Hist. Equalization", compute_psnr(original, he), compute_ssim(original, he))
    print_metrics("CLAHE             ", compute_psnr(original, clahe), compute_ssim(original, clahe))

    # ── Stage 3: Edge Detection (detect pen strokes / text boundaries) ────────
    print("\nStage 3: Edge Detection")
    sobel = edges.sobel_edges(clahe)
    laplacian = edges.laplacian_edges(clahe)
    canny = edges.canny_edges(clahe, low=30, high=100)

    show_stage(
        "Stage 3 — Edge Detection (pen strokes and text boundaries)",
        [clahe, sobel, laplacian, canny],
        ["CLAHE Enhanced", "Sobel", "Laplacian", "Canny (used next)"],
        cmap_list=["gray", "gray", "gray", "gray"],
        save_path=os.path.join(output_dir, "stage3_edges.png"),
    )

    # ── Stage 4: Morphological Operations (clean up broken strokes) ───────────
    print("\nStage 4: Morphological Operations")
    # Segment first to get binary text for morphology
    binary_raw, _ = segmentation.otsu_threshold(clahe)
    # Invert so text is white on black for morphological ops
    binary_inv = cv2.bitwise_not(binary_raw)
    closed_strokes = morphology.closing(binary_inv, ksize=3)   # connect broken strokes
    opened_clean = morphology.opening(closed_strokes, ksize=2)  # remove small noise dots
    # Restore: black text on white background
    cleaned = cv2.bitwise_not(opened_clean)

    show_stage(
        "Stage 4 — Morphological Operations (repairing broken pen strokes)",
        [binary_raw, binary_inv, closed_strokes, opened_clean, cleaned],
        ["Otsu Binary", "Inverted", "Closing (join strokes)", "Opening (remove noise)", "Restored"],
        cmap_list=["gray"] * 5,
        save_path=os.path.join(output_dir, "stage4_morphology.png"),
    )

    # ── Stage 5: Segmentation → Final Clean Document ──────────────────────────
    print("\nStage 5: Segmentation & Final Output")
    binary_otsu, thresh_val = segmentation.otsu_threshold(denoised)
    binary_adaptive = segmentation.adaptive_threshold(clahe, block_size=15, C=4)

    show_stage(
        "Stage 5 — Segmentation (separating text from paper background)",
        [original, binary_otsu, binary_adaptive, cleaned],
        ["Original Photo", f"Otsu (thresh={thresh_val:.0f})", "Adaptive Threshold", "Final Clean Output"],
        cmap_list=[None, "gray", "gray", "gray"],
        save_path=os.path.join(output_dir, "stage5_segmentation.png"),
    )

    # Save final clean image separately
    final_path = os.path.join(output_dir, "final_clean_document.png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(final_path, cleaned)
    print(f"  Final clean document saved: {final_path}")

    print(f"\nDone. Results saved to '{output_dir}/'")
