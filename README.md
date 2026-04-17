# Handwritten Notes Digitization — Image Processing Pipeline

> Coursework project for Image Processing

## Problem Statement

Handwritten notes captured using smartphones often suffer from poor image quality due to camera noise, uneven lighting, and low contrast — making them difficult to read, share, or archive digitally. This project implements an image processing pipeline that denoises the captured image, enhances contrast, detects text stroke boundaries, repairs broken strokes, and segments text from the background — producing a clean, readable digital document.

## Pipeline Stages

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | Gaussian & Median Filtering | Remove camera grain and noise |
| 2 | Histogram Equalization & CLAHE | Fix uneven lighting and low contrast |
| 3 | Sobel, Laplacian, Canny | Detect pen stroke and text boundaries |
| 4 | Morphological Operations | Repair broken strokes, remove noise dots |
| 5 | Otsu & Adaptive Thresholding | Segment text from paper background |

## Output

- Stage-wise comparison plots saved to `output/`
- Final clean document saved as `output/final_clean_document.png`
- PSNR and SSIM metrics printed at each stage

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# With your own notes image (jpg, png, webp supported)
python main.py path/to/your_notes.jpg

# Without an image (downloads a sample automatically)
python main.py
```

## Project Structure

```
image_pipeline/
├── main.py               # Entry point
├── pipeline.py           # Pipeline orchestration
├── metrics.py            # PSNR & SSIM computation
├── visualizer.py         # Plots and saves stage outputs
├── requirements.txt
├── stages/
│   ├── noise.py          # Noise addition & removal
│   ├── enhancement.py    # Histogram EQ & CLAHE
│   ├── edges.py          # Sobel, Laplacian, Canny
│   ├── morphology.py     # Erode, Dilate, Open, Close
│   └── segmentation.py   # Otsu, Adaptive threshold
└── output/               # All results saved here
```
