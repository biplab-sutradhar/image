

# Digitization — Image Processing Pipeline
 

## Problem Statement

Handwritten notes captured on smartphones often experience quality degradation due to noise, inconsistent lighting, and low contrast. These issues make the content difficult to read, share, or store digitally. This project presents an image processing pipeline designed to enhance such images by reducing noise, improving contrast, detecting text boundaries, restoring broken strokes, and separating text from the background — resulting in a clear and readable digital document.

## Pipeline Stages

| Stage | Technique                      | Purpose                                                    |
| ----- | ------------------------------ | ---------------------------------------------------------- |
| 1     | Gaussian & Median Filtering    | Reduce noise and camera grain                              |
| 2     | Histogram Equalization & CLAHE | Improve contrast and correct uneven lighting               |
| 3     | Sobel, Laplacian, Canny        | Identify text edges and stroke boundaries                  |
| 4     | Morphological Operations       | Restore broken strokes and eliminate small noise artifacts |
| 5     | Otsu & Adaptive Thresholding   | Separate text from the background                          |

## Output

* Step-by-step comparison images saved in the `output/` directory
* Final processed document stored as `output/final_clean_document.png`
* PSNR and SSIM metrics displayed for each stage
 


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
│   ├── noise.py          # Noise addition & removal
│   ├── enhancement.py    # Histogram EQ & CLAHE
│   ├── edges.py          # Sobel, Laplacian, Canny
│   ├── morphology.py     # Erode, Dilate, Open, Close
│   └── segmentation.py   # Otsu, Adaptive threshold
└── output/               # All results saved here
```
