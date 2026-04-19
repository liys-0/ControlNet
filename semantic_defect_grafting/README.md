# Semantic Defect Grafting Pipeline (Method 1)

This directory contains a pure OpenCV implementation of Method 1: Semantic Defect Grafting. It extracts defects from real PFIB images and grafts them onto synthetic Avalon CAD images.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open `pipeline.py` and set the input/output directories:
   - `PFIB_INPUT_DIR`: Path to your real PFIB images.
   - `AVALON_INPUT_DIR`: Path to your synthetic Avalon images (must have matching filenames).
   - `OUTPUT_DIR`: Path where the composited images will be saved.

3. Run the pipeline:
   ```bash
   python pipeline.py
   ```

## Pipeline Steps
1. **Alignment**: Aligns the PFIB image to the Avalon image using `cv2.findTransformECC` or `cv2.phaseCorrelate`.
2. **Defect Extraction**: Extracts defects using absolute difference and adaptive thresholding.
3. **CAD-ification**: Simplifies the defect shapes into CAD-like polygons using `cv2.findContours` and `cv2.approxPolyDP`.
4. **Compositing**: Overlays the CAD-ified defects onto the Avalon image using `cv2.bitwise_and` / `cv2.bitwise_or`.
