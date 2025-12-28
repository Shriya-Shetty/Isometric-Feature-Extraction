## Project Methodology

This project follows a structured pipeline to automatically extract pipeline components and text labels from isometric pipeline drawings.

---

### Step 1: Data Understanding
- Studied isometric pipeline drawings and engineering symbols.
- Identified key pipeline elements such as pipes, welds, valves, elbows, tees, flanges, and supports.
- Understood how text labels (line numbers, sizes, tags) are placed near pipeline components.

---

### Step 2: Data Preparation
- Converted image-based PDF drawings into high-resolution PNG images (400–600 DPI).
- Ensured minimal quality loss during PDF to image conversion to preserve thin lines and small text.
- Organized images for annotation and model training.

---

### Step 3: Dataset Annotation
- Annotated pipeline components using polygon-based segmentation in Roboflow.
- Created labels for multiple classes including pipe segments, vertices, and fittings.
- Verified and cleaned annotations to ensure accuracy.

---

### Step 4: Model Training
- Trained YOLO-based instance segmentation models on the annotated dataset.
- Experimented with multiple models and training configurations to improve detection accuracy.
- Evaluated model performance based on segmentation quality and component visibility.

---

### Step 5: Pipeline and Component Detection
- Applied the trained YOLO model to detect pipeline components in unseen drawings.
- Extracted bounding boxes and segmentation masks for detected parts.
- Used detection results as spatial references for further processing.

---

### Step 6: Image Processing (OpenCV)
- Applied OpenCV techniques to detect thick pipeline lines and borders.
- Used line thickness analysis to distinguish main pipelines from auxiliary drawing elements.
- Improved structural understanding of the drawing.

---

### Step 7: Label Extraction (OCR)
- Cropped regions near detected pipeline components where labels are usually present.
- Applied image preprocessing (grayscale conversion and thresholding).
- Used OCR to extract text labels along with their coordinates.

---

### Step 8: Data Cleaning and Filtering
- Removed low-confidence OCR results.
- Filtered extracted text to keep only valid pipeline-style labels.
- Corrected common OCR errors in alphanumeric characters.

---

### Step 9: Structured Output Generation
- Stored extracted labels, coordinates, and component data in CSV format.
- Generated clear visual overlays with color-coded segmentation masks.
- Prepared outputs suitable for analysis, reporting, and demonstrations.

---

### Step 10: Evaluation and Refinement
- Compared outputs from different models and preprocessing settings.
- Identified resolution loss during PDF to PNG conversion and corrected it.
- Retrained models using improved data for better accuracy.

---

### Summary

This end-to-end workflow combines deep learning, traditional computer vision, and OCR to create a robust system for automated extraction of pipeline information from isometric engineering drawings.
