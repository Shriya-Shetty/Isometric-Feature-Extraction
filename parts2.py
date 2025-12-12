from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import random

# ---------- CONFIG ----------
MODEL_PATH = "parts_runs/train/seg_experiment/weights/best.pt"
IMAGE_FOLDER = r"C:\Users\SHRIYA\Downloads\piplining-pages"
OUTPUT_DIR = "predictions_enhanced"
CONF_THRESHOLD = 0.08

# Visualization settings
MASK_ALPHA = 0.4          # Transparency of filled masks (0=transparent, 1=opaque)
CONTOUR_THICKNESS = 3     # Thickness of polygon outlines
LABEL_FONT_SCALE = 0.7    # Size of class labels
LABEL_THICKNESS = 2       # Thickness of label text
SHOW_CONFIDENCE = True    # Show confidence scores
SHOW_BBOX = False         # Show bounding boxes
DISTINCT_COLORS = True    # Use distinct colors per instance (vs per class)
# ----------------------------

def generate_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors

def get_text_size_adaptive(text, img_width):
    """Get adaptive font scale based on image width"""
    base_scale = img_width / 1000
    return max(0.4, min(1.0, base_scale))

# Load model
model = YOLO(MODEL_PATH)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Run inference
print("Running inference...")
results = model.predict(
    source=IMAGE_FOLDER,
    conf=CONF_THRESHOLD,
    save=False,  # We'll save custom visualizations
    save_txt=True,
    project=OUTPUT_DIR,
    name="annotations"
)

print(f"\nProcessing {len(results)} image(s)...\n")

# Process each image
for img_idx, result in enumerate(results):
    img_path = Path(result.path)
    print(f"[{img_idx+1}/{len(results)}] {img_path.name}")
    
    # Load original image
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Get detections
    masks = result.masks
    boxes = result.boxes
    
    if masks is None or len(masks) == 0:
        print("  ⚠ No detections found")
        cv2.imwrite(str(Path(OUTPUT_DIR) / img_path.name), img)
        continue
    
    class_names = result.names
    num_detections = len(masks)
    
    # Generate colors
    if DISTINCT_COLORS:
        colors = generate_colors(num_detections)
    else:
        # Colors per class
        num_classes = len(class_names)
        class_colors = generate_colors(num_classes)
        colors = [class_colors[int(boxes.cls[i])] for i in range(num_detections)]
    
    # Create overlay for transparency
    overlay = img.copy()
    
    # Draw all masks and annotations
    for i in range(num_detections):
        class_id = int(boxes.cls[i])
        class_name = class_names[class_id]
        confidence = float(boxes.conf[i])
        color = colors[i]
        
        # Get polygon points
        polygon = masks.xy[i].astype(np.int32)
        
        # Fill mask with transparency
        cv2.fillPoly(overlay, [polygon], color)
        
        # Draw polygon outline (thicker and brighter)
        cv2.polylines(img, [polygon], True, color, CONTOUR_THICKNESS)
        
        # Draw bounding box if enabled
        if SHOW_BBOX:
            bbox = boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Prepare label
        if SHOW_CONFIDENCE:
            label = f"{class_name} {confidence:.2f}"
        else:
            label = class_name
        
        # Get label position (top of polygon)
        label_y = int(polygon[:, 1].min())
        label_x = int(polygon[:, 0].min())
        
        # Adaptive font scale
        font_scale = get_text_size_adaptive(label, w) * LABEL_FONT_SCALE
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, LABEL_THICKNESS
        )
        
        # Draw label background
        cv2.rectangle(img, 
                     (label_x, label_y - text_h - baseline - 5),
                     (label_x + text_w + 5, label_y),
                     color, -1)
        
        # Draw label text
        cv2.putText(img, label,
                   (label_x + 2, label_y - baseline - 2),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale,
                   (255, 255, 255),  # White text
                   LABEL_THICKNESS,
                   cv2.LINE_AA)
    
    # Blend overlay with original image
    img = cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0)
    
    # Add summary text at top
    summary = f"Detected: {num_detections} object(s)"
    cv2.putText(img, summary, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, summary, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save enhanced visualization
    output_path = Path(OUTPUT_DIR) / f"{img_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), img)
    
    # Print detection summary
    class_counts = {}
    for i in range(num_detections):
        class_name = class_names[int(boxes.cls[i])]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"  ✓ Detected: {', '.join([f'{v} {k}' for k, v in class_counts.items()])}")
    print(f"  💾 Saved to: {output_path}")

print(f"\n{'='*60}")
print(f"✨ Processing complete!")
print(f"📁 Annotations (txt): {OUTPUT_DIR}/annotations/labels/")
print(f"🖼️  Visualizations: {OUTPUT_DIR}/")
print(f"{'='*60}")

# Optional: Create a side-by-side comparison
print("\n📊 Creating comparison images...")
for img_idx, result in enumerate(results):
    img_path = Path(result.path)
    original = cv2.imread(str(img_path))
    annotated_path = Path(OUTPUT_DIR) / f"{img_path.stem}_annotated.jpg"
    
    if annotated_path.exists():
        annotated = cv2.imread(str(annotated_path))
        
        # Resize if needed to same height
        h1, h2 = original.shape[0], annotated.shape[0]
        if h1 != h2:
            scale = h1 / h2
            annotated = cv2.resize(annotated, None, fx=scale, fy=scale)
        
        # Concatenate side by side
        comparison = np.hstack([original, annotated])
        
        # Add labels
        cv2.putText(comparison, "Original", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(comparison, "Segmented", (original.shape[1] + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        comparison_path = Path(OUTPUT_DIR) / f"{img_path.stem}_comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison)

print("✅ Done! Check the output folder.")