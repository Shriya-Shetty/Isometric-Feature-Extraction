from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
MODEL_PATH = r"parts_runs\train\seg_experiment\weights\best.pt"
INPUT_IMAGE = "test.png"
OUTPUT_DIR = "predictions"
CONF_THRESHOLD = 0.1  # confidence threshold
# ----------------------------

# Load trained model
model = YOLO(MODEL_PATH)

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Run inference
results = model.predict(
    source=INPUT_IMAGE,
    conf=CONF_THRESHOLD,
    save=True,  # saves annotated image
    project=OUTPUT_DIR,
    name="results",
    exist_ok=True
)

# Process results for each image (in case of batch)
for idx, result in enumerate(results):
    print(f"\n{'='*50}")
    print(f"Image: {result.path}")
    print(f"{'='*50}")
    
    # Get all detections
    boxes = result.boxes  # bounding boxes
    masks = result.masks  # segmentation masks
    
    if masks is None:
        print("No objects detected!")
        continue
    
    # Get class names
    class_names = result.names
    
    # Iterate through each detection
    for i in range(len(masks)):
        # Class info
        class_id = int(boxes.cls[i])
        class_name = class_names[class_id]
        confidence = float(boxes.conf[i])
        
        # Bounding box
        bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
        
        # Segmentation mask (polygon points)
        mask = masks.xy[i]  # Nx2 array of polygon vertices
        
        print(f"\nDetection {i+1}:")
        print(f"  Class: {class_name} (ID: {class_id})")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  BBox: {bbox}")
        print(f"  Polygon points: {len(mask)} vertices")
        print(f"  First 3 points: {mask[:3]}")
    
    # ============================================
    # OPTION 1: Save annotations to text file (YOLO format)
    # ============================================
    txt_path = Path(OUTPUT_DIR) / "results" / f"{Path(result.path).stem}.txt"
    with open(txt_path, 'w') as f:
        for i in range(len(masks)):
            class_id = int(boxes.cls[i])
            mask = masks.xy[i]
            
            # Normalize coordinates to [0, 1]
            h, w = result.orig_shape
            normalized_mask = mask / [w, h]
            
            # Write: class_id x1 y1 x2 y2 x3 y3 ...
            line = f"{class_id}"
            for point in normalized_mask:
                line += f" {point[0]:.6f} {point[1]:.6f}"
            f.write(line + "\n")
    
    print(f"\nAnnotations saved to: {txt_path}")
    
    # ============================================
    # OPTION 2: Get results per class
    # ============================================
    print("\n--- Results by Class ---")
    class_results = {}
    
    for i in range(len(masks)):
        class_id = int(boxes.cls[i])
        class_name = class_names[class_id]
        
        if class_name not in class_results:
            class_results[class_name] = []
        
        class_results[class_name].append({
            'confidence': float(boxes.conf[i]),
            'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
            'polygon': masks.xy[i].tolist(),
            'mask_binary': masks.data[i].cpu().numpy()  # binary mask (HxW)
        })
    
    # Print summary by class
    for class_name, detections in class_results.items():
        print(f"\n{class_name}: {len(detections)} instance(s)")
        for j, det in enumerate(detections):
            print(f"  Instance {j+1}: conf={det['confidence']:.3f}, "
                  f"polygon_points={len(det['polygon'])}")
    
    # ============================================
    # OPTION 3: Visualize with custom colors
    # ============================================
    img = cv2.imread(result.path)
    
    # Define colors for each class (BGR format)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]
    
    for i in range(len(masks)):
        class_id = int(boxes.cls[i])
        class_name = class_names[class_id]
        confidence = float(boxes.conf[i])
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw polygon
        mask = masks.xy[i].astype(np.int32)
        cv2.polylines(img, [mask], True, color, 2)
        
        # Fill mask with transparency
        overlay = img.copy()
        cv2.fillPoly(overlay, [mask], color)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Add label
        bbox = boxes.xyxy[i].cpu().numpy().astype(int)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img, label, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save custom visualization
    custom_output = Path(OUTPUT_DIR) / "results" / f"{Path(result.path).stem}_custom.jpg"
    cv2.imwrite(str(custom_output), img)
    print(f"\nCustom visualization saved to: {custom_output}")

print(f"\n{'='*50}")
print(f"All results saved to: {OUTPUT_DIR}/results/")
print(f"{'='*50}")