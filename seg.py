from ultralytics import YOLO
import cv2
import numpy as np

# ----------------------------------
# SETTINGS
# ----------------------------------

MODEL_PATH = r"runs\segment\train2\weights\best.pt"
SOURCE = "page_4.png"
OUTPUT_NAME = "output_seg1.png"

# Visualization settings
CONF_THRESHOLD = 0.25  # confidence threshold
LINE_WIDTH = 2         # bounding box line width
FONT_SCALE = 0.6       # label font size
MASK_ALPHA = 0.4       # mask transparency (0=transparent, 1=opaque)

# ----------------------------------
# DETECTION WITH PROPER SEGMENTATION
# ----------------------------------

def run_detection():
    print("🔍 Loading model...")
    model = YOLO(MODEL_PATH)

    print(f"📸 Running segmentation on: {SOURCE}")
    results = model.predict(
        source=SOURCE,
        save=False,
        imgsz=1024,
        conf=CONF_THRESHOLD,
        iou=0.45,
        max_det=300,        # allow more detections
        retina_masks=True,  # higher quality masks
        verbose=False
    )

    # ----------------------------------
    # CUSTOM VISUALIZATION
    # ----------------------------------
    result = results[0]
    
    # Get original image
    img = cv2.imread(SOURCE)
    if img is None:
        print(f"❌ Error: Could not read image from {SOURCE}")
        return None
    
    # Create overlay for masks
    overlay = img.copy()
    
    # Draw segmentation masks and boxes
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # segmentation masks
        boxes = result.boxes.data.cpu().numpy()  # bounding boxes
        
        print(f"\n📊 Found {len(masks)} objects:")
        
        # Define colors for each class (BGR format)
        colors = {
            0: (255, 0, 0),      # Blue
            1: (0, 255, 0),      # Green
            2: (0, 0, 255),      # Red
            3: (255, 255, 0),    # Cyan
            4: (255, 0, 255),    # Magenta
            5: (0, 255, 255),    # Yellow
        }
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Get class and confidence
            cls = int(box[5])
            conf = float(box[4])
            
            # Get class name
            class_name = model.names[cls]
            
            print(f"  - {class_name}: {conf:.2f}")
            
            # Resize mask to image size
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_bool = mask_resized > 0.5
            
            # Get color for this class
            color = colors.get(cls, (0, 255, 0))
            
            # Apply colored mask
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, LINE_WIDTH)
            
            # Draw label with background
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2
            )
            
            # Label background
            cv2.rectangle(
                overlay, 
                (x1, y1 - label_h - baseline - 5), 
                (x1 + label_w + 5, y1), 
                color, 
                -1
            )
            
            # Label text
            cv2.putText(
                overlay, 
                label, 
                (x1 + 2, y1 - baseline - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                FONT_SCALE, 
                (255, 255, 255), 
                2
            )
        
        # Blend overlay with original image
        result_image = cv2.addWeighted(img, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
    else:
        print("\n⚠️ No segmentation masks detected!")
        result_image = img

    # Save result
    cv2.imwrite(OUTPUT_NAME, result_image)

    print(f"\n✅ Segmentation Completed!")
    print(f"📂 Output saved as: {OUTPUT_NAME}")
    print(f"📏 Image size: {img.shape[1]}x{img.shape[0]}")

    return results


# ----------------------------------
# ALTERNATIVE: USE ULTRALYTICS PLOT
# ----------------------------------

def run_detection_simple():
    """Simpler version using built-in plot"""
    print("🔍 Loading model...")
    model = YOLO(MODEL_PATH)

    print(f"📸 Running segmentation on: {SOURCE}")
    results = model.predict(
        source=SOURCE,
        save=False,
        imgsz=1024,
        conf=CONF_THRESHOLD,
        retina_masks=True,
        verbose=True
    )

    # Use built-in plotting with better settings
    result_image = results[0].plot(
        conf=True,           # show confidence
        line_width=2,        # box line width
        font_size=12,        # label font size
        pil=False,           # return as numpy array
        img=None,            # use original image
        labels=True,         # show labels
        boxes=True,          # show boxes
        masks=True,          # show masks
        probs=False          # don't show probabilities
    )

    cv2.imwrite(OUTPUT_NAME, result_image)
    
    print(f"\n✅ Detection Completed!")
    print(f"📂 Output saved as: {OUTPUT_NAME}")

    return results


if __name__ == "__main__":
    # Choose one:
    run_detection()          # Custom visualization with better control
    # run_detection_simple() # Simpler version using built-in plot