from ultralytics import YOLO
import cv2
import json

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = r"runs\segment\train2\weights\best.pt" 
SOURCE = "test.png"             # input image
OUTPUT_IMG = "output1.png"       # annotated output saved in base folder
OUTPUT_JSON = "direction_coords.json"

# -----------------------------
# DETECTION + COORD EXTRACTION
# -----------------------------
def run_detection():
    model = YOLO(MODEL_PATH)

    print("🔍 Running detection...")
    results = model.predict(source=SOURCE, save=False, imgsz=1024)

    r = results[0]

    # Draw prediction image
    annotated = r.plot()
    cv2.imwrite(OUTPUT_IMG, annotated)
    print(f"📂 Output image saved as: {OUTPUT_IMG}")

    # Get class names
    class_names = model.names

    direction_coords = []  # to store coordinates of DirectionChange instances

    # Check if masks exist
    if r.masks is None:
        print("⚠️ No masks detected.")
        return

    masks = r.masks.xy  # list of polygons

    for mask, cls_id in zip(masks, r.boxes.cls):
        cls_id = int(cls_id)
        cls_name = class_names[cls_id]

        if cls_name == "DirectionChange":
            # mask is N x 2 polygon array → convert to python list
            coords = mask.tolist()
            direction_coords.append(coords)

    # Save coordinates to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(direction_coords, f, indent=4)

    print(f"📌 Extracted {len(direction_coords)} DirectionChange polygons")
    print(f"📁 Coordinates saved to: {OUTPUT_JSON}")

    # Also print them
    for i, poly in enumerate(direction_coords):
        print(f"\n🔸 DirectionChange #{i+1} coordinates:")
        for point in poly:
            print(point)

    return direction_coords


if __name__ == "__main__":
    run_detection()
