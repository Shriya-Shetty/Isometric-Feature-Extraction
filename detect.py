from ultralytics import YOLO
import os

# ----------------------------------
# SETTINGS
# ----------------------------------

MODEL_PATH = r"runs\segment\train2\weights\best.pt"      # trained YOLO model
SOURCE = "test.png"         # image or folder
OUTPUT_NAME = "output.png"  # result saved in base folder

# ----------------------------------
# DETECTION
# ----------------------------------

def run_detection():
    print("🔍 Loading model...")
    model = YOLO(MODEL_PATH)

    print(f"📸 Running detection on: {SOURCE}")
    results = model.predict(
        source=SOURCE,
        save=False,       # do NOT save into runs/
        imgsz=1024
    )

    # ----------------------------------
    # SAVE RESULT IN BASE FOLDER
    # ----------------------------------
    result_image = results[0].plot()       # draw masks + boxes on image

    # Save manually
    import cv2
    cv2.imwrite(OUTPUT_NAME, result_image)

    print(f"\n✅ Detection Completed!")
    print(f"📂 Output saved in base folder as: {OUTPUT_NAME}")

    return results


if __name__ == "__main__":
    run_detection()
