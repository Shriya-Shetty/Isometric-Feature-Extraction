from ultralytics import YOLO
import shutil
import os

# -------------------------------
# SETTINGS
# -------------------------------

DATA_YAML = r"Isometric Pipeline Segmentation.v2i.yolov8\data.yaml"       # your Roboflow dataset yaml
MODEL_NAME = "yolov8s-seg.pt"    # segmentation model
EPOCHS = 50
IMAGE_SIZE = 1024
OUTPUT_NAME = "best.pt"          # final output in this folder

# -------------------------------
# TRAIN
# -------------------------------

print("🔧 Loading YOLOv8 segmentation model...")
model = YOLO(MODEL_NAME)

print("🚀 Starting training...")
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=4,
    workers=2
)

print("\n🎉 Training Completed!")
best_weight_path = "runs2/segment/train/weights/best.pt"

# -------------------------------
# COPY best.pt TO CURRENT FOLDER
# -------------------------------

if os.path.exists(best_weight_path):
    shutil.copy(best_weight_path, OUTPUT_NAME)
    print(f"✅ Saved best.pt in current folder: {OUTPUT_NAME}")
else:
    print("❌ ERROR: best.pt not found! Check training folder structure.")
