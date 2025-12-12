from ultralytics import YOLO
import shutil
import os

# -------------------------------
# SETTINGS
# -------------------------------

DATA_YAML = r"Isometric-piping.v12i.yolov8\data.yaml"
MODEL_NAME = "yolov8s-seg.pt"
EPOCHS = 50
IMAGE_SIZE = 1024
OUTPUT_NAME = "best.pt"  # saved in current folder

# Custom training folder
CUSTOM_RUNS = "runs2"     # <--- your custom output directory

# -------------------------------
# ENSURE runs2 EXISTS
# -------------------------------

if not os.path.exists(CUSTOM_RUNS):
    os.makedirs(CUSTOM_RUNS)
    print(f"📁 Created folder: {CUSTOM_RUNS}")

# -------------------------------
# TRAIN
# -------------------------------

print("🔧 Loading YOLOv8 segmentation model...")
model = YOLO(MODEL_NAME)

print("🚀 Starting training... (saving results inside runs2)")
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=4,
    workers=2,
    project=CUSTOM_RUNS,      # <--- forces output folder
    name="train",             # <--- final path: runs2/segment/train
    exist_ok=True
)

print("\n🎉 Training Completed!")

# -------------------------------
# PATH TO best.pt
# -------------------------------

best_weight_path = os.path.join(CUSTOM_RUNS, "train", "weights", "best.pt")

# -------------------------------
# COPY best.pt TO CURRENT FOLDER
# -------------------------------

if os.path.exists(best_weight_path):
    shutil.copy(best_weight_path, OUTPUT_NAME)
    print(f"✅ Saved best.pt in current folder as: {OUTPUT_NAME}")
else:
    print("❌ ERROR: best.pt not found! Check training folder structure:", best_weight_path)
