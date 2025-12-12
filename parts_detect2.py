from ultralytics import YOLO
from pathlib import Path

model = YOLO(r"parts_runs\train\seg_experiment\weights\best.pt")

# Process entire folder
image_folder = r"C:\Users\SHRIYA\Downloads\piplining-pages"
results = model.predict(
    source=image_folder,
    conf=0.09,
    save=True,
    save_txt=True,  # saves annotations in YOLO format
    project="predictions",
    name="batch_results"
)