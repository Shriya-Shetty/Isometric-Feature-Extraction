from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained model
model = YOLO(r"runs\segment\train2\weights\best.pt")

# Run inference
results = model("test.png", save=False)

img = cv2.imread("test.png")

for r in results:
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()

        for mask in masks:
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            # Color the pipeline
            overlay = img.copy()
            overlay[mask > 0] = (0, 0, 255)  # red

            img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

cv2.imwrite("pipeline_colored.png", img)
