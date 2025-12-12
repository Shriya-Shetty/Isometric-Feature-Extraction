import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to get black lines (thick and thin)
# Lower threshold value captures darker (thicker) lines
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# Method 1: Detect thick lines by measuring local line width
# Create a kernel to detect thick lines
thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Opening operation removes thin lines
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thick_kernel, iterations=1)

# Closing to connect nearby parts of thick lines
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=2)

# Find the thickest lines by measuring distance transform
dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Threshold distance transform to get only thick lines
# Higher value = thicker lines only
thickness_threshold = 2.5  # Adjust this: higher = only thickest lines
_, thick_lines_mask = cv2.threshold(dist_transform, thickness_threshold, 255, cv2.THRESH_BINARY)
thick_lines_mask = thick_lines_mask.astype(np.uint8)

# Dilate to restore full thickness
dilate_kernel = np.ones((7, 7), np.uint8)
thick_lines_mask = cv2.dilate(thick_lines_mask, dilate_kernel, iterations=2)

# Create highlighted versions
# Version 1: Dim background with bright pipeline
highlighted = img.copy()
dimmed = cv2.addWeighted(img, 0.3, img, 0, 0)
highlighted = dimmed.copy()
highlighted[thick_lines_mask > 0] = img[thick_lines_mask > 0]

# Version 2: Color the thick lines
colored_highlight = img.copy()
overlay = img.copy()
overlay[thick_lines_mask > 0] = [0, 255, 255]  # Cyan color
colored_highlight = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

# Version 3: Outline the thick lines
outline = img.copy()
contours, _ = cv2.findContours(thick_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(outline, contours, -1, (0, 255, 0), 2)

# Version 4: Show only the thick lines
pipeline_only = np.ones_like(img) * 255
pipeline_only[thick_lines_mask > 0] = img[thick_lines_mask > 0]

# Display results
plt.figure(figsize=(18, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Drawing')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(binary, cmap='gray')
plt.title('All Black Lines')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(dist_transform, cmap='jet')
plt.title('Line Thickness Map')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(thick_lines_mask, cmap='gray')
plt.title('Detected Thickest Lines')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
plt.title('Highlighted (Dimmed Background)')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(colored_highlight, cv2.COLOR_BGR2RGB))
plt.title('Colored Highlight (Cyan)')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(outline, cv2.COLOR_BGR2RGB))
plt.title('Green Outline')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(cv2.cvtColor(pipeline_only, cv2.COLOR_BGR2RGB))
plt.title('Thickest Lines Only')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save outputs
cv2.imwrite('thickest_lines_highlighted.png', highlighted)
cv2.imwrite('thickest_lines_colored.png', colored_highlight)
cv2.imwrite('thickest_lines_only.png', pipeline_only)

print(f"Detection complete!")
print(f"Adjust 'thickness_threshold' (currently {thickness_threshold}) to control sensitivity:")
print(f"  - Lower value (1.5-2.0): captures more lines")
print(f"  - Higher value (3.0-4.0): only the very thickest lines")