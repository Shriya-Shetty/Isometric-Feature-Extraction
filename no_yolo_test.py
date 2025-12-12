import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_pipeline_vertices(image_path):
    """
    Detect pipeline and highlight vertices using classical computer vision
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological operations to connect broken edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Find contours (pipeline boundaries)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output images
    result1 = img.copy()
    result2 = img.copy()
    
    # Method 1: Harris Corner Detection
    gray_float = np.float32(gray)
    harris_corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    
    # Threshold for corner detection
    result1[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
    
    # Method 2: Shi-Tomasi Corner Detection (Good Features to Track)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, 
                                      minDistance=10, blockSize=3)
    
    if corners is not None:
        corners = corners.astype(int)  # Fixed for newer NumPy versions
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result2, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(result2, (x, y), 8, (0, 0, 255), 2)
    
    # Draw contours on both results
    cv2.drawContours(result1, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(result2, contours, -1, (255, 0, 0), 2)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corners (Red)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
    plt.title('Shi-Tomasi Corners (Green)')
    plt.axis('off')
    
    # Hough Line Transform for line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=30, maxLineGap=10)
    
    result3 = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result3, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Mark line endpoints as potential vertices
            cv2.circle(result3, (x1, y1), 6, (0, 255, 0), -1)
            cv2.circle(result3, (x2, y2), 6, (0, 255, 0), -1)
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(result3, cv2.COLOR_BGR2RGB))
    plt.title('Hough Lines + Endpoints')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('pipeline_detection_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as: pipeline_detection_results.png")
    plt.show()
    
    # Save individual result images
    cv2.imwrite('output_harris_corners.jpg', result1)
    cv2.imwrite('output_shitomasi_corners.jpg', result2)
    cv2.imwrite('output_hough_lines.jpg', result3)
    cv2.imwrite('output_edges.jpg', edges)
    
    print("\nOutput images saved:")
    print("- output_harris_corners.jpg")
    print("- output_shitomasi_corners.jpg")
    print("- output_hough_lines.jpg")
    print("- output_edges.jpg")
    
    return result2

# Usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "page_4.png"
    result = detect_pipeline_vertices(image_path)
    
    print("Pipeline vertices detected and highlighted!")
    print("\nMethods used:")
    print("1. Harris Corner Detection (Red points)")
    print("2. Shi-Tomasi Corners (Green circles)")
    print("3. Hough Lines with endpoints (Blue lines + green dots)")
    print("\n✓ All output images saved in the current directory!")