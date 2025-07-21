import cv2
import numpy as np
import math

def edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 30, 150)  # Adjust the threshold values as per your requirements
    

    return edges

def categorize_shape(contour):
    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)

    # Approximate the contour to get the number of sides
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    # Get the number of sides
    sides = len(approx)
    # Categorize based on the number of sides
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        # Check if it's a square or rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif sides > 4:
        # Check if it's a circle or irregular shape based on the circularity
        area = cv2.contourArea(contour)
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        if circularity >= 0.85:
            return "Circle"
        else:
            return "Irregular"
    else:
        return "Irregular"

# Load the input image
img = cv2.imread('5 - Copy.jpeg')

height, width, channels = img.shape
aspect_ratio = height / width
new_height = int(600 * aspect_ratio)

image = cv2.resize(img, (600, new_height))

# Perform edge detection
edges = edge_detection(image)

# Find contours in the edge-detected image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area and hierarchy to get only shapes
filtered_contours = []
shape_categories = []
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 100 and hierarchy[0][i][3] == -1:
        filtered_contours.append(cnt)
        shape_category = categorize_shape(cnt)
        shape_categories.append(shape_category)

# Draw the filtered contours on a black canvas
canvas = np.zeros_like(image)
cv2.drawContours(canvas, filtered_contours, -1, (0, 255, 0), 2)

# Assign IDs to filtered contours and write IDs, area, and shape categories next to contours on the canvas image
for i, cnt in enumerate(filtered_contours):
    area = cv2.contourArea(cnt)
    x, y, _, _ = cv2.boundingRect(cnt)
    cv2.putText(canvas, f"ID: {i}", (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Area: {area}", (x + 20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Shape: {shape_categories[i]}", (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    print("ID: "+str(i))
    print("Area: "+str(area))
    print("Shape: "+str(shape_categories[i]))
    print()

# Write object count, smallest object ID, and largest object ID on the canvas image
object_count = len(filtered_contours)
smallest_object_id = min(range(object_count), key=lambda x: cv2.contourArea(filtered_contours[x]))
largest_object_id = max(range(object_count), key=lambda x: cv2.contourArea(filtered_contours[x]))
cv2.putText(canvas, f"Object Count: {object_count}", (10, new_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(canvas, f"Smallest Object ID: {smallest_object_id}", (10, new_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(canvas, f"Largest Object ID: {largest_object_id}", (10, new_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
print("Object Count: "+str(object_count))
print("Smallest Object ID: "+str(smallest_object_id))
print("Largest Object ID: "+str(largest_object_id))
print()
# Display the original image and the canvas image with the text annotations
cv2.imwrite("output_canvas.jpg", canvas)
cv2.imwrite("output_original.jpg", image)
print("Images saved. Check your folder.")