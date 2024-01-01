import cv2 as cv
import numpy as np

original_path = 'C:\\Users\\Hadi\\Desktop\\Computer Vision\\cv_project\\18_piece.png'

# Load the image
image = cv.imread(original_path)

# Convert the image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply adaptive thresholding to obtain a binary image
_, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)

# Find contours in the binary image
contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image and extract color information
image_with_contours = image.copy()

contour_colors = []
total_segments = 0
puzzle_pieces = {}
for i, contour in enumerate(contours):

    cv.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)

    contour_points = contour[:, 0, :]
    # Draw the approximated contour

    # Extract contour points
    contour_points = contour[:, 0, :]

    # Sample color from the original image at each contour point
    #contour_colors.append([image[y, x] for x, y in contour_points])
        
    total_segments += 1

    top_left = min(contour_points, key=lambda point: point[0] + point[1])
    top_right = max(contour_points, key=lambda point: point[0] - point[1])
    bottom_left = min(contour_points, key=lambda point: point[0] - point[1])
    bottom_right = max(contour_points, key=lambda point: point[0] + point[1])
    sorted_points = sorted(contour_points, key=lambda point: point[0])
    cv.circle(image_with_contours, tuple(top_left), 5, (0, 0, 255), -1)
    cv.circle(image_with_contours, tuple(top_right), 5, (0, 0, 255), -1)
    cv.circle(image_with_contours, tuple(bottom_left), 5, (0, 0, 255), -1)
    cv.circle(image_with_contours, tuple(bottom_right), 5, (0, 0, 255), -1)

    top_edge = np.array([image[y, x] for x, y in np.linspace(top_left+(0,3), top_right+(0,3), num=100, dtype=int)])
    bottom_edge = np.array([image[y, x] for x, y in np.linspace(bottom_left-(0,3), bottom_right-(0,3), num=100, dtype=int)])
    left_edge = np.array([image[y, x] for x, y in np.linspace(top_left+(3,0), bottom_left+(3,0), num=100, dtype=int)])
    right_edge = np.array([image[y, x] for x, y in np.linspace(top_right-(3,0), bottom_right-(3,0), num=100, dtype=int)])
    
    puzzle_pieces[i+1] = {
        'top_edge': top_edge,
        'bottom_edge': bottom_edge,
        'left_edge': left_edge,
        'right_edge': right_edge
    }
    moments = cv.moments(contour)
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        
        cv.circle(image_with_contours, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
    
       
    contour_number = str(i + 1)
    cv.putText(image_with_contours, contour_number, (centroid_x, centroid_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    


print("Total contour segments:", total_segments)

# Display the original image with contours and corner dots
cv.imshow("Original Image with Contours and Corners", image_with_contours)
cv.waitKey(0)
cv.destroyAllWindows()

# Print the extracted contour colors
for i, colors in enumerate(contour_colors):
    print(f"Contour {i + 1} colors:")
    for color in colors:
        print(color)