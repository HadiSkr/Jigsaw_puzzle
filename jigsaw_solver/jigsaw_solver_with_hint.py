import cv2 as cv
import numpy as np
import os

def resize_image(image_path, scale_percent):
    # Read the image
    img = cv.imread(image_path)

    # Calculate the new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    return resized

def extract_puzzle_pieces(image_path, n, m):
    image = cv.imread(image_path)
    image = cv.resize(image, (image.shape[1] // 1, image.shape[0] // 1))
    height, width, _ = image.shape
    width = int(image.shape[1] * 100 / 100)
    height = int(image.shape[0] * 100 / 100)
    piece_height = height // n
    piece_width = width // m

    puzzle_pieces = {}
    for i in range(n):
        for j in range(m):
            piece = image[i * piece_height: (i + 1) * piece_height, j * piece_width: (j + 1) * piece_width]
            puzzle_pieces[(i, j)] = piece

    return puzzle_pieces


current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'images', '15_piece.png')
image_path2 = os.path.join(current_dir, 'images', 'lolo.jpg')


image = cv.imread(image_path)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)
contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
image_with_contours = image.copy()

min_contour_area = 100  # Adjust the minimum contour area as needed
filtered_contours = []
for contour in contours:
    area = cv.contourArea(contour)
    if area > min_contour_area:
        filtered_contours.append(contour)

final_pieces = extract_puzzle_pieces(image_path2, 3, 5)

aligned_image = np.zeros_like(image_with_contours)
for i, contour in enumerate(filtered_contours):
    mask = np.zeros_like(image_with_contours)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    cv.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv.FILLED)
    #cv.imshow(f"Contour {i}", mask)
    cv.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 3)  # Change the color and thickness as needed
    cv.waitKey(0)
    hist = cv.calcHist([image_with_contours], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist).flatten()

    best_match = None
    best_match_score = -1

    for (k, l), final_piece in final_pieces.items():
        final_hist = cv.calcHist([final_piece], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        final_hist = cv.normalize(final_hist, final_hist).flatten()

        score = cv.compareHist(hist, final_hist, cv.HISTCMP_INTERSECT)

        if score > best_match_score:
            best_match = (k, l)
            best_match_score = score

    print("Contour", i, "matches with piece", best_match)

    M = cv.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.putText(image_with_contours, str(i), (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# for (k, l), final_piece in final_pieces.items():
#     cv.imshow(f"Piece {k},{l}", final_piece)
#     cv.waitKey(0)

cv.imshow("Image with Contour Indices", image_with_contours)
cv.waitKey(0)
cv.destroyAllWindows()


