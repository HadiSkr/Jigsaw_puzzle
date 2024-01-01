import cv2
import numpy as np
from scipy import ndimage

def calculate_edge_intensity(edge):
    return np.mean(edge)

def calculate_edge_sharpness(edge):
    return np.std(edge)

def calculate_edge_profile(edge):
    return edge

def calculate_edge_continuity(edge):
    return ndimage.measurements.label(edge)[1]

# def calculate_edge_junctions(edge):
#     # Convert the edge to grayscale if it is a color image
#     if len(edge.shape) == 3:
#         edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)

#     # Convert the edge to 32-bit float if it is not already
#     if edge.dtype != np.float32:
#         edge = np.float32(edge)

#     return cv2.cornerHarris(edge, 2, 3, 0.04)


def extract_puzzle_pieces(image_path, n):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    height, width, _ = image.shape

    piece_height = height // n
    piece_width = width // n

    puzzle_pieces = {}
    for i in range(n):
        for j in range(n):
            piece = image[i * piece_height: (i + 1) * piece_height, j * piece_width: (j + 1) * piece_width]
            puzzle_pieces[(i, j)] = {
                'image': piece,
                'top_edge': piece[0, :, :],
                'bottom_edge': piece[-1, :, :],
                'left_edge': piece[:, 0, :],
                'right_edge': piece[:, -1, :]
            }

    return puzzle_pieces

# Read the input image
path = "C:\\Users\\Hadi\\Desktop\\Computer Vision\\cv_project\\new.jpg"
image = cv2.imread(path)

# Set the number of puzzle pieces
n = 4

# Extract puzzle pieces
puzzle_pieces = extract_puzzle_pieces(path, n)

# Set the low and high thresholds for Canny edge detection
low_threshold = 40
high_threshold = 95

for piece_coords, piece_info in puzzle_pieces.items():
    piece_image = piece_info['image']

    # Apply Canny edge detection
    edges = cv2.Canny(piece_image, low_threshold, high_threshold)

    # Find contours in the binary image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get image dimensions
    height, width = edges.shape[:2]

    # Filter contours that touch the outer boundaries of the image
    outer_edges = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                outer_edges.append(contour)
                break

    # Draw the outer edges on a black image
    outer_edges_image = np.zeros_like(edges)
    cv2.drawContours(outer_edges_image, outer_edges, -1, 255, thickness=cv2.FILLED)

    # Calculate features for each outer edge
    for edge in outer_edges:
        # Calculate orientation
        [vx, vy, x, y] = cv2.fitLine(edge, cv2.DIST_L2, 0, 0.01, 0.01)
        orientation = np.arctan2(vy, vx)  # Orientation in radians

        # Calculate thickness
        dist_transform = cv2.distanceTransform(cv2.bitwise_not(outer_edges_image), cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        thickness = max_val * 2  # Diameter of the largest circle that fits within the edge

        # Calculate curvature
        epsilon = 0.01 * cv2.arcLength(edge, True)
        approx = cv2.approxPolyDP(edge, epsilon, True)
        curvature = len(approx)  # The more vertices in the approximation, the higher the curvature

        # Calculate edge intensity
        edge_intensity = calculate_edge_intensity(edge)

        # Calculate edge sharpness
        edge_sharpness = calculate_edge_sharpness(edge)

        # Calculate edge profile
        edge_profile = calculate_edge_profile(edge)

        # Calculate edge continuity
        edge_continuity = calculate_edge_continuity(edge)

        edge_features = {
            'orientation': orientation,
            'thickness': thickness,
            'curvature': curvature,
            'edge_intensity': edge_intensity,
            'edge_sharpness': edge_sharpness,
            'edge_profile': edge_profile,
            'edge_continuity': edge_continuity
        }

        # Add the features to the piece info
        piece_info['edge_features'] = edge_features

        # Calculate edge junctions
    #edge_junctions = calculate_edge_junctions(edge)

        print(f'Edge Orientation: {orientation}, Thickness: {thickness}, Curvature: {curvature}, Edge Intensity: {edge_intensity}, Edge Sharpness: {edge_sharpness}, Edge Profile: {edge_profile}, Edge Continuity: {edge_continuity}')

    # Display the original piece image and the outer edges
    cv2.imshow(f'Piece {piece_coords}', piece_image)
    cv2.imshow(f'Piece {piece_coords} Outer Edges', outer_edges_image)
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()