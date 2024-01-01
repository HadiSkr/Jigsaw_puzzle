import cv2 as cv
import numpy as np
import helper



def compute_shape_similarity(edge1, edge2):
    # Get the minimum and maximum x, y coordinates of the edge points
    edge1_x = [point[0][0] for point in edge1]
    edge1_y = [point[0][1] for point in edge1]
    edge2_x = [point[0][0] for point in edge2]
    edge2_y = [point[0][1] for point in edge2]

    min_x = min(min(edge1_x), min(edge2_x))
    max_x = max(max(edge1_x), max(edge2_x))
    min_y = min(min(edge1_y), min(edge2_y))
    max_y = max(max(edge1_y), max(edge2_y))

    # Calculate the height and width
    height = max_y - min_y + 1
    width = max_x - min_x + 1

    # Convert the edge points to a binary image
    edge1_img = np.zeros((height, width), dtype=np.uint8)
    edge2_img = np.zeros((height, width), dtype=np.uint8)
    edge1_pts = np.array([[[point[0][0] - min_x, point[0][1] - min_y]] for point in edge1], dtype=np.int32)
    edge2_pts = np.array([[[point[0][0] - min_x, point[0][1] - min_y]] for point in edge2], dtype=np.int32)
    cv.drawContours(edge1_img, edge1_pts, -1, 255, -1)
    cv.drawContours(edge2_img, edge2_pts, -1, 255, -1)

    # Find the contour of the binary images
    contours1, _ = cv.findContours(edge1_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv.findContours(edge2_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Calculate the Hu moments of the contours
    edge1_moments = cv.moments(contours1[0])
    edge2_moments = cv.moments(contours2[0])

    # Compute the Hu moments descriptors
    edge1_hu_moments = cv.HuMoments(edge1_moments).flatten()
    edge2_hu_moments = cv.HuMoments(edge2_moments).flatten()

    # Calculate the shape similarity score using Hu moments
    shape_similarity_score = cv.compareHist(edge1_hu_moments, edge2_hu_moments, cv.HISTCMP_CORREL)

    return shape_similarity_score

def compute_cross_correlation(edge1, edge2):
    edge1_bgr = cv.cvtColor(edge1, cv.COLOR_RGB2BGR)
    edge2_bgr = cv.cvtColor(edge2, cv.COLOR_RGB2BGR)
    gray1 = cv.cvtColor(edge1_bgr, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(edge2_bgr, cv.COLOR_BGR2GRAY)

    normalized_gray1 = cv.normalize(gray1, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    normalized_gray2 = cv.normalize(gray2, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

    result = cv.matchTemplate(normalized_gray1, normalized_gray2, cv.TM_CCORR_NORMED)

    return result[0][0]


#contour_edges_colors = test2.contour_edges_colors
puzzle_pieces = test3.puzzle_pieces

correlation_scores = {}
opposite_edges = {'top_edge': 'bottom_edge', 'bottom_edge': 'top_edge', 'left_edge': 'right_edge', 'right_edge': 'left_edge'}

for index1, edges1 in puzzle_pieces.items():
    for edge1_key, edge1_colors in edges1.items():
        opposite_edge_key = opposite_edges[edge1_key]
        #edge1_contour = contour_edges[index1][edge1_key]  # Get the contour for edge1

        for index2, edges2 in puzzle_pieces.items():
            if index1 == index2:
                continue
            edge2_colors = edges2[opposite_edge_key]
           # edge2_contour = contour_edges[index2][opposite_edge_key]  # Get the contour for edge2

            correlation_score = compute_cross_correlation(edge1_colors, edge2_colors)

            # Combine the scores as per your preference (e.g., weighted sum)
            combined_score = correlation_score 

            correlation_scores[(index1, edge1_key, index2, opposite_edge_key)] = combined_score

sorted_scores = sorted(correlation_scores.items(), key=lambda x: x[1], reverse=True)

edge_connections = {}
top_edge_connections = {}

for (index1, edge1_key, index2, edge2_key), score in sorted_scores:
    # if len(edge_connections) == limit:
    #     top_edge_connections = copy.deepcopy(edge_connections)

    # if len(edge_connections) >= (n*n*4-n*4)/2:
    #     break

    if any(index1 == index and edge1_key == edge_key for index, edge_key in edge_connections.keys()) or \
       any(index1 == index and edge1_key == edge_key for index, edge_key in edge_connections.values()):
        continue
    if any(index2 == index and edge2_key == edge_key for index, edge_key in edge_connections.keys()) or \
       any(index2 == index and edge2_key == edge_key for index, edge_key in edge_connections.values()):
        continue
    edge_connections[(index1, edge1_key)] = (index2, edge2_key)

i = 0
for (index1, edge1_key), (index2, edge2_key) in edge_connections.items():
    score = correlation_scores[(index1, edge1_key, index2, edge2_key)]
    print(f"Edge {index1}, {edge1_key} is connected to Edge {index2}, {edge2_key} with a combined score of {score}")
    i += 1
print(i)