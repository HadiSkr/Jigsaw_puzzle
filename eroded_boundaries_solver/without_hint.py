import cv2 as cv
import numpy as np
import check_puzzle
import copy
def without_hint(path):
    def get_updated_value(key, value, **kwargs):
        if key == ord('+'):
            return value, 0, 0
        elif key == ord('-'):
            return -value, 0, 0
        elif key == ord('*'):
            return 0, kwargs.get('value2', value), 0
        elif key == ord('/'):
            return 0, -kwargs.get('value2', value), 0
        elif key == ord('8'):
            return 0, 0, kwargs.get('value3', value)
        elif key == ord('2'):
            return 0, 0, -kwargs.get('value3', value)
        elif key == 27:
            raise Exception('Terminated by user!')
        else:
            return 0, 0, 0


    n = 4
    def error_handler(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                cv.destroyAllWindows()
                print(f'{ex}')

        return wrapper

    limit = (n*n*4-n*4)/2
    fix = 1


    def check (pieces,edge_connections,top_edge_connections,n, fix):
        result = check_puzzle.check_wrong_pieces(pieces, top_edge_connections, n)
        if result[2] == 0:
            print (result[0])
        else:
            id1 = result[0]
            id2 = result[1]
            for key, value in list(top_edge_connections.items()):
                index1, edge1_key = key
                index2, edge2_key = value
                if (index1 == id1 and index2 == id2) or (index1 == id2 and index2 == id1):
                    del top_edge_connections[key]
                    index_to_retrieve = limit + fix 
                    records = list(edge_connections.items())
                    specific_record = records[int(index_to_retrieve - 1)]
                    top_edge_connections[specific_record[0]] = specific_record[1]
                    fix = fix + 1
                    break
            check (pieces,edge_connections,top_edge_connections,n, fix)
        



    @error_handler
    def extract_puzzle_pieces(image_path):
        image = cv.imread(image_path)
        image = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
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

    def compute_cross_correlation(edge1, edge2):
        if edge1.ndim == 3 and edge1.shape[2] == 3:  # Check if edge1 is a 3-channel image
            edge1_gray = cv.cvtColor(edge1, cv.COLOR_BGR2GRAY)
        else:
            edge1_gray = edge1


        if edge2.ndim == 3 and edge2.shape[2] == 3:  # Check if edge2 is a 3-channel image
            edge2_gray = cv.cvtColor(edge2, cv.COLOR_BGR2GRAY)
        else:
            edge2_gray = edge2

        cross_corr = cv.matchTemplate(edge1_gray, edge2_gray, cv.TM_CCORR_NORMED)
        # cross_corr = cv.matchTemplate(edge1_gray, edge2_gray, cv.TM_CCOEFF_NORMED)
        
        return cross_corr[0, 0]

    pieces = extract_puzzle_pieces(path)

    # Load the image
    image = cv.imread(path)

    # Get the image height and width
    height, width = image.shape[:2]

    # Define the piece height and width

    # Display each puzzle piece and retrieve color information of edges
    for index, piece in pieces.items():
        i, j = index
        print(f"Color info for Puzzle Piece ({i+1}, {j+1}) edges:")
        print("Top Edge:")
        print(piece['top_edge'])
        print("Bottom Edge:")
        print(piece['bottom_edge'])
        print("Left Edge:")
        print(piece['left_edge'])
        print("Right Edge:")
        print(piece['right_edge'])
        print()
        cv.waitKey(0)
        cv.destroyAllWindows()


    color_histograms = {}
    for index, piece in pieces.items():
        image = piece['image']
        hist = cv.calcHist([image], [0, 1, 2], None, [7, 7, 7], [0, 256, 0, 256, 0, 256])
        hist = cv.normalize(hist, hist).flatten()
        color_histograms[index] = hist



    correlation_scores = {}
    opposite_edges = {'top_edge': 'bottom_edge', 'bottom_edge': 'top_edge', 'left_edge': 'right_edge', 'right_edge': 'left_edge'}

    for index1, piece1 in pieces.items():
        for edge1_key, edge1 in piece1.items():
            if edge1_key == 'image':
                continue
            opposite_edge_key = opposite_edges[edge1_key]
            for index2, piece2 in pieces.items():
                if index1 == index2:
                    continue
                edge2 = piece2[opposite_edge_key]
                score = compute_cross_correlation(edge1, edge2)
                hist_score = cv.compareHist(color_histograms[index1], color_histograms[index2], cv.HISTCMP_BHATTACHARYYA)
                correlation_scores[(index1, edge1_key, index2, opposite_edge_key)] = (score, hist_score)


    sorted_scores = sorted(correlation_scores.items(), key=lambda x: x[1][0] * 1 + x[1][1] * 0.0, reverse=True)

    edge_connections = {}
    top_edge_connections = {}

    for (index1, edge1_key, index2, edge2_key), score in sorted_scores:
        if len(edge_connections) == limit:
            top_edge_connections = copy.deepcopy(edge_connections)
        if any(index1 == index and edge1_key == edge_key for index, edge_key in edge_connections.keys()) or \
        any(index1 == index and edge1_key == edge_key for index, edge_key in edge_connections.values()):
            continue
        if any(index2 == index and edge2_key == edge_key for index, edge_key in edge_connections.keys()) or \
        any(index2 == index and edge2_key == edge_key for index, edge_key in edge_connections.values()):
            continue
        edge_connections[(index1, edge1_key)] = (index2, edge2_key)

    print (len(top_edge_connections))

    check(pieces,edge_connections,top_edge_connections,n, fix)

    for (index1, edge1_key), (index2, edge2_key) in top_edge_connections.items():
        score = correlation_scores[(index1, edge1_key, index2, edge2_key)]
        print(f"Edge {index1}, {edge1_key} is connected to Edge {index2}, {edge2_key} with a correlation score of {score}")

    top_left_piece = None
    for piece in pieces.keys():
        if not any(edge == 'left_edge' or edge == 'top_edge' for _, edge in top_edge_connections.keys() if _ == piece) and \
        not any(edge == 'left_edge' or edge == 'top_edge' for _, edge in top_edge_connections.values() if _ == piece):
            top_left_piece = piece
            break

    print(f"The top-left piece is {top_left_piece}")

    bottom_right_piece = None


    cuurent_piece = top_left_piece
    grid = [[None] * n for _ in range(n)]
    for i in range(n):
            for j in range(n):
                if j==0:
                    grid[i][j] = cuurent_piece
                else:
                    for (index1, edge1_key), (index2, edge2_key) in top_edge_connections.items():
                        if index1 == cuurent_piece and edge1_key == 'right_edge':
                            grid[i][j] = index2
                            cuurent_piece = index2
                            break
                        if index2 == cuurent_piece and edge2_key == 'right_edge':
                            grid[i][j] = index1
                            cuurent_piece = index1
                            break
            if i is not (n-1):
                for (index1, edge1_key), (index2, edge2_key) in top_edge_connections.items():
                        if index1 == grid[i][0] and edge1_key == 'bottom_edge':
                            cuurent_piece = index2
                            break
                        if index2 == grid[i][0] and edge2_key == 'bottom_edge':
                            cuurent_piece = index1
                            break
    # Print the grid
    for row in grid:
        print(row)
        
    for i in range(n):
        for j in range(n):
            if grid[i][j] == None:
                grid[i][j] = (i, j)



    piece_height = height // n
    piece_width = width // n

    # Resize the canvas to match the size of the assembled image
    canvas_height = piece_height * n
    canvas_width = piece_width * n
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    resized_pieces = {}
    for index, piece in pieces.items():
        resized_piece = cv.resize(piece['image'], (piece_width, piece_height))
        resized_pieces[index] = {
            'image': resized_piece,
            'top_edge': resized_piece[0, :, :],
            'bottom_edge': resized_piece[-1, :, :],
            'left_edge': resized_piece[:, 0, :],
            'right_edge': resized_piece[:, -1, :]
        }


    for i in range(n):
        for j in range(n):
            piece_index = grid[i][j]
            piece = resized_pieces[piece_index]['image']
            x = j * piece_width
            y = i * piece_height
            canvas[y:y+piece_height, x:x+piece_width] = piece
        
    # Display the assembled image
    output_image_path = 'output.jpg'
    cv.imwrite(output_image_path, canvas)
