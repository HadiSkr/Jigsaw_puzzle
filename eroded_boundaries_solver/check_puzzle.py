def check_wrong_pieces(pieces, edge_connections, n):
    bottom_right_piece = None
    bottom_left_piece = None


    for piece in pieces.keys():
        if not any(edge == 'right_edge' or edge == 'bottom_edge' for _, edge in edge_connections.keys() if _ == piece) and \
        not any(edge == 'right_edge' or edge == 'bottom_edge' for _, edge in edge_connections.values() if _ == piece):
            bottom_right_piece = piece
            break
    print(f"The bottom-right piece is {bottom_right_piece}")


    for piece in pieces.keys():
        if not any(edge == 'left_edge' or edge == 'bottom_edge' for _, edge in edge_connections.keys() if _ == piece) and \
        not any(edge == 'left_edge' or edge == 'bottom_edge' for _, edge in edge_connections.values() if _ == piece):
            bottom_left_piece = piece
            break
    print(f"The bottom-left piece is {bottom_left_piece}")


    current_piece = bottom_right_piece
    for i in range(n):
            for (index1, edge1_key), (index2, edge2_key) in edge_connections.items():
                 if (current_piece == index1 and edge1_key == 'right_edge') or (current_piece == index2 and edge2_key == 'right_edge'):
                      return [index1, index2, -1]
            if i is not (n-1):
                for (index1, edge1_key), (index2, edge2_key) in edge_connections.items():
                    if (current_piece == index1 and edge1_key == 'top_edge'):
                        current_piece = index2
                    if (current_piece == index2 and edge2_key == 'top_edge'):
                        current_piece = index1


    current_piece = bottom_left_piece
    for i in range(n):
            for (index1, edge1_key), (index2, edge2_key) in edge_connections.items():
                 if (current_piece == index1 and edge1_key == 'left_edge') or (current_piece == index2 and edge2_key == 'left_edge'):
                      return [index1, index2, -1]
            if i is not (n-1):
                for (index1, edge1_key), (index2, edge2_key) in edge_connections.items():
                    if (current_piece == index1 and edge1_key == 'top_edge'):
                        current_piece = index2
                    if (current_piece == index2 and edge2_key == 'top_edge'):
                        current_piece = index1
                        
    return ["no error occured","Great",0]

