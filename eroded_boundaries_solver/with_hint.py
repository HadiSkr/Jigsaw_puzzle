import os
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import Tk

def extract_features(image):
    # Use SIFT 
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def find_tile_in_shuffled_image(original_tile, shuffled_image, tile_width, tile_height):
    height, width, _ = shuffled_image.shape
    min_mse = float('inf')
    best_tile = None
    best_idx = None

    # Convert the images to grayscale
    original_tile_gray = cv2.cvtColor(original_tile, cv2.COLOR_BGR2GRAY)

    for i in range(0, height - tile_height + 1, tile_height):
        for j in range(0, width - tile_width + 1, tile_width):
            shuffled_tile = shuffled_image[i:i+tile_height, j:j+tile_width]
            shuffled_tile_gray = cv2.cvtColor(shuffled_tile, cv2.COLOR_BGR2GRAY)

            mse = mean_squared_error(original_tile_gray.flatten(), shuffled_tile_gray.flatten())

            if mse < min_mse:
                min_mse = mse
                best_tile = shuffled_tile
                best_idx = (i, j)

    return best_tile, best_idx

def solve_puzzle(shuffled_image_path, original_image_path, output_path, grid_size):
    shuffled_image = cv2.imread(shuffled_image_path, cv2.IMREAD_COLOR)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)

    # Get the size of the tiles
    tile_width, tile_height = shuffled_image.shape[1] // grid_size[0], shuffled_image.shape[0] // grid_size[1]

    # Create an empty array for the solved image
    solved_image = np.zeros_like(shuffled_image)

    # Iterate over each tile in the original image
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            # This is the tile in the original image
            original_tile = original_image[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width]

            shuffled_tile, shuffled_idx = find_tile_in_shuffled_image(original_tile, shuffled_image, tile_width, tile_height)

            solved_image[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width] = shuffled_tile

    # Save the solved puzzle
    cv2.imwrite(output_path, solved_image)

if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the main window
    shuffled_image_path = filedialog.askopenfilename(title="Select the shuffled image")
    original_image_path = filedialog.askopenfilename(title="Select the original image")
    output_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")], title="Save the solved puzzle as")
    rows = simpledialog.askinteger("Input", "Enter the number of rows")
    cols = simpledialog.askinteger("Input", "Enter the number of columns")
    grid_size = (rows, cols)

    solve_puzzle(shuffled_image_path, original_image_path, output_path, grid_size)
