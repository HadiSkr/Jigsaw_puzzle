import os
import random
from PIL import Image, ImageDraw
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import Tk

def slice_image(image_path, grid_size):
    with Image.open(image_path) as image:
        width, height = image.size
        tile_width, tile_height = int(width/grid_size[0]), int(height/grid_size[1])
        tiles = []
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                box = (j*tile_width, i*tile_height, (j+1)*tile_width, (i+1)*tile_height)
                tile = image.crop(box)
                tiles.append(tile)
        random.shuffle(tiles)
        return tiles

def save_image(tiles, grid_size, output_path):
    width, height = tiles[0].size
    result = Image.new('RGB', (width*grid_size[0], height*grid_size[1]))
    draw = ImageDraw.Draw(result)
    
    # Paste tiles and draw grid lines
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            result.paste(tiles[i*grid_size[0]+j], (j*width, i*height))
            # draw.rectangle([(j*width, i*height), ((j+1)*width, (i+1)*height)], outline="black")
    
    result.save(output_path)

if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the main window
    image_path = filedialog.askopenfilename(title="Select the image")
    rows = simpledialog.askinteger("Input", "Enter the number of rows")
    cols = simpledialog.askinteger("Input", "Enter the number of columns")
    grid_size = (rows, cols)
    output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], title="Save the sliced image as")
    tiles = slice_image(image_path, grid_size)
    save_image(tiles, grid_size, output_path)
