# Jigsaw Puzzle

This project is a jigsaw puzzle game that can be played in two levels: Level One and Level Two. The game includes a graphical user interface (GUI) and different modes of play.

## Level One

In Level One, you have the option to play with or without hint.

To play the game **level-1**, you first need to slice and shuffle the images using the following command:

```bash
python slicing_shuffle_images.py
```

### With Hint

To play the game with hints, use the following command:

```bash
python with_hint.py
```

### Without Hint

To play the game without hints, use the following command:

```bash
python GUI_without_hint.py
```

## Level Two

You can also choose to play with or without hints. Use the following commands to start the game:

### With Hints

```bash
python with_hint.py
```

### Without Hints

```bash
python without_hint.py
```

After each level, the images will be saved in the ‘images’ folder.

### Dependencies

Make sure you have the following Python libraries installed:

`pip install -r requirements.txt`

### Contributing

We welcome contributions to this project. Please feel free to submit a pull request or open an issue on GitHub.

### License

This project is licensed under the [MIT License](LICENSE).
