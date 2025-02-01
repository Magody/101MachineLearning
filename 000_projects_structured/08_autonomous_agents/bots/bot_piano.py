from pyautogui import *
import pyautogui
import time
import keyboard
import random


def click(x, y):
    """Clicks with an random delay

    Args:
        x (float): _description_
        y (float): _description_
    """
    time.sleep(random.random()*0.1)
    pyautogui.click(x, y)


def isPixelRGBInRange(x: int, y: int, rgb_low: list, rgb_high: list) -> bool:
    """Check if pixel position is inside the RGB range

    Args:
        x (float): _description_
        y (float): _description_
        rgb_low (list): _description_
        rgb_high (list): _description_

    Returns:
        bool: _description_
    """
    pixel: list = pyautogui.pixel(x, y)

    is_pixel_in_range: bool = True
    for i in range(3):
        is_pixel_color_in_range: bool = pixel[i] >= rgb_low[i] and pixel[i] <= rgb_high[i]
        if not is_pixel_color_in_range:
            is_pixel_in_range = False
            break

    return is_pixel_in_range


class Tile:

    def __init__(self, pos_x, pos_y, width):
        """Init positions

        Args:
            pos_x (_type_): Position center for x
            pos_y (_type_): Position y reference
            width (_type_): Width of all tile
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width


y = 512
width_tile = 57
tiles = [
    Tile(331, y, width_tile),
    Tile(388, y, width_tile),
    Tile(445, y, width_tile),
    Tile(503, y, width_tile)
]

# _ = input("Press enter to begin: ")
print("Beginning bot, press Q to stop")
while not keyboard.is_pressed("q"):

    for tile in tiles:
        is_tile_pixel_black: bool = isPixelRGBInRange(
            tile.pos_x, tile.pos_y,
            [0, 0, 0], [2, 2, 2]
        )
        if is_tile_pixel_black:
            input(f"Click in ({tile.pos_x},{tile.pos_y})?")
            click(tile.pos_x, tile.pos_y)
            print("Click")
