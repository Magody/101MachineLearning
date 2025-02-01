from .Position import Position
import random


class Pipe:

    GAP = 150
    VELOCITY = 7

    def __init__(self, initial_position_x, image_top, image_bottom):
        self.position = Position(initial_position_x, 0)
        self.height = random.randrange(50, 300)
        self.top = self.height - image_bottom.get_height()
        self.bottom = self.height + self.GAP

        self.image_top = image_top
        self.image_bottom = image_bottom

        self.passed = False  # collision purpose

    def move(self):
        self.position.x = self.position.x - self.VELOCITY

    def isOutsideScreen(self):
        return (self.position.x + self.image_top.get_width()) < 0
