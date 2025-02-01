from .Position import Position


class Base:
    """
    Represents the floor moving
    """

    VELOCITY = 5

    def __init__(self, image, initial_position_y):
        self.image = image
        self.width = image.get_width()

        self.position = Position(0, initial_position_y)
        self.x_figure1 = 0
        self.x_figure2 = self.width

    def move(self):
        self.x_figure1 = self.x_figure1 - self.VELOCITY
        self.x_figure2 = self.x_figure2 - self.VELOCITY

        if self.isFigureOutsideTheWindow(self.x_figure1):
            self.x_figure1 = self.x_figure2 + self.width

        if self.isFigureOutsideTheWindow(self.x_figure2):
            self.x_figure2 = self.x_figure1 + self.width

    def isFigureOutsideTheWindow(self, x_figure):
        return (x_figure + self.width) < 0
