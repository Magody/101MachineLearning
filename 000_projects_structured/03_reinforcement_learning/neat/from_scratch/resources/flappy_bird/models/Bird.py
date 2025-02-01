from .Position import Position


class Bird:

    ROTATION_MAX = 25
    ROTATION_VELOCITY = 20
    ANIMATION_TIME = 5

    images = {
        "BIRD_WINGS_UP": None,
        "BIRD_WINGS_MIDDLE": None,
        "BIRD_WINGS_DOWN": None
    }

    def __init__(self, initial_position_x, initial_position_y):
        self.position = Position(initial_position_x, initial_position_y)
        self.tilt = 0  # incline the bird
        self.tick_count = 0  # seconds that we were moving
        self.velocity = 0
        self.height = self.position.y
        self.image_count = 0
        self.image_actual = Bird.images["BIRD_WINGS_UP"]

    def jump(self, quantity):
        self.velocity = quantity
        self.tick_count = 0  # if velocity change
        self.height = self.position.y

    def move(self):
        self.tick_count = self.tick_count + 1

        move_down_pixels = self.velocity * self.tick_count + 1.5 * self.tick_count**2

        if move_down_pixels >= 16:
            move_down_pixels = 16  # clamp the value to maximun 16
        elif move_down_pixels < 0:  # moves upwards
            move_down_pixels = move_down_pixels - 2

        self.position.y = self.position.y + move_down_pixels

        if self.isMovingUpwards(move_down_pixels):
            self.rotateUpwards()
        elif self.tilt > -90:  # it allows us to rotate 90 degrees down, fast
            self.rotateDownwards()

    def chooseImageOfAnimation(self, image_count):
        """Animation hot encoded"""
        if image_count <= self.ANIMATION_TIME:
            self.image_actual = Bird.images['BIRD_WINGS_UP']
        elif image_count <= self.ANIMATION_TIME*2:
            self.image_actual = Bird.images['BIRD_WINGS_MIDDLE']
        elif image_count <= self.ANIMATION_TIME*3:
            self.image_actual = Bird.images['BIRD_WINGS_DOWN']
        elif image_count <= self.ANIMATION_TIME*4:
            self.image_actual = Bird.images['BIRD_WINGS_MIDDLE']
        elif image_count == self.ANIMATION_TIME*4 + 1:
            self.image_actual = Bird.images['BIRD_WINGS_UP']
            self.image_count = 0

        # Especial case when bird is rotates 80 degree down
        # The wings should stay at neutral level
        if self.tilt <= -80:
            self.image_actual = Bird.images['BIRD_WINGS_MIDDLE']
            self.image_count = self.ANIMATION_TIME*2  # Starts where the program should 'middle'

    def isMovingUpwards(self, move_down_pixels):
        return move_down_pixels < 0 or self.position.y < self.height + 50

    def rotateUpwards(self):
        if self.tilt < self.ROTATION_MAX:
            self.tilt = self.ROTATION_MAX

    def rotateDownwards(self):
        self.tilt = self.tilt - self.ROTATION_VELOCITY

    def isOutsideScreen(self, screen_height, base_height):
        came_out_below = self.position.y + self.image_actual.get_height() >= (screen_height - base_height)
        came_out_above = self.position.y < 0
        return came_out_above or came_out_below
