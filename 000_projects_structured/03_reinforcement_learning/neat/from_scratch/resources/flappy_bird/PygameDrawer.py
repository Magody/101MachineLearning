from pygame.transform import rotate


class PygameDrawer:

    STAT_FONT = None
    WINDOW_WIDTH = 0
    WINDOW_HEIGHT = 0

    def __init__(self, window, stat_font, window_width, window_height):
        self.window = window
        self.STAT_FONT = stat_font
        self.WINDOW_WIDTH = window_width
        self.WINDOW_HEIGHT = window_height

    def draw_window(self, birds, pipes, base, score, background_image, gen):
        self.window.blit(background_image, (0, 0))

        for pipe in pipes:
            self.__drawPipe(pipe)

        text = self.STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
        self.window.blit(text, (self.WINDOW_WIDTH - 10 - text.get_width(), 10))

        text2 = self.STAT_FONT.render("Generation: " + str(gen), 1, (255, 255, 255))
        self.window.blit(text2, (10, 10))

        text3 = self.STAT_FONT.render("Population (genomes): " + str(len(birds)), 1, (255, 255, 255))
        self.window.blit(text3, (10, 40))

        self.__drawBase(base)

        for bird in birds:
            self.__drawBird(bird)

    def __drawBase(self, base):
        self.window.blit(base.image, (base.x_figure1, base.position.y))
        self.window.blit(base.image, (base.x_figure2, base.position.y))

    def __drawBird(self, bird):
        bird.image_count = bird.image_count + 1
        bird.chooseImageOfAnimation(bird.image_count)
        rotated_image = rotate(bird.image_actual, bird.tilt)
        new_rect = rotated_image.get_rect(center=bird.image_actual.get_rect(topleft=(bird.position.x, bird.position.y)).center)
        self.window.blit(rotated_image, new_rect.topleft)

    def __drawPipe(self, pipe):
        self.window.blit(pipe.image_top, (pipe.position.x, pipe.top))
        self.window.blit(pipe.image_bottom, (pipe.position.x, pipe.bottom))
