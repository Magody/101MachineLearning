import pygame
from models.evolution.Genome import Genome
from models.Neat import Neat
from tqdm import tqdm
import random
import math
from models.evolution.Specie import SpecieConfig
from models.evolution.Genome import GenomeConfig
import os
from models.network.Activation import ActivationFunction
from resources.flappy_bird.PygameDrawer import PygameDrawer
from resources.flappy_bird.models.Bird import Bird
from resources.flappy_bird.models.Pipe import Pipe
from resources.flappy_bird.models.Base import Base
from resources.flappy_bird.Parameters import Parameters
from models.network.Neuron import ConfigNeuron
from models.network.Connection import ConfigConnection

clear = lambda: os.system('clear')


def getPygameImageFromPath(image_name):
    path_root:str = "/home/magody/programming/python/data_science/projects/neat/from_scratch/"
    path_img:str = os.path.join(f"{path_root}resources/flappy_bird/imgs", image_name)
    print(path_img)
    return pygame.transform.scale2x(pygame.image.load(path_img))

def getPygameMask(image):
    return pygame.mask.from_surface(image)

def flipImage(image, xbool, ybool):
    return pygame.transform.flip(image, xbool, ybool)


GENERATION = 0
pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)

# Section: Transforming to pygame images
BASE_PYGAME_IMAGES = [getPygameImageFromPath(Parameters.BASE_IMAGE)]
BACKGROUND_PYGAME_IMAGE = [getPygameImageFromPath(Parameters.BACKGROUND_IMAGE)]
BIRD_PYGAME_IMAGES = [getPygameImageFromPath(image) for image in Parameters.BIRD_IMAGES]
PIPE_PYGAME_IMAGES = [getPygameImageFromPath(image) for image in Parameters.PIPE_IMAGES]

PIPE_PYGAME_IMAGE_DOWNWARDS = flipImage(PIPE_PYGAME_IMAGES[0], False, True)
PIPE_PYGAME_IMAGE_UPWARDS = PIPE_PYGAME_IMAGES[0]


Bird.images['BIRD_WINGS_UP'] = BIRD_PYGAME_IMAGES[0]
Bird.images['BIRD_WINGS_MIDDLE'] = BIRD_PYGAME_IMAGES[1]
Bird.images['BIRD_WINGS_DOWN'] = BIRD_PYGAME_IMAGES[2]

pygame_window = pygame.display.set_mode((Parameters.WINDOW_WIDTH, Parameters.WINDOW_HEIGHT))
drawer = PygameDrawer(pygame_window, STAT_FONT, Parameters.WINDOW_WIDTH, Parameters.WINDOW_HEIGHT)
clock = pygame.time.Clock()


def existCollition(bird, pipe):
    bird_mask = getPygameMask(bird.image_actual)
    top_mask = getPygameMask(pipe.image_top)
    bottom_mask = getPygameMask(pipe.image_bottom)
    top_offset = (pipe.position.x - bird.position.x, pipe.top - round(bird.position.y))
    bottom_offset = (pipe.position.x - bird.position.x, pipe.bottom - round(bird.position.y))

    b_point = bird_mask.overlap(bottom_mask, bottom_offset)
    t_point = bird_mask.overlap(top_mask, top_offset)

    return b_point or t_point


def findPipeToAnalize(birds, pipes):
    if len(pipes) > 1 and birds[0].position.x > pipes[0].position.x + pipes[0].image_top.get_width():
        return 1
    else:
        return 0


def distance(a, b):
    return abs(a - b)


def fitnessFunction(neat: Neat):
    global GENERATION
    GENERATION += 1
    score = 0

    genomes_copy = []  # with initial fitness equal to 0
    birds = []

    for genome in neat.genomes:
        birds.append(Bird(Parameters.WINDOW_WIDTH//2 - 20, Parameters.WINDOW_HEIGHT//2 - 50))
        genome.score = 0
        genomes_copy.append(genome)  # check: reference

    base = Base(BASE_PYGAME_IMAGES[0], Parameters.WINDOW_HEIGHT-Parameters.BASE_HEIGHT)
    pipes = [Pipe(Parameters.WINDOW_WIDTH+100, PIPE_PYGAME_IMAGE_DOWNWARDS, PIPE_PYGAME_IMAGE_UPWARDS)]

    game_running = True

    sum_output1 = 0
    sum_output2 = 0
    while game_running:


        # PARA AUMENTAR LA VELOCIDAD COMENTAR ESTO
        # clock.tick(30)  # 30 ticks every second

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
                pygame.quit()
                quit()

        if len(birds) == 0:
            game_running = False
            continue

        # get the next pipe in front
        pipe_index = findPipeToAnalize(birds, pipes)
        # action for the frame
        assert len(birds) == len(genomes_copy)
        for index, bird in enumerate(birds):
            # gravity to the bird 
            bird.move()
            # each frame alive means a reward of 0.1
            genomes_copy[index].score += 0.1

            div_normalization = 1  # 1 Parameters.WINDOW_HEIGHT
            input:list = [
                bird.position.y/div_normalization,
                distance(bird.position.y, Parameters.WINDOW_HEIGHT)/div_normalization,
                distance(bird.position.y, pipes[pipe_index].height)/div_normalization,
                distance(bird.position.y, pipes[pipe_index].bottom)/div_normalization
            ]
            output:list = genomes_copy[index].forward(input)
            
            
                
                
            if output[0] > 0.5:
                sum_output1 += 1/len(birds)
                bird.jump(-9)
            else:
                sum_output2 += 1/len(birds)
            

        add_pipe = False
        pending_remove = []
        for pipe in pipes:

            for index, bird in enumerate(birds):
                if existCollition(bird, pipe):
                    genomes_copy[index].score -= 1
                    # dead
                    if len(birds) == 1:
                        print(genomes_copy[index])
                    birds.pop(index)
                    genomes_copy.pop(index)

                if not pipe.passed and pipe.position.x < bird.position.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.isOutsideScreen():
                pending_remove.append(pipe)

            # movement to left in this frame
            pipe.move()

        if add_pipe:
            # passed a pipe
            score = score + 1

            for genome in genomes_copy:
                genome.score += 5

            pipes.append(Pipe(Parameters.WINDOW_WIDTH, PIPE_PYGAME_IMAGE_DOWNWARDS, PIPE_PYGAME_IMAGE_UPWARDS))

        for pending in pending_remove:
            pipes.remove(pending)

        for index, bird in enumerate(birds):
            if bird.isOutsideScreen(Parameters.WINDOW_HEIGHT, Parameters.BASE_HEIGHT):
                # dead but without penalization
                if len(birds) == 1:
                    print(genomes_copy[index])
                birds.pop(index)
                genomes_copy.pop(index)

        # la base tiene su velocidad
        base.move()

        drawer.draw_window(birds, pipes, base, score, BACKGROUND_PYGAME_IMAGE[0], GENERATION)
        pygame.display.update()

    
    total = sum_output1 + sum_output2
    # print(sum_output1/total, sum_output2/total)
    # print()


if __name__ == '__main__':
    clear()
    # seed = 2
    # replicate experiment
    # random.seed(seed)
    verbose_level = 5
    
    input_size = 4
    output_size = 1
    max_population = 200
    epochs = 100

    configGenome:GenomeConfig = GenomeConfig(
        probability_mutate_connection_add=0.5,
        probability_mutate_connection_delete=0.5,
        probability_mutate_node_add= 0.2,
        probability_mutate_node_delete= 0.2,
        MAX_HIDDEN_NEURONS = 20,
        configNeuron=ConfigNeuron(bias_min_value=-30, bias_max_value=30),
        configConnection=ConfigConnection(weight_min_value=-30, weight_max_value=30)
    )

    configSpecie:SpecieConfig = SpecieConfig(
        STAGNATED_MAXIMUM = 20,
        probability_crossover = 0.8,
        C1=1,
        C2=1,
        C3=0.5,
        specie_threshold=3
    )
    
    neat:Neat = Neat(
        input_size,output_size,max_population,epochs,configGenome,
        configSpecie,elitist_save=2, 
        activationFunctionHidden=ActivationFunction.tanh,
        activationFunctionOutput=ActivationFunction.sigmoid
    )

    for i in tqdm(range(epochs)):
        fitnessFunction(neat)
        neat.evolve(verbose_level=verbose_level-1,debug_step=1)
    
    print(neat.genome_best)