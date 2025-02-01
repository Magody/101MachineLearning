from multiprocessing import connection
import pygame
from models.Neat import *
import random
from models.evolution.Genome import Genome
from tqdm import tqdm
from tests import TestLogicGates




class Button:
    """Create a button, then blit the surface in the while loop"""
 
    def __init__(self, text,  pos, font, bg="black", feedback=""):
        self.x, self.y = pos
        self.font = pygame.font.SysFont("Arial", font)
        if feedback == "":
            self.feedback = "text"
        else:
            self.feedback = feedback
        self.change_text(text, bg)
 
    def change_text(self, text, bg="black"):
        """Change the text whe you click"""
        self.text = self.font.render(text, 1, pygame.Color("White"))
        self.size = self.text.get_size()
        self.surface = pygame.Surface(self.size)
        self.surface.fill(bg)
        self.surface.blit(self.text, (0, 0))
        self.rect = pygame.Rect(self.x, self.y, self.size[0], self.size[1])
 
    def show(self):
        screen.blit(self.surface, (self.x, self.y))
 
    def click(self, event, callback):
        x, y = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                if self.rect.collidepoint(x, y):
                    callback()


WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
TEXTCOLOR = WHITE

CONNECTION_ENABLED = (108, 235, 226)
CONNECTION_DISABLED = (156, 87, 19)

BACKGROUND_COLOR = (  0,   0,  0)




(width, height) = (900, 600)

running = True

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("NEAT")
screen.fill(BACKGROUND_COLOR)
button1 = Button(
    "[0,0]",
    (100, 100),
    font=30,
    bg="navy",
    feedback="You clicked me")
button2 = Button(
    "[0,1]",
    (200, 100),
    font=30,
    bg="navy",
    feedback="You clicked me")
button3 = Button(
    "[1,0]",
    (300, 100),
    font=30,
    bg="navy",
    feedback="You clicked me")
button4 = Button(
    "[1,1]",
    (400, 100),
    font=30,
    bg="navy",
    feedback="You clicked me")

pygame.display.update()

font = pygame.font.Font(None, 16)
font_neuron_output =  pygame.font.Font(None, 30)




######

input_size = 2 # + 1 bias
output_size = 1
max_population = 200 # 200
epochs = 1000
seed = 2
verbose_level = 10
test = TestLogicGates()

configGenome:GenomeConfig = GenomeConfig(
    probability_perturb = 0.9,
    probability_mutate_connections_weight = 0.8,
    probability_mutate_connection_add=0.05,
    probability_mutate_connection_delete=0.1,
    probability_mutate_node_add= 0.05,
    probability_mutate_node_delete= 0.05,
    std_weight_initialization = 0.99,
    weight_step = 0.1,
    MAX_HIDDEN_NEURONS = 3
)

configSpecie:SpecieConfig = SpecieConfig(
    STAGNATED_MAXIMUM = 20,
    probability_crossover = 0.9,
    C1=1,
    C2=1,
    C3=2,
    specie_threshold=3
)

neat:Neat = Neat(
    input_size,output_size,max_population,epochs,configGenome,configSpecie,elitist_save=2,
    activationFunctionHidden=ActivationFunction.relu,
    activationFunctionOutput=ActivationFunction.sigmoid
)


global current_input
current_input = [0,0]
best_genome = test.run_experiment(neat,seed=seed,gate="xor",verbose_level=verbose_level)

test.printSummary(best_genome)
global genome_draw
genome_draw = best_genome


#####

def clean():
    screen.fill(BACKGROUND_COLOR)
    pygame.display.update()

def getXYPixel(neuron):
    x = width * neuron.x
    y = height * neuron.y
    return (x,y)


def drawNeuron(neuron,offset=(0,0),color=BLUE):
    (x,y) = getXYPixel(neuron)
    x += offset[0]
    y += offset[1]
    pygame.draw.circle(screen, color, (x,y), 12)
    x -= 5
    y -= 3
    screen.blit(font_neuron_output.render(f"{round(neuron.output,2)}", True, WHITE), (x,y))




def drawConnections(connections:list):

    for i in range(len(connections)):
        connection:Connection = connections[i]

        point_start = getXYPixel(connection.neuronFrom)
        point_end = getXYPixel(connection.neuronTo)

        pygame.draw.line(
            screen,
            CONNECTION_ENABLED if connection.enabled else CONNECTION_DISABLED, 
            point_start, 
            point_end
        )

        point_middle = (
            (point_end[0]+point_start[0])/2 - 30,
            (point_end[1]+point_start[1])/2,
        )
        screen.blit(font.render(f"{round(connection.weight,3)}", True, WHITE), point_middle)


def drawGenome(genome:Genome):
    clean()

    global current_input
    genome.forward(current_input)

    

    for neuron in genome.input_neurons.data:
        drawNeuron(neuron,color=BLUE)

    for neuron in genome.hidden_neurons.data:
        drawNeuron(neuron,color=(59, 94, 68))

    for i,neuron in enumerate(genome.output_neurons.data):
        drawNeuron(neuron,color=(125, 25, 60))

    drawConnections(genome.connections.data)

   
    

    pygame.display.update()

######


def callbackForward1():
    global current_input, genome_draw
    current_input = [0,0]
    drawGenome(genome_draw)

def callbackForward2():
    global current_input, genome_draw
    current_input = [0,1]
    drawGenome(genome_draw)

def callbackForward3():
    global current_input, genome_draw
    current_input = [1,0]
    drawGenome(genome_draw)

def callbackForward4():
    global current_input, genome_draw
    current_input = [1,1]
    drawGenome(genome_draw)

#genome_draw = neat.empty_genome()

drawGenome(genome_draw)

while running:
    ev = pygame.event.get()

    button1.show()
    button2.show()
    button3.show()
    button4.show()
    pygame.display.update()

    for event in ev:
        button1.click(event, callbackForward1)
        button2.click(event, callbackForward2)
        button3.click(event, callbackForward3)
        button4.click(event, callbackForward4)
        

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:
                genome_draw = neat.empty_genome()
                clean()
                drawGenome(genome_draw)

            if event.key == pygame.K_t:
                drawGenome(genome_draw)

            if event.key == pygame.K_a:
                neat.mutateGenomeNode(genome_draw)
                drawGenome(genome_draw)
            
            if event.key == pygame.K_s:
                neat.mutateGenomeLink(genome_draw)
                drawGenome(genome_draw)

            if event.key == pygame.K_d:
                genome_draw.mutateRandomLinkToggle()
                drawGenome(genome_draw)
            
            if event.key == pygame.K_w:
                genome_draw.mutateConnectionWeights()
                drawGenome(genome_draw)


        if event.type == pygame.QUIT:
            running = False
