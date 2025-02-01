
from Chromosome import Chromosome
import random
import math
import copy
import numpy as np

class Population:
    THRESHOLD_SIMILARITY = 0.99
    
    fitness_best = -1

    def __init__(self, gens_set: list, max_population: int=10, mutation_rate: float=0.9):
        # array of Chromosome
        self.chromosomes = []
        # set of posible gens for population. ALL GENS, this is the universe
        self.gens_set = gens_set
        # max population
        self.max_population = max_population
        # mutation percentage
        self.mutation_rate = mutation_rate

        self.fitness_avg = 0

    
    def generate_initial_population(self, num_gens) -> None:
        # initial population
        self.chromosomes = []
        for _ in range(self.max_population):
            self.chromosomes.append(Chromosome(self.gens_set, num_gens))

    def fitness_function(self, equation_inputs: list) -> list:
        # i assume that the aptitude function will be the same as fitness function
        # (without pondering values)

        # Calculating the fitness value of each solution in the current population.
        # The fitness function calculates the sum of products between each input and its corresponding weight.

        # equation_inputs dims=1*N (N is the number of variables X)
        # population dims=M*N (M is the number of individuals, each row is a cromosome)

        # each cromosome has its score by Y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6

        fitness = [0 for _ in range(self.max_population)]
        self.fitness_avg = 0

        for index_individual in range(self.max_population):


            fitness[index_individual] = np.matmul(np.array(self.chromosomes[index_individual].gens), np.transpose(np.array(equation_inputs)))

        return fitness

    def mutate(self, offspring_crossover: list):
        # Mutation changes a single gene in each offspring randomly.
        # receive chromosome list
        offspring_mutated = offspring_crossover

        num_individuals = len(offspring_mutated)
        num_gens = len(offspring_mutated[0].gens)

        for index_individual in range(num_individuals):
            # The random value to be added to the gene.

            # throw dice
            if random.random() < self.mutation_rate:

                random_gen = random.randint(0, num_gens-1)
                offspring_mutated[index_individual].gens[random_gen] = random.choice(self.gens_set)
           

        return offspring_mutated

    @staticmethod
    def select_best_individuals_by_elitist(fitness: list, num_best_individuals: int) -> list:
        """
        elitist method only select the two best parents
        Selecting the best individuals in the current generation as 
        parents for producing the offspring of the next generation.
        """

        best_individuals_values = [-math.inf for _ in range(num_best_individuals)]
        best_individuals_index = [0 for _ in range(num_best_individuals)]

        for index_individual in range(len(fitness)):
            individual_fitness = fitness[index_individual]

            first_min_value = best_individuals_values[0]
            first_min_index = 0

            for index_min, value in enumerate(best_individuals_values):
                if value < first_min_value:
                    first_min_value = value
                    first_min_index = index_min


            if individual_fitness > first_min_value:
                best_individuals_values[first_min_index] = individual_fitness
                best_individuals_index[first_min_index] = index_individual


        return best_individuals_index

    @staticmethod
    def crossover(parents: list, num_individuals: int) -> list:
        # IT CAN BE IMPROVED. Only making combinations of 2 parents
        # return Chromosomes list

        num_parents = len(parents)
        num_gens = len(parents[0].gens)
        
        offspring = []
        
        # The point at which crossover takes place between two parents. 
        # Usually, it is at the center. There are a lot of methods
        # better than this one.
        
        crossover_point = math.floor(num_gens/2)

        # its a secuencial crossover, which is bad due to prob. duplication

        for index_individual in range(num_individuals):
            # Index of the first parent to mate.
            parent1_idx = index_individual % num_parents
            # Index of the second parent to mate
            parent2_idx = (index_individual+1) % num_parents
            # The new offspring will have its first half of its genes taken from
            # the first parent.
            offspring.append(copy.deepcopy(parents[parent1_idx]))
            # The new offspring will have its second half of its genes taken from
            # the second parent. 
            # print(parent1_idx, parent2_idx, len(parents))
            # print("crossover_point", crossover_point)
            # print("BEFORE", offspring[-1])
            offspring[-1].gens[crossover_point:] = parents[parent2_idx].gens[crossover_point:]

            # print("AFTER", offspring[-1])


        return offspring

    @staticmethod
    def has_reached_the_top(fitness: list) -> bool:
        # function for convergency

        bool_value = np.std(fitness) < 0.0001

        return bool_value
        
        
