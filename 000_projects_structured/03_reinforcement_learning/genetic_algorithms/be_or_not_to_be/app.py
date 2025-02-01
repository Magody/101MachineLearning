from Population import Population
import matplotlib.pyplot as plt
import copy

def text2array_unicode(string: str) -> list:
    """
    Return an array of char ascii codes for each character in string
    """
    array_unicode = []
    for letter in string:
        array_unicode.append(ord(letter))
    return array_unicode

def array2text_unicode(array: list) -> str:

    string = ""
    for code in array:
        string += chr(code)
    return string


generations = 200  # its just a stop step in case the program cant find solution
max_population = 50  # while more high, more posibilities to find a good combination faster, but more processing
num_parents_to_select = 3  # is better tu select very little, no more than 10% of max population

# all posible gens
gens_set = text2array_unicode("abcdefghijklmnopqrstuvwxyz.-, ")
mutation_rate = 0.9

target = "to be or not to be"


target_unicode_set = text2array_unicode(target)
num_gens = len(target_unicode_set)  # length of text

population = Population(gens_set, max_population, mutation_rate)
population.generate_initial_population(num_gens)

# PLOT VARIABLES
history_fitness_mean = []
history_change = []



for generation in range(1, generations+1):

    # fitness is a list with scores of each individual
    fitness = population.fitness_function(target_unicode_set)
    
    # being elitist means that the parents are the best 2 (maybe a little more)
    
    best_individuals_index = Population.select_best_individuals_by_elitist(fitness, num_parents_to_select)
    parents_selected = []
    parents_selected_fitness = []

    for best_index in best_individuals_index:
        
        parents_selected.append(copy.deepcopy(population.chromosomes[best_index]))
        parents_selected_fitness.append(fitness[best_index])
        
    
    ################### RESULTS ACTUAL GENERATION ###################
    print(f"\n*\nBest result for generation {generation}\n")
    print("Individuals (chromosomes)", [str(parent) for parent in parents_selected])
    
    print("fitness for individuals", parents_selected_fitness)
    
    history_fitness_mean.append(sum(fitness)/len(fitness))
    ###################
    
    offspring_crossover = Population.crossover(copy.deepcopy(parents_selected), max_population-num_parents_to_select)

    # Creating the new population based on the parents and offspring.
    offspring_mutated = population.mutate(offspring_crossover)

    #print(f"Selected{len(parents_selected)}, mutated{len(offspring_mutated)}, chrom{len(population.chromosomes)}")
    # print("PARENTS SELECTED", [str(parent) for parent in parents_selected])
    # print("SELECTED MUTATED", [str(parent) for parent in offspring_mutated])
    population.chromosomes[:num_parents_to_select] = parents_selected[:]
    population.chromosomes[num_parents_to_select:] = offspring_mutated[:]
    #print([str(parent) for parent in population.chromosomes])
    
    
    if Population.has_reached_the_top(parents_selected_fitness, target_unicode_set):
       print("FUNCTION HAS CONVERGED")
       break




final_generations = len(history_fitness_mean)


best_chromosome_index = Population.select_best_individuals_by_elitist(fitness, 1)
best_chromosome = population.chromosomes[best_chromosome_index[0]]
print("Answer", array2text_unicode(best_chromosome.gens))
print("Target", target)

"""
plt.plot(range(1, final_generations+1), history_fitness_mean)
plt.show()
"""

