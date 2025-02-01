from models.evolution.Genome import Genome
from models.network.Neuron import ConfigNeuron
from models.network.Connection import ConfigConnection
from models.Neat import Neat
from tqdm import tqdm
import random
import math
from models.evolution.Specie import SpecieConfig
from models.evolution.Genome import GenomeConfig
from models.network.Activation import ActivationFunction
import os

from models.network.Neuron import ConfigNeuron
from models.network.Connection import ConfigConnection
clear = lambda: os.system('clear')


class TestLogicGates:
    
    def scoreOR(self,genome:Genome)->float:
        # The resulting number was squared to give proportionally more fitness the closer a network was to a solution.
                
        fitness:float = 4.0
        
        xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (1.0,)]

        for xi, xo in zip(xor_inputs, xor_outputs):
            output = genome.forward(xi)
            fitness -= (output[0] - xo[0]) ** 2
            
        return fitness * fitness
    
    def scoreAND(self,genome:Genome)->float:
        # The resulting number was squared to give proportionally more fitness the closer a network was to a solution.
                
        fitness:float = 4.0
        
        xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        xor_outputs = [   (0.0,),     (0.0,),     (0.0,),     (1.0,)]

        for xi, xo in zip(xor_inputs, xor_outputs):
            output = genome.forward(xi)
            fitness -= (output[0] - xo[0]) ** 2
            
        return fitness * fitness
        
    def scoreXOR(self,genome:Genome)->float:
        # The resulting number was squared to give proportionally more fitness the closer a network was to a solution.
                
        fitness:float = 4.0
        
        xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

        for xi, xo in zip(xor_inputs, xor_outputs):
            output = genome.forward(xi)
            fitness -= (output[0] - xo[0]) ** 2
            
        return fitness * fitness
    
    def executeSanityCheck(self,begin=1,end=10,verbose_level = 0):
        input_size = 2 # + 1 bias
        output_size = 1
        max_population = 200 # 200
        epochs = 100

        configGenome:GenomeConfig = GenomeConfig(
            probability_mutate_connection_add=0.5,
            probability_mutate_connection_delete=0.5,
            probability_mutate_node_add= 0.2,
            probability_mutate_node_delete= 0.2,
            MAX_HIDDEN_NEURONS = 10,
            configNeuron=ConfigNeuron(bias_min_value=-2, bias_max_value=2),
            configConnection=ConfigConnection(weight_min_value=-5, weight_max_value=5)
        )

        configSpecie:SpecieConfig = SpecieConfig(
            STAGNATED_MAXIMUM = 20,
            probability_crossover = 0.9,
            C1=1,
            C2=1,
            C3=0.5,
            specie_threshold=3
        )


        # best, worst
        scores = [-math.inf, math.inf] 
        seeds = [-1, -1]
        bad_seeds = []

        

        for s in range(begin,end+1):
            
            # replicate experiment
            random.seed(s)

            neat:Neat = Neat(
                input_size,output_size,max_population,epochs,configGenome,configSpecie,elitist_save=2,
                activationFunctionHidden=ActivationFunction.relu,
                activationFunctionOutput=ActivationFunction.sigmoid_steepened
            )

            result = self.run_experiment(neat,gate="xor",verbose_level=verbose_level-1).score

            if result <= 3:
                bad_seeds.append(s)
                print(f"Bad score {s}", result)
            if result > scores[0]:
                scores[0] = result
                seeds[0] = s

            if result < scores[1]:
                scores[1] = result
                seeds[1] = s

            
        print("best,worst")
        print(scores)
        print(seeds)
        print("BAD seeds", bad_seeds)

    def printSummary(self,genome:Genome):
        print(f"BEST SCORE {genome.score}")
        print("CONNECTIONS: ", genome.connections.size())
        print(genome)
        for n in genome.hidden_neurons.data:
            # print("BIAS: ", n.bias)
            pass
        print(f"0 - 0 =", round(genome.forward([0,0])[0],2))
        print(f"0 - 1 =", round(genome.forward([0,1])[0],2))
        print(f"1 - 0 =", round(genome.forward([1,0])[0],2))
        print(f"1 - 1 =", round(genome.forward([1,1])[0],2))
        print("Mutation rates:")
        print(genome.mutation_rates)

    def run_experiment(self,neat:Neat, gate:str="xor",verbose_level=0)->Genome:

        epochs = neat.epochs
        completed = False
        debug_step = epochs//100 # epochs//10 
        for i in tqdm(range(epochs)):
            
            
            # we can collect scores by frame, in this case we can directly collect from functions
            for j in range(len(neat.genomes)):
                genome:Genome = neat.genomes[j]
                # set score and key to sort by
                if gate.lower() == "xor":
                    genome.setScore(self.scoreXOR(genome))
                elif gate.lower() == "or":
                    genome.setScore(self.scoreOR(genome))
                elif gate.lower() == "and":
                    genome.setScore(self.scoreAND(genome))
                else:
                    raise Exception(f"Unknown gate '{gate}'")
                
                if genome.score > 15.8:
                    completed = True
                    break
            if completed:
                break
            
            if verbose_level > 10:
                print("\nNEW EPOCH")
                for j in range(len(neat.genomes)):
                    genome:Genome = neat.genomes[j]
                    print(f"GB{j}: {genome}")
                
                
            neat.evolve(verbose_level=verbose_level-1)
            # best_genome = neat.getBestGenome()
            # self.printSummary(best_genome)
            # print()

        # sanity score: check
        # evaluate final genomes
        for j in range(len(neat.genomes)):
            genome:Genome = neat.genomes[j]
            # set score and key to sort by
            if gate.lower() == "xor":
                genome.setScore(self.scoreXOR(genome))
            elif gate.lower() == "or":
                genome.setScore(self.scoreOR(genome))
            elif gate.lower() == "and":
                genome.setScore(self.scoreAND(genome))
            else:
                raise Exception(f"Unknown gate '{gate}'")

        best_genome = neat.getBestGenome()

        
        if verbose_level > 0:
            self.printSummary(best_genome)
        return best_genome

    

clear()
test = TestLogicGates()
test.executeSanityCheck(0,0,5)