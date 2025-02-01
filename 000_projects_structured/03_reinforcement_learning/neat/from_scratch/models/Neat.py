
from sympy import S
from .data_structures.HashSorted import HashSorted
from .evolution.Genome import Genome, GenomeConfig
from .network.Neuron import *
from .evolution.Specie import Specie, SpecieConfig
from .network.Activation import ActivationFunction
from .network.Connection import *
from tqdm import tqdm


class Neat:

    species:HashSorted = None # type Specie
    genomes:list = [] # type Genome


    def __init__(
        self, input_size:int, output_size:int, population:int, epochs:int, 
        configGenome:GenomeConfig, configSpecie:SpecieConfig, elitist_save:int = 2,
        activationFunctionHidden = ActivationFunction.relu, activationFunctionOutput = ActivationFunction.sigmoid_steepened
    ):
        self.genome_best = None
        self.genomes = []
        self.species = HashSorted()

        self.elitist_save = elitist_save
        
        self.activationFunctionHidden = activationFunctionHidden
        configGenome.activationFunctionHidden = activationFunctionHidden
        self.activationFunctionOutput = activationFunctionOutput

        self.configGenome = configGenome
        self.configSpecie = configSpecie
        # The stagnation number varies with probability of genomes and total number of epochs
        

        self.reset(input_size, output_size, population,epochs)


    def reset(self, input_size:int, output_size:int, population:int,epochs:int)->None:
        # print("d:NEAT:reset")

        self.generation = 0

        self.input_size = input_size
        self.output_size = output_size
        self.POPULATION = population
        self.epochs = epochs
        Connection.ALL_CONNECTIONS.clear()
        Neuron.map_neuron_innovation_number.clear()
        self.genomes.clear()
        

        # creates base input neurons and save them as reference
        for i in range(input_size):
            y = (i+1)/float(input_size+1)
            innovation_number:int = i+1
            Neuron.getNeuronNew(0.1,y,innovation_number, self.activationFunctionHidden, self.configGenome.configNeuron)

        # creates base output neurons and save them as reference
        for i in range(output_size):
            y = (i+1)/float(output_size+1)
            innovation_number:int = input_size + (i+1)
            Neuron.getNeuronNew(0.9,y,innovation_number,self.activationFunctionOutput, self.configGenome.configNeuron)

        # creates dynamic genomes list
        for i in range(self.POPULATION):
            genome:Genome = Genome.empty_genome(input_size, output_size, self.activationFunctionHidden, connect_input_output=True,config=self.configGenome)
            self.addGenomeAndSpeciate(genome)

    
    def run(self, eval_function, FITNESS_THRESHOLD=math.inf, verbose_level=0):
        for i in tqdm(range(self.epochs)):
        
        
            # we can collect scores by frame, in this case we can directly collect from functions
            for j in range(len(self.genomes)):
                genome:Genome = self.genomes[j]
                genome.setScore(eval_function(genome))
            self.evolve(verbose_level=verbose_level-1)
            
            if self.genome_best.score > FITNESS_THRESHOLD:
                break
            
        return self.genome_best
    
    def isExtinction(self):
        return self.species.size() == 0
               
    def evolve(self,verbose_level=1,debug_step=1)->None:
                
        self.removeStagnantSpecies()
        if self.isExtinction():
            print("EXTINCTION, RESET")            
            self.reset(self.input_size, self.output_size, self.POPULATION, self.epochs)
            return
        
        fitnesses_adjusted = self.calculateFitnessesSpecies()
        # self.removeSpeciesWeak()
        
        sizes_previous = [len(s.genomes.data) for s in self.species.data]
        min_species_size = max(2, self.elitist_save)
        sizes_actual = Neat.compute_spawn(fitnesses_adjusted, sizes_previous, self.POPULATION, min_species_size)
        
        self.generation += 1
        self.reducePopulation(SURVIVORS=0.2) # species[*].genomes already sorted to do this
        self.populate(sizes_actual, verbose_level=verbose_level-10)
        
        # show speciated population
        if verbose_level > 0 and (self.generation == 1 or self.generation == self.epochs or self.generation % debug_step == 0): 
            self.printSpecies(verbose_level-1)
            print(end="")
            # print(sizes_previous, sizes_actual)
        
    
        

        
    def populate(self, sizes:list, verbose_level=0)->None:
        """
        Genomes of each specie have to be sorted before
        Genomes % only have to be alive before
        """
        
        population_new:list = []
        
        for i in range(len(sizes)):
            # each species always at least gets to retain its elites.
            size_remaining: int = max(sizes[i], self.elitist_save)
            specie: Specie = self.species.get(i)
            
            index = specie.genomes.size()-1
            
            # Transfer elites to new generation.                    
            for i in range(min(self.elitist_save, len(specie.genomes.data))):
                population_new.append(Genome.copy(specie.genomes.data[index-i], copy_specie=False))
                size_remaining -= 1
                
            if size_remaining <= 0:
                # specie complete
                continue
            
            while size_remaining > 0:
                size_remaining -= 1
                child:Genome = specie.breed(self.configGenome)
                if verbose_level > 0:
                    print("CHILD BREED: ", child)
                history = child.mutate()
                if verbose_level > 0:
                    print("CHILD MUTAT: ", child)
                    for key, value in history.items():
                        s = history[key]["summary"]
                        if s == "Nothing":
                            continue
                        print(key, s)
                child.id_specie = -1
                population_new.append(child)
                        
         
        # empty current species and genomes references
        for index in range(self.species.size()):
            specie:Specie = self.species.get(index)
            specie_genomes: HashSorted = specie.genomes
            for genome in specie_genomes.data:
                # remove the references of genomes in central neat instance
                self.genomes.remove(genome)

            while specie_genomes.size() > 0:
                specie_genomes.removeByIndex(0)
                
            assert len(specie_genomes.data) == 0
            assert not specie.representative is None
            
            # set new representative from the new population
            candidates = []
            for g in population_new:
                d = specie.distance(specie.representative, g)
                candidates.append((d, g))
                
            _, new_rep = min(candidates, key=lambda x: x[0])
            specie.representative = new_rep
            self.genomes.append(new_rep)
            specie.put(new_rep)
            population_new.remove(new_rep)
            
        # representative deleted and added
        assert len(self.genomes) == self.species.size()   
        
        for genome in population_new:
            self.addGenomeAndSpeciate(genome)
                
    
    def removeExtinctSpecies(self)->None:
        i:int = self.species.size()-1
        while i >= 0:

            # if only representative is left...
            if self.species.get(i).genomes.size() <= 1:
                self.removeSpecie(i)

            i -= 1


    def removeSpeciesWeak(self)->None:
        # remove species that not aport in the total average sum
        total_sum:float = 0
        for specie in self.species.data:
            total_sum += specie.score


        i:int = self.species.size()-1
        while i >= 0:
            contribution:float = math.floor((self.species.get(i).score/total_sum) * self.POPULATION)
            if contribution < 1:
                self.removeSpecie(i)
            i -= 1

    def removeStagnantSpecies(self)->bool:
        # remove species that not aport in the total average sum
        
        removed = False
                
        i:int = self.species.size()-1
        while i >= 0:
            if self.species.get(i).generations_from_last_improve > self.species.get(i).config.STAGNATED_MAXIMUM:
                # print(f"REMOVED SPECIE {self.species.get(i).id}")
                self.removeSpecie(i)
                removed = True          
            i -= 1
        
        return removed
        

    def removeSpecie(self, index:int):
                
        for genome in self.species.get(index).genomes.data:
            # remove the references of genomes
            self.genomes.remove(genome)

        self.species.removeByIndex(index)

    def addGenomeAndSpeciate(self, genome:Genome):
        self.genomes.append(genome)
        self.addToSpecie(genome)

    def addToSpecie(self,genome:Genome)->None:
        """
        Assumes the representative was already defined
        """
            
        assert genome.id_specie == -1
        
        candidates = []
        for i in range(len(self.species.data)):
            specie:Specie = self.species.data[i]
            distance_similarity:float = specie.distance(genome, specie.representative)
            if distance_similarity < specie.config.specie_threshold:
                # it is part of the specie
                candidates.append((distance_similarity,specie))
        
        if len(candidates) > 0:
            _, specie = min(candidates, key=lambda x: x[0])
            specie.put(genome)
        else:
            self.species.add(Specie(genome, self.configSpecie))           
            

    def calculateFitnessesSpecies(self, increase_generation=True):
        """
        Set fitness and adjusted fitness for every specie
        """
        # print("Calculating fitness")
        fitnesses_all = []
        # evaluate scores for each specie
        for i in range(len(self.species.data)):
            s:Specie = self.species.data[i]
            s.genomes.data.sort(key=lambda genome: genome.score)
            
            
            v:float = 0
            for i in range(len(s.genomes.data)):
                genome:Genome = s.genomes.data[i]
                s.best_score_genome = round(max(s.best_score_genome, genome.score),2)                
                v += genome.score
                fitnesses_all.append(genome.score)
                
                #if len(genome.connections.data) > 3:
                #    print(genome)

            if self.genome_best is None or self.genome_best.score < s.genomes.data[-1].score:
                self.genome_best = s.genomes.data[-1]
            
            # calculate the mean specie fitness (msf)
            s.score = v/s.genomes.size()
            
            
            if increase_generation:
                s.increaseGeneration()
            
        # calculating adjusted fitnesses
        min_fitness = min(fitnesses_all)
        max_fitness = max(fitnesses_all)
        # Do not allow the fitness range to be zero, as we divide by it below.
        fitnesses_adjusted = []
        fitness_range = max(1.0, max_fitness - min_fitness)
        for i in range(len(self.species.data)):
            s:Specie = self.species.data[i]
            s.fitness_adjusted = (s.score - min_fitness) / fitness_range
            fitnesses_adjusted.append(s.fitness_adjusted)

        return fitnesses_adjusted
        
    
    def reducePopulation(self, SURVIVORS:float = 0.2)->None:
        # kill X% of population
        for i in range(len(self.species.data)):
            specie:Specie = self.species.data[i]
            percentage = 1-SURVIVORS
            # kill clients with low score
            amount_to_kill:int = int(math.floor(percentage * specie.genomes.size()))
            amount_survive = specie.genomes.size() - amount_to_kill
            if amount_survive < 2:
                amount_to_kill -= (2 - amount_survive) # increase survivors to minimum 2
            

            i:int = 0
            while i < amount_to_kill:
                genome:Genome = specie.genomes.get(0)
                genome.id_specie = -5
                genome.score = 0
                specie.genomes.removeByIndex(0)
                self.genomes.remove(genome)
                i += 1


    def getBestGenome(self)->Genome:

        best_score = -math.inf
        best_genome = None
        for genome in self.genomes:
            if genome.score > best_score:
                best_score = genome.score
                best_genome = genome
        return best_genome

    def printSpecies(self,verbose_level=0)->None:
        print("##########################################")
        for i in range(len(self.species.data)):
            s:Specie = self.species.data[i]
            print(s)
        
        if verbose_level > 0:
            print("Best genome", self.genome_best)
        print("##########################################")

    @staticmethod
    def compute_spawn(fitnesses_adjusted, previous_sizes, pop_size, min_species_size):
        """
        Compute the proper number of offspring per species (proportional to fitness).
        Code got from neat-python library
        """
        af_sum = sum(fitnesses_adjusted)

        spawn_amounts = []
        for af, ps in zip(fitnesses_adjusted, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts