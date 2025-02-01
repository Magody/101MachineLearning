
from ..data_structures.HashSorted import HashSorted
from ..evolution.Genome import Genome, GenomeConfig
from ..network.Neuron import *
from ..network.Connection import *
import random
import math

class SpecieConfig:
    STAGNATED_MAXIMUM = 50
    probability_crossover:float = 0.75, # paper: In each generation, 25% of offspring resulted from mutation without crossover. With 75% we mutate new crossover
    C1=1, #->1 Specie delta disjoint
    C2=1, #->1 Specie delta excess
    C3=3, #->3 0.4 Specie delta weight # for DPNV, may be better idea to increase this to 3
    specie_threshold=3 #->4  # Specie delta threshold
    # when c1,c2,c3 relative higher than threshold -> more specie division
    def __init__(
        self,
        STAGNATED_MAXIMUM = 50,
        probability_crossover:float = 0.75,
        C1=1,
        C2=1,
        C3=3,
        specie_threshold=3
    ):
        self.STAGNATED_MAXIMUM = STAGNATED_MAXIMUM
        self.probability_crossover:float = probability_crossover
        self.C1=C1
        self.C2=C2
        self.C3=C3
        self.specie_threshold=specie_threshold


# A specie contains one or more genomes
class Specie:
    STATIC_COUNTER = -1

    id:int = -1
    genomes:HashSorted = None # type Genome
    representative:Genome = None


    def __init__(
        self,
        representative:Genome,
        config:SpecieConfig
    ):
        self.config = config

        Specie.STATIC_COUNTER += 1
        self.id = Specie.STATIC_COUNTER
        
        self.fitness_adjusted = 0
        self.prev_fitness = -math.inf  

        self.best_score:float = -math.inf
        self.best_score_genome:float = -math.inf
        self.generations_from_last_improve:int = 0
        self.generations_alive:int = 0

        self.can_reproduce:int = True

        self.genomes = HashSorted()
        representative.id_specie = self.id
        self.representative = representative
        self.genomes.add(representative)
        self.score:float = 0

    def __str__(self)->str:
        return f"Specie:{self.id}. Last: {self.generations_from_last_improve}. BestG: {self.best_score_genome}. BestFit: {self.best_score} Fit: {round(self.score,2)} Adj: {round(self.fitness_adjusted,2)}. Pop.: {self.genomes.size()}. Nodes: {len(self.representative.neurons)}. Conns: {self.representative.connections.size()}"
        

    """
    calculated the distance between this genome g1 and a second genome g2
        - g1 must have the highest innovation number!
    """
    def distance(self, g1:Genome, g2:Genome)->float:
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """
            
        len_connections_g1:int = g1.connections.size()
        len_connections_g2:int = g2.connections.size()

        highest_innovation_gene1 = 0
        if len_connections_g1 != 0:
            highest_innovation_gene1 = g1.connections.get(len_connections_g1-1).innovation_number

        highest_innovation_gene2 = 0
        if len_connections_g2 != 0:
            highest_innovation_gene2 = g2.connections.get(len_connections_g2-1).innovation_number

        if highest_innovation_gene1 < highest_innovation_gene2:
            g = g1
            g1 = g2
            g2 = g
            # recalculate lens
            len_connections_g1:int = g1.connections.size()
            len_connections_g2:int = g2.connections.size()
            
            
        distance_gene_node = 0.0
        disjoint_nodes = 0
        
        for k2,n2 in g2.neurons.items():
            if not k2 in g1.neurons:
                disjoint_nodes += 1
                
        for k1,n1 in g1.neurons.items():
            if not k1 in g2.neurons:
                disjoint_nodes += 1
                continue
                
            n2 = g2.neurons[k1]
            # Homologous genes compute their own distance value.
            distance_gene_node += n1.distance(n2, self.config.C3)

        max_nodes = max(1,len(g1.neurons), len(g2.neurons))
        distance_gene_node = (distance_gene_node +
                            (self.config.C1 * disjoint_nodes))/max_nodes


        index_g1:int = 0
        index_g2:int = 0
        disjoint:int = 0
        excess:int = 0
        weight_diff:float = 0
        similar:int = 0

        while index_g1 < len_connections_g1 and index_g2 < len_connections_g2:
            gene1:Connection = g1.connections.get(index_g1)
            gene2:Connection = g2.connections.get(index_g2)

            in1:int = gene1.innovation_number
            in2:int = gene2.innovation_number

            if in1 == in2:
                # similar gene
                similar += 1
                weight_diff += gene1.distance(gene2, self.config.C3)
                index_g1 += 1
                index_g2 += 1
            elif in1 > in2:
                # disjoint gene of b
                disjoint += 1
                index_g2 += 1
            else:
                # disjoint gene of a
                disjoint += 1
                index_g1 += 1

        weight_diff /= max(1,similar)
        excess = len_connections_g1 - index_g1

        # in the paper genes are connections, the same crossed in crossover
        g1_number_genes = len_connections_g1 # + len(g1.neurons)
        g2_number_genes = len_connections_g2 # + len(g2.neurons)

        # factor N, the number of genes in the larger genome, normalizes for genome size
        N = max(1, g1_number_genes,g2_number_genes)
        
        factor_disjoint:float = ((self.config.C1 * disjoint)/N)
        factor_excess:float = ((self.config.C2 * excess)/N)
        factor_weight:float = (weight_diff)

        distance_gene_connection = factor_disjoint + factor_excess + factor_weight
        return distance_gene_node + distance_gene_connection


    def put(self, genome:Genome)->bool:
        """
        genome have to be part of specie similarity < self.config.specie_threshold
        put only if correspond to specie
        """
        genome.id_specie = self.id
        self.genomes.add(genome)

    def forcePut(self, genome:Genome)->None:
        genome.id_specie = self.id
        self.genomes.add(genome)


    def increaseGeneration(self)->None:
        self.generations_alive += 1

        score_rounded:float = round(self.score, 2)
        
        if score_rounded > self.best_score:
            self.best_score = score_rounded
            self.generations_from_last_improve = 0
        else:
            self.generations_from_last_improve += 1
            
        self.prev_fitness = score_rounded

        
    def reset(self):
        
        self.representative = self.genomes.getRandomElement()

        for i in range(len(self.genomes.data)):
            genome:Genome = self.genomes.data[i]
            genome.id_specie = -1

        
        self.genomes.clear()
        self.genomes.add(self.representative)
        self.representative.id_specie = self.id
        self.score = 0

    def size(self)->int:
        return self.genomes.size()
    
    def breed(self, configGenome:GenomeConfig)->Genome:
        
        p:float = random.random()
        len_genomes:int = self.genomes.size()

        if p < self.config.probability_crossover:
            input_size:int = self.representative.input_neurons.size()
            output_size:int = self.representative.output_neurons.size()
            genome_container:Genome = Genome.empty_genome(input_size,output_size,connect_input_output=False,config=configGenome)
            # todo: we can control and improve if are the same?
            g1:Genome = self.genomes.get(random.randint(0,len_genomes-1))
            g2:Genome = self.genomes.get(random.randint(0,len_genomes-1))

            if g1.score > g2.score:
                return Genome.crossover(g1,g2,genome_container)
            return Genome.crossover(g2,g1,genome_container)

        return Genome.copy(self.genomes.get(random.randint(0,len_genomes-1)))

   