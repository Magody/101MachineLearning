import random

from ..network.Neuron import *
from ..network.Connection import *
from ..network.Network import *

from ..Colors import wc

class GenomeConfig:

    def __init__(
        self,
        probability_mutate_connection_add:float = 0.5, 
        probability_mutate_connection_delete:float = 0.5,
        probability_mutate_node_add:float = 0.2,
        probability_mutate_node_delete:float = 0.2,
        MAX_HIDDEN_NEURONS:int = 1,
        activationFunctionHidden = ActivationFunction.sigmoid_steepened,
        configNeuron: ConfigNeuron = None,
        configConnection: ConfigConnection = None
        
    ):
        self.probability_mutate_connection_add:float = probability_mutate_connection_add
        self.probability_mutate_connection_delete:float = probability_mutate_connection_delete
        self.probability_mutate_node_add:float = probability_mutate_node_add
        self.probability_mutate_node_delete:float = probability_mutate_node_delete
        self.MAX_HIDDEN_NEURONS:int = MAX_HIDDEN_NEURONS
        self.activationFunctionHidden = activationFunctionHidden
        self.configNeuron = configNeuron
        self.configConnection = configConnection;



class Genome(Network):

    score:float = 0

    id_specie: int = -1

    mutation_rates = dict()


    def __init__(
        self,
        input_neurons:list,
        hidden_neurons:list,
        output_neurons:list,
        config:GenomeConfig                
    ):
        super().__init__(input_neurons,hidden_neurons,output_neurons,config.activationFunctionHidden)

        self.id_specie = -1

        self.config = config

        self.setScore(0)
        # Mutation properties
        self.mutation_rates = dict()
        self.mutation_rates["connection_add"] = config.probability_mutate_connection_add
        self.mutation_rates["connection_delete"] = config.probability_mutate_connection_delete
        self.mutation_rates["node_add"] = config.probability_mutate_node_add
        self.mutation_rates["node_delete"] = config.probability_mutate_node_delete


    @staticmethod
    def empty_genome(input_size:int, output_size:int, prefill:bool=True, connect_input_output:bool=True, config:GenomeConfig=None):

        input_neurons:list = []
        output_neurons:list = []

        if prefill:

            for innovation_number in range(1, input_size+1):
                input_neurons.append(Neuron.getNeuron(innovation_number))

            for innovation_number in range(input_size+1,input_size+output_size+1):
                output_neurons.append(Neuron.getNeuron(innovation_number))

        genome = Genome(input_neurons,[],output_neurons,config)
        if connect_input_output:
            # initial connection
            for input_neuron in genome.input_neurons.data:
                for output_neuron in genome.output_neurons.data:
                    genome.insertConnection(Connection.getConnection(input_neuron,output_neuron, config.configConnection))

        return genome

    @staticmethod
    def copy(genome, copy_specie=True):

        genome_copy:Genome = Genome(
            [],
            [],
            [],
            genome.config
        )


        if copy_specie:
            genome_copy.id_specie = genome.id_specie
        genome_copy.score = genome.score

        

        for innovation_number,neuron in genome.neurons.items():
            genome_copy.insertNeuron(Neuron.copy(neuron))

        for connection in genome.connections.data:

            neuronFrom, neuronTo = genome_copy.getSingletonContainers(connection)
            genome_copy.insertConnection(Connection.copy(connection,neuronFrom, neuronTo))

        return genome_copy


    def evaluateInput(self,input:list)->list:
        return self.forward(input)

    def mutate(self, single_structural_mutation=False)->None:
        history_mutation = {
            "node_add":{"summary":"Nothing"},
            "node_delete":{"summary":"Nothing"},
            "connection_add":{"summary":"Nothing"},
            "connection_delete":{"summary":"Nothing"},
            "nodes":{"summary":"Nothing"},
            "connections":{"summary":"Nothing"}
        }
        # alter the rate for every mutation
        for mutation,rate in self.mutation_rates.items():

            if random.random() < 0.5:
                self.mutation_rates[mutation] = 0.95 * rate
            else:
                self.mutation_rates[mutation] = 1.05 * rate
        
        # Mutate: Topology
        if single_structural_mutation:
            
            div = max(1, (self.mutation_rates["node_add"] + self.mutation_rates["node_delete"] +
                          self.mutation_rates["connection_add"] + self.mutation_rates["connection_delete"]))
                        
            r:float = random.random()
            if r < ((self.mutation_rates["node_add"])/div):
                history_mutation["node_add"] = self.mutateGenomeNodeAdd()

            if r < ((self.mutation_rates["node_add"] + self.mutation_rates["node_delete"])/div):
                history_mutation["node_delete"] = self.mutateGenomeNodeDelete()
                
            if r < ((self.mutation_rates["node_add"] + self.mutation_rates["node_delete"] + self.mutation_rates["connection_add"])/div):
                history_mutation["connection_add"] = self.mutateGenomeConnectionAdd()
                
            if r < ((self.mutation_rates["node_add"] + self.mutation_rates["node_delete"] + self.mutation_rates["connection_add"] + self.mutation_rates["connection_delete"])/div):
                history_mutation["connection_delete"] = self.mutateGenomeConnectionDelete()
                
        else:
            
            # either add or delete
            if random.random() < self.mutation_rates["node_add"]:
                history_mutation["node_add"] = self.mutateGenomeNodeAdd()
            if random.random() < self.mutation_rates["node_delete"]:
                history_mutation["node_delete"] = self.mutateGenomeNodeDelete()
                
            if random.random() < self.mutation_rates["connection_add"]:
                history_mutation["connection_add"] = self.mutateGenomeConnectionAdd()
            if random.random() < self.mutation_rates["connection_delete"]:
                history_mutation["connection_delete"] = self.mutateGenomeConnectionDelete()

        history_mutation["nodes"] = self.mutateNeurons()
        history_mutation["connections"] = self.mutateConnectionWeights()
            
        return history_mutation


    def mutateGenomeConnectionAdd(self):
        """
        Add connection, if connection exists then enable
        """
        history = {"summary": "Nothing"}
        
        a, b = self.getRandomGenePair()
        assert a.x != b.x or a.innovation_number != b.innovation_number

        con:Connection = None
        if a.x < b.x:
            con = Connection(a,b, self.config.configConnection)
        else:
            con = Connection(b,a, self.config.configConnection)

        con = Connection.getConnection(con.neuronFrom, con.neuronTo, self.config.configConnection)

        if self.existConnection(con):
            if not con.enabled.value:
                con.enabled.value = True
                history["summary"] = f"Reenabled {con}"
            return history


        self.insertConnection(con)
        history["summary"] = f"Inserted connection {con}"
        return history
        
    def mutateGenomeConnectionDelete(self):
        history = {"summary": "Nothing"}
        con:Connection = self.getRandomConnection()
        if not con is None:
            # con.enabled = False
            self.removeConnection(con)
            history["summary"] = f"Removed connection {con}"
            
        return history
                
    def mutateGenomeNodeAdd(self):
        history = {"summary": "Nothing"}
        
        if self.hidden_neurons.size() > self.config.MAX_HIDDEN_NEURONS:
            return history
        
        # add hidden node in EXISTING connection
        con:Connection = self.getRandomConnection()
        
        if con is None:
            # there is no connections yet
            history = self.mutateGenomeConnectionAdd()
            return history
        

        from_neuron:Neuron = con.neuronFrom
        to_neuron:Neuron = con.neuronTo

        # always split the link into two nodes
        replace_index, key = Connection.getReplaceIndex(from_neuron,to_neuron,self.config.configConnection)

        middle:Neuron = None

        create_new_node:bool = True

        if replace_index > 0:
            # already exist a node in the connection
            if not self.existNeuronInnovationNumber(replace_index):
                # If already exist a neuron in general, but not in genome, create a copy
                middle = Neuron.getNeuron(replace_index)
                create_new_node = False
                history["summary"] = f"Add node:Reuse node {middle}"
        
        if create_new_node:
            # if no previous neuron was created between this connection
            x = (from_neuron.x + to_neuron.x)/2
            y = (from_neuron.y + to_neuron.y)/2 + (random.random() * 0.1 - 0.05)
            innovation_number:int = len(Neuron.map_neuron_innovation_number)+1
            middle:Neuron = Neuron.getNeuronNew(x,y,innovation_number,self.activationFunctionHidden, self.config.configNeuron)
            Connection.setReplaceIndex(from_neuron,to_neuron,middle.innovation_number,self.config.configConnection)
            
            history["summary"] = f"Add node:New node {middle}"
        

        con1:Connection = Connection.getConnection(from_neuron,middle, self.config.configConnection)
        con2:Connection = Connection.getConnection(middle,to_neuron, self.config.configConnection)

        # before link: set weight to 1
        con1.enabled.value = True
        con1.weight.value = 1
        
        # next link: restore the properties of original long connection
        con2.enabled.value = con.enabled.value
        con2.weight.value = con.weight.value

        # if input [1,2,3] and output [4]
        # adding hidden node is hidden [5]
        # originally 1->4, and we want to put the node 5 in middle
        # now 1->5->4. the original connection 1->4 have to be removed
        # genome.removeConnection(con) If we remove, then in a mutation will create it again... a loop forever
        # con.enabled = False
        self.removeConnection(con)
        # the middle is a brand new or existing in general storage mapping
        self.insertNeuron(middle)

        # now we add the segmented connection
        self.insertConnection(con1)
        self.insertConnection(con2)
        
        
        history["summary"] += f". Add Connections: {con1} {con2} to replace {con}"
        
        
        return history

    def mutateGenomeNodeDelete(self):
        history = {"summary": "Nothing"}
        if len(self.hidden_neurons.data) == 0:
            # Nothing to delete
            return history
        
        neuron_to_delete: Neuron = random.choice(self.hidden_neurons.data)
        
        connections_to_delete = []
        
        for i in range(len(self.connections.data)):
            connection: Connection = self.connections.data[i]
            is_neuron_from = connection.neuronFrom.innovation_number == neuron_to_delete.innovation_number
            is_neuron_to = connection.neuronTo.innovation_number == neuron_to_delete.innovation_number
            if is_neuron_from or is_neuron_to:
                connections_to_delete.append(connection)                    
        
        for connection in connections_to_delete:
            self.removeConnection(connection)
            
        self.removeNeuron(neuron_to_delete)
        
        history["summary"] = f"Removed node: {neuron_to_delete} and its connections {connections_to_delete}"
        return history
    
    def mutateNeurons(self):
        
        history = {"summary": ""}
        
        for innovation_number, neuron in self.neurons.items():
            if neuron.x == 0.1:
                continue
            history["summary"] += f"\nNeuron {neuron} mutate: {neuron.mutate()['summary']}"
            
        return history
            

    def mutateConnectionWeights(self):
        
        history = {"summary": "Nothing"}
        
        for connection in self.connections.data:
            s = connection.mutate()['summary']
            if s != "Nothing":
                if history["summary"] == "Nothing":
                    history["summary"] = ""
                history["summary"] += f"\n{connection} mutate: {s}"
            
        return history

    """
    * creates a new genome.
     * g1 should have the higher score
     *  - take all the genes of a
     *  - if there is a genome in a that is also in b, choose randomly
     *  - do not take disjoint genes of b
     *  - take excess genes of a if they exist
     return Genome
    """
    @staticmethod
    def crossover(g1, g2, genome, probability_preserve_disabled=0.75):
        """ 
        g1 has to be the genome with highest score.
        """
        
        # genome is a "empty" base genome with existing or no nodes

        index_g1 = 0
        index_g2 = 0

        # Connections have to be sorted for this alignment
        # genome.getSingletonContainers add the node if not exists
        while index_g1 < g1.connections.size() and index_g2 < g2.connections.size():

            connection1:Connection = g1.connections.get(index_g1)
            connection2:Connection = g2.connections.get(index_g2)

            in1:int = connection1.innovation_number
            in2:int = connection2.innovation_number

            if in1 == in2:
                # homologous gene
                connection_inherit:Connection = None
                if random.random() < 0.5:
                    # create or retrieve containers if exist
                    neuronFrom, neuronTo = genome.getSingletonContainers(connection1)
                    connection_inherit = Connection.copy(connection1,neuronFrom, neuronTo)
                else:
                    neuronFrom, neuronTo = genome.getSingletonContainers(connection2)
                    connection_inherit = Connection.copy(connection2,neuronFrom, neuronTo)

                if not connection1.enabled.value or not connection2.enabled.value:
                    #paper: "There was a n% chance that an inherited gene was disabled if it was disabled in either parent."
                    if connection_inherit.enabled.value and random.random() < probability_preserve_disabled:
                        connection_inherit.enabled.value = False

                genome.insertConnection(connection_inherit)

                index_g1 += 1
                index_g2 += 1
            elif in1 > in2:
                # disjoint gene of b not in a. Preserve "not put in a"
                index_g2 += 1                
            else:
                # disjoint gene of a not in b. Preserve "put in a"
                neuronFrom, neuronTo = genome.getSingletonContainers(connection1)
                connection_inherit = Connection.copy(connection1,neuronFrom, neuronTo)
                genome.insertConnection(connection_inherit)
                index_g1 += 1


        while index_g1 < g1.connections.size():
            connection1:Connection = g1.connections.get(index_g1)
            # create or retrieve neurons
            neuronFrom, neuronTo = genome.getSingletonContainers(connection1)
            connection_inherit = Connection.copy(connection1,neuronFrom, neuronTo)

            genome.insertConnection(connection_inherit)
            index_g1 += 1
            
        
        # the mutation rates may have been modified 
        for mutation,rate in g1.mutation_rates.items():
            genome.mutation_rates[mutation] = rate


        return genome

    def setScore(self,score):
        self.score = score
        self.key = score
        
    def __str__(self)->str:

        output:str = wc("green", f"|Score: {round(self.score,2)} Nodes: {len(self.neurons)}|\t")
        for connection in self.connections.data:
            output += f"{connection}\t"
        return output
        