import random
from .Activation import *
from .Neuron import *
from .Connection import *
from ..data_structures.HashSorted import *

class Network(HashStructure):


    def __init__(
        self,
        input_neurons:list,
        hidden_neurons:list,
        output_neurons:list,
        activationFunctionHidden = ActivationFunction.relu
    ):
        self.activationFunctionHidden = activationFunctionHidden
        # replica for fast access
        self.neurons:dict = dict() # mapping <innovation_number:int,neuron:Neuron>
        for neuron in input_neurons:
            self.neurons[neuron.innovation_number] = neuron
        for neuron in hidden_neurons:
            self.neurons[neuron.innovation_number] = neuron
        for neuron in output_neurons:
            self.neurons[neuron.innovation_number] = neuron

        # redundancy, more RAM for faster proccessing
        self.connections:HashSorted = HashSorted(use_key=True) # use innovation number to hash, not the object
        # neurons sorted by x for forward and graphs steps
        self.input_neurons:HashSorted = HashSorted()
        self.hidden_neurons:HashSorted = HashSorted()
        self.output_neurons:HashSorted = HashSorted()

        for neuron in input_neurons:
            self.input_neurons.addSorted(neuron)
        for neuron in hidden_neurons:
            self.hidden_neurons.addSorted(neuron)
        for neuron in output_neurons:
            self.output_neurons.addSorted(neuron)

        # Input just for hashing. type: hashmap<to_neuron:int,list[Connection]> 
        self.input_connections:dict = {neuron.innovation_number:[] for neuron in self.input_neurons.data}
        # type: hashmap<to_neuron:int,list[Connection]> 
        self.hidden_connections:dict = {neuron.innovation_number:[] for neuron in self.hidden_neurons.data}
        # type: hashmap<to_neuron:int,list[Connection]> 
        self.output_connections:dict = {neuron.innovation_number:[] for neuron in self.output_neurons.data} # type: hashmap<to_neuron:int,list[Connection]>


    def getSingletonContainers(self, connection:Connection):
        # O(1)
        neuronFromContainer:Neuron = None
        neuronToContainer:Neuron = None

        if self.existNeuronInnovationNumber(connection.neuronFrom.innovation_number):
            neuronFromContainer = self.neurons[connection.neuronFrom.innovation_number]
        else:
            neuronFromContainer = Neuron.copy(connection.neuronFrom)
            self.insertNeuron(neuronFromContainer)

        if self.existNeuronInnovationNumber(connection.neuronTo.innovation_number):
            neuronToContainer = self.neurons[connection.neuronTo.innovation_number]
        else:
            neuronToContainer = Neuron.copy(connection.neuronTo)
            self.insertNeuron(neuronToContainer)

        return neuronFromContainer, neuronToContainer

    def existNeuronInnovationNumber(self, innovation_number:int):
        # O(1)
        return innovation_number in self.neurons

    def existConnection(self, connection:Connection):
        # innovation_number is important here
        # we only check if innovation_Number have been seen

        return self.connections.contains(connection)

    def insertNeuron(self,neuron:Neuron)->None:
        if self.existNeuronInnovationNumber(neuron.innovation_number):
            return

        neuron_type:int = neuron.getNeuronType()

        self.neurons[neuron.innovation_number] = neuron

        if neuron_type == EnumConnectionTypes.TYPE_INPUT:
            self.input_neurons.addSorted(neuron)
        elif neuron_type == EnumConnectionTypes.TYPE_HIDDEN:
            self.hidden_neurons.addSorted(neuron)     
        elif neuron_type == EnumConnectionTypes.TYPE_OUTPUT:
            self.output_neurons.addSorted(neuron)            
        else:
            raise Exception(f"Not supported connection type: {neuron.x}")
        
        assert len(self.neurons) == (self.input_neurons.size()+self.hidden_neurons.size()+self.output_neurons.size())
        

    def removeNeuron(self,neuron:Neuron)->None:
        if not self.existNeuronInnovationNumber(neuron.innovation_number):
            return

        neuron_type:int = neuron.getNeuronType()

        del self.neurons[neuron.innovation_number]

        if neuron_type == EnumConnectionTypes.TYPE_INPUT:
            self.input_neurons.remove(neuron)
        elif neuron_type == EnumConnectionTypes.TYPE_HIDDEN:
            self.hidden_neurons.remove(neuron)   
        elif neuron_type == EnumConnectionTypes.TYPE_OUTPUT:
            self.output_neurons.remove(neuron)        
        else:
            raise Exception(f"Not supported connection type: {neuron.x}")
        
        assert len(self.neurons) == (self.input_neurons.size()+self.hidden_neurons.size()+self.output_neurons.size())
        
    def insertConnection(self,connection:Connection)->None:
       
        connection_type:int = connection.getConnectionType()
        self.connections.addSorted(connection)

        if connection_type == EnumConnectionTypes.TYPE_HIDDEN:
            if not connection.neuronTo.innovation_number in self.hidden_connections:
                self.hidden_connections[connection.neuronTo.innovation_number] = []   
            self.hidden_connections[connection.neuronTo.innovation_number].append(connection)
            
        elif connection_type == EnumConnectionTypes.TYPE_OUTPUT:
            if not connection.neuronTo.innovation_number in self.output_connections:
                self.output_connections[connection.neuronTo.innovation_number] = []   

            self.output_connections[connection.neuronTo.innovation_number].append(connection)
            
        else:
            raise Exception(f"Not supported connection type: {connection.neuronTo} {connection_type}")        

        

    def removeConnection(self,connection:Connection)->None:
        # connection: should have the same object reference
       
        connection_type:int = connection.getConnectionType()

        if connection_type == EnumConnectionTypes.TYPE_HIDDEN:
            self.hidden_connections[connection.neuronTo.innovation_number].remove(connection)
        elif connection_type == EnumConnectionTypes.TYPE_OUTPUT:
            self.output_connections[connection.neuronTo.innovation_number].remove(connection)
        else:
            raise Exception(f"Not supported connection type: {connection.neuronTo} {connection_type}")

        self.connections.remove(connection)
        
        
        
    def checkCongruencyConnections(self):
        total_connections_hidden = 0
        total_connections_output = 0
        
        for key,conns in self.hidden_connections.items():
            total_connections_hidden += len(conns)
            
        
        for key,conns in self.output_connections.items():
            total_connections_output += len(conns)
        
        
        return self.connections.size() == (total_connections_hidden+total_connections_output)
            

    def getRandomNeuron(self):

        innovation_numbers:list = list(self.neurons.keys())
        len_neurons:int = len(innovation_numbers)

        if len_neurons == 0:
            return None

        selected:int = innovation_numbers[random.randint(0,len_neurons-1)]
        return self.neurons[selected]

    def getRandomGenePair(self):
        families:list = [0,1,2]
        a:Neuron = self.getRandomNeuron()
        b:Neuron = None

        family:int = -1
        if a.x == 0.1:
            family = 0
        elif a.x == 0.9:
            family = 2
        else:
            # is hidden
            family = 1
        if family != 1:
            families.remove(family)
            
        other_family = families[random.randint(0,len(families)-1)]

        if other_family == 1:
            front_neurons = []
            for h in self.hidden_neurons.data:
                if h.x <= a.x:
                    continue
                front_neurons.append(h)
                    
            len_front_hidden = len(front_neurons)
            
            if len_front_hidden == 0:
                # Can not connect to "nothing" and can not connect to itself
                families.remove(other_family)
                other_family = families[random.randint(0,len(families)-1)]
            else:
                b = front_neurons[random.randint(0,len_front_hidden-1)]

        if other_family == 0:
            b = self.input_neurons.getRandomElement()
        
        if other_family == 2:
            b = self.output_neurons.getRandomElement()

        
        return a,b

    def getRandomConnection(self):
        return self.connections.getRandomElement()

    def forward(self,input:list)->list:
        # input double
        len_input_neurons = self.input_neurons.size()
        len_hidden_neurons = self.hidden_neurons.size()
        len_output_neurons = self.output_neurons.size()

        if len(input) != len_input_neurons:
            raise Exception("Data doesn't fit")
        
        # Forward input (feed)
        for i in range(len_input_neurons):
            self.input_neurons.get(i).output = input[i]

        # forward hidden in order by x
        for i in range(len_hidden_neurons):
            hidden_neuron:Neuron = self.hidden_neurons.get(i)
            hidden_neuron_output:float = 0
            
            if hidden_neuron.innovation_number in self.hidden_connections:
                # otherwise:
                # Hidden_neuron may be connected to output or nothing
                # Exist connection from another hidden or input neuron before this
                hidden_neuron_connections:list = self.hidden_connections[hidden_neuron.innovation_number]
                
                for j in range(len(hidden_neuron_connections)):
                    c:Connection = hidden_neuron_connections[j]
                    if c.enabled:
                        hidden_neuron_output += c.weight.value * c.neuronFrom.output
            
            hidden_neuron.output = hidden_neuron.activationFunction(hidden_neuron.bias.value + hidden_neuron.response.value * hidden_neuron_output)

        # forward output layer

        output = [0 for _ in range(len_output_neurons)]
        for i in range(len_output_neurons):
            output_neuron:Neuron = self.output_neurons.get(i)
            output_neuron_output:float = 0
            
            if output_neuron.innovation_number in self.output_connections:
                # Exist connection from another hidden or input neuron before this
                output_neuron_connections:list = self.output_connections[output_neuron.innovation_number]            
                
                for j in range(len(output_neuron_connections)):
                    c:Connection = output_neuron_connections[j]
                    if c.enabled:
                        output_neuron_output += c.weight.value * c.neuronFrom.output
            
            output_neuron.output = output_neuron.activationFunction(output_neuron.bias.value + output_neuron.response.value * output_neuron_output)
            output[i] = output_neuron.output

        return output