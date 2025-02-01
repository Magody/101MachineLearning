from .Gene import Gene, GeneAttributeFloat
from ..Colors import wc

class EnumConnectionTypes:
    TYPE_UNKNOWN = -1
    TYPE_INPUT = 0
    TYPE_HIDDEN = 1
    TYPE_OUTPUT = 2
    
class ConfigNeuron:
           
    def __init__(
        self,
        bias_mutate_rate = 0.7,
        bias_value = None,
        bias_init_mean = 0,
        bias_init_stdev = 1,
        bias_init_type = "gaussian",
        bias_replace_rate = 0.1,
        bias_mutate_power = 0.5,
        bias_min_value = -2,
        bias_max_value = 2,
        response_mutate_rate = 0.0,
        response_value = None,
        # initial value is 1
        response_init_mean = 1,
        response_init_stdev = 0,
        response_init_type = "gaussian",
        response_replace_rate = 0.0,
        response_mutate_power = 0.0,
        response_min_value = -1,
        response_max_value = 1
        
    ):
        self.attribute_reference_bias = GeneAttributeFloat(
            bias_mutate_rate, 
            bias_value, 
            bias_init_mean, 
            bias_init_stdev, 
            bias_init_type, 
            bias_replace_rate, 
            bias_mutate_power, 
            bias_min_value, 
            bias_max_value
        )
        
        self.attribute_reference_response = GeneAttributeFloat(
            response_mutate_rate, 
            response_value, 
            response_init_mean, 
            response_init_stdev, 
            response_init_type, 
            response_replace_rate, 
            response_mutate_power, 
            response_min_value, 
            response_max_value
        )
        
    def copy(self):
        return ConfigNeuron(
            self.attribute_reference_bias.mutate_rate, 
            self.attribute_reference_bias.value, 
            self.attribute_reference_bias.init_mean, 
            self.attribute_reference_bias.init_stdev, 
            self.attribute_reference_bias.init_type, 
            self.attribute_reference_bias.replace_rate, 
            self.attribute_reference_bias.mutate_power, 
            self.attribute_reference_bias.min_value, 
            self.attribute_reference_bias.max_value,
            self.attribute_reference_response.mutate_rate, 
            self.attribute_reference_response.value, 
            self.attribute_reference_response.init_mean, 
            self.attribute_reference_response.init_stdev, 
            self.attribute_reference_response.init_type, 
            self.attribute_reference_response.replace_rate, 
            self.attribute_reference_response.mutate_power, 
            self.attribute_reference_response.min_value, 
            self.attribute_reference_response.max_value
        )
        
        

class Neuron(Gene):
    ### GLOBAL/STATIC VARIABLES
    map_neuron_innovation_number:dict = dict() # type <innovation_number:int, sample_x_y:int[2]>. Example: m[2] = (0.1,0.212121)
    
    ### GENE ATTRIBUTES
    bias:GeneAttributeFloat = None
    response:GeneAttributeFloat = None
    
    ### OTHER
    # type_neuron/x: 0.1->input node, 0.9->output node, other->hidden node
    # if we want to plot the position of this neuron this will be in (x,y)
    x:float = 0
    # y: just a helper to determine its position in network
    y:float = 0
    
    
    activationFunction = None # type function
    output:float = 0

    # private
    _configNeuron = None
    
    def __init__(self,x:float,y:float,innovation_number:int,activationFunction, configNeuron:ConfigNeuron):
        super().__init__(innovation_number)
        self.x = x
        self.key = x
        self.y = y
        self.output:float = 0
        self.bias = configNeuron.attribute_reference_bias.copy()
        self.response = configNeuron.attribute_reference_response.copy()
        self._configNeuron = configNeuron.copy()

        self.activationFunction = activationFunction

    def mutate(self):
        history = {"summary": ""}
        if self.getNeuronType() == EnumConnectionTypes.TYPE_INPUT:
            return history
            
        history["summary"] += f"BIAS:{self.bias.mutate()['summary']}"
        s = self.response.mutate()['summary']
        if s != "Nothing":
            history["summary"] += f". RESPONSE:{s}" 
        return history
        
    def distance(self, other, compatibility_weight_coefficient):
        d = abs(self.bias.value - other.bias.value) + abs(self.response.value - other.response.value)
        if self.activationFunction != other.activationFunction:
            d += 1.0
        return d * compatibility_weight_coefficient
    
    @staticmethod
    def copy(neuron):
        neuron_copy = Neuron(neuron.x,neuron.y,neuron.innovation_number,neuron.activationFunction, neuron._configNeuron)
        neuron_copy.output = neuron.output
        neuron_copy.bias = neuron.bias.copy()
        neuron_copy.response = neuron.response.copy()
        return neuron_copy
        
    def getNeuronType(self)->int:

        x_destiny:float = self.x
        # assume the connection is from input/hidden to hidden
        connection_type:int = EnumConnectionTypes.TYPE_HIDDEN
        
        if x_destiny == 0.1:
            connection_type = EnumConnectionTypes.TYPE_INPUT
        elif x_destiny == 0.9:
            # the connection comes from input/hidden to output
            connection_type = EnumConnectionTypes.TYPE_OUTPUT

        return connection_type



    def equals(self, object):
        if not type(object) is Gene:
            return False
        
        return self.innovation_number == object.innovation_number

    def __str__(self):
        type_neuron = "I"
        if self.x > 0.1 and self.x < 0.9:
            type_neuron = "H"
        elif self.x == 0.9:
            type_neuron = "O"
            
        string_bias = ""
        if type_neuron != "I":
            string_bias = wc('cyan', '[' + str(round(self.bias.value,1)) + ']')
        return f"{wc('blue', str(self.innovation_number))}{string_bias}{type_neuron}"

    def hashCode(self):
        return self.innovation_number

    @staticmethod
    def getNeuronNew(x:float,y:float,innovation_number:int,activation_function, configNeuron,exist:bool=False):
        n:Neuron = Neuron(x,y,innovation_number,activation_function, configNeuron)
        if not exist:
            # no exist, so add it to storage
            Neuron.map_neuron_innovation_number[innovation_number] = (n.x,n.y,n.activationFunction,n._configNeuron)
        return n

    @staticmethod
    def getNeuron(innovation_number_expected:int):
        # singleton for x and y for unique innovation_number        

        len_neurons_created:int = len(Neuron.map_neuron_innovation_number)


        innovation_number:int = len_neurons_created + 1
        if innovation_number_expected <= len_neurons_created:
            # in the history this gene was already created
            innovation_number = innovation_number_expected
            sample = Neuron.map_neuron_innovation_number[innovation_number]
            x = sample[0]
            y = sample[1]
            activation_function = sample[2]
            configNeuron = sample[3]
            return Neuron.getNeuronNew(x,y,innovation_number,activation_function,configNeuron,True)

        else:
            raise Exception("Unknown neuron: Use getNeuronNew instead")

        
