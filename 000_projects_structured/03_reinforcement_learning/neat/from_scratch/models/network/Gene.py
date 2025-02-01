from ..data_structures.HashSorted import *
from random import gauss, uniform

class GeneAttributeBase(object):
    value = None # any
    def __init__(self,mutate_rate):
        self.mutate_rate = mutate_rate

class GeneAttributeFloat(GeneAttributeBase):
    
    def __init__(
        self,
        mutate_rate = 0.8,
        value = None,
        init_mean = 0,
        init_stdev = 1,
        init_type = "gaussian",
        replace_rate = 0.1,
        mutate_power = 0.5,
        min_value = 0,
        max_value = 1
    ):   
        super().__init__(mutate_rate)
        self.init_mean:float = init_mean
        self.init_stdev:float =init_stdev
        self.init_type:str = init_type
        self.replace_rate:float =replace_rate
        self.mutate_power:float =mutate_power
        self.min_value:float =min_value
        self.max_value:float =max_value
        if value is None:
            self.init_value()
        else:
            self.value = value
        
    def copy(self):
        
        return GeneAttributeFloat(
            self.mutate_rate,
            self.value,
            self.init_mean,
            self.init_stdev,
            self.init_type,
            self.replace_rate,
            self.mutate_power,
            self.min_value,
            self.max_value
        )
        
    def clamp(self, value):
        return max(min(value, self.max_value), self.min_value)

    def init_value(self):

        if ('gauss' in self.init_type) or ('normal' in self.init_type):
            g = gauss(self.init_mean, self.init_stdev)
            self.value = self.clamp(g)

        if 'uniform' in self.init_type:
            min_value = max(self.min_value, (self.init_mean - (2 * self.init_stdev)))
            max_value = min(self.max_value, (self.init_mean + (2 * self.init_stdev)))
            self.value = uniform(min_value, max_value)

    def mutate(self):
        history = {"summary": "Nothing"}
        
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        r = random.random()
        if r < self.mutate_rate:
            step:float = gauss(0.0, self.mutate_power)
            value_new = self.clamp(self.value + step)
            if value_new != self.value:
                history["summary"] = f"Changed from {round(self.value,2)} to {round(value_new,2)}"
            self.value = value_new
            
        elif r < self.mutate_rate + self.replace_rate:
            value_last = self.value
            self.init_value()
            
            history["summary"] = f"Reboot from {round(value_last,2)} to {round(self.value,2)}"

        return history

class GeneAttributeBool(GeneAttributeBase):
    
    def __init__(
        self,
        mutate_rate = 0.01,
        value = None,
        default = "True",
        rate_to_true_add = 0.001,
        rate_to_false_add = 0.001,
    ):   
        super().__init__(mutate_rate)
        self.default:str = str(default).lower()
        self.rate_to_true_add:float =rate_to_true_add
        self.rate_to_false_add:float =rate_to_false_add
        if value is None:
            self.init_value()
        else:
            self.value = value
        
    def copy(self):
        return GeneAttributeBool(
            self.mutate_rate,
            self.value,
            self.default,
            self.rate_to_true_add,
            self.rate_to_false_add
        )
    
    def init_value(self):
        if self.default in ('1', 'on', 'yes', 'true'):
            self.value = True
        elif self.default in ('0', 'off', 'no', 'false'):
            self.value = False
        elif self.default in ('random', 'none'):
            self.value = bool(random() < 0.5)

    def mutate(self):
        history = {"summary": "Nothing"}
        mutate_rate = self.mutate_rate

        if self.value:
            mutate_rate += self.rate_to_false_add
        else:
            mutate_rate += self.rate_to_true_add

        if mutate_rate > 0:
            r = random.random()
            if r < mutate_rate:
                # may change is not garanteed
                value_new = random.random() < 0.5
                if value_new != self.value:
                    history["summary"] = f"Changed from {self.value} to {value_new}"
                self.value = value_new

        return history
    
    
class Gene(HashStructure):
    # inherith key
    innovation_number:int = 0
    def __init__(self,innovation_number):
        self.innovation_number = innovation_number
        
    
    

