import math

class ActivationFunction:

    @staticmethod
    def tanh(x:float):
        """Tanh activation"""
        return math.tanh(x)
    
    @staticmethod
    def clamp(x,value_min,value_max):
        return max(min(x,value_max), value_min)
        
    @staticmethod
    def sigmoid(x:float):
        # default activation is sigmoid
        x_clamp = ActivationFunction.clamp(x,-20,20)
        return 1 / (1 + math.exp(-x_clamp))

    
    @staticmethod
    def relu(x:float):
        # default activation is sigmoid
        return x if x > 0 else 0

    @staticmethod
    def sigmoid_steepened(x:float, step=4.9):
        # steepened sigmoid: allows more fine tuning at extreme activations. 
        # It's optimized to be close to linear during its steepest ascent between activations
        x_clamp = ActivationFunction.clamp(x,-10,10)
        return 1 / (1 + math.exp(-step*x_clamp))
