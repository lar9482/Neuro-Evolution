from enum import Enum
import math

class Type(Enum):
    Input = 0
    Hidden = 1
    Output = 2

class node:
    def __init__(self, type = Type.Input, id = 0):
        self.type = type
        self.value = 0
        self.activated_value = 0

        #Node id to identify nodes in the future
        self.id = id

    def feed(self, weight, value):
        self.value += weight*value

    def activate_node(self):
        self.activated_value = self.__tanh(self.value)

    def reset_node(self):
        self.value = 0
        self.activated_value = 0

    def __sigmoid(self, x):
        if (x > 50):
            return 0.99999
        elif (x < -50):
            return 0.00001
        else:
            return (math.exp(x)) / (1 + math.exp(x))
        
    def __tanh(self, x):
        if (x > 50):
            return 0.99999
        elif (x < -50):
            return -0.99999
        else:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
            
    #Less than 'operator' for comparison of 'nodes'
    def __lt__(self, other):
        return self.id <= other.id