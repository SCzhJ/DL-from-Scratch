import numpy as np

class Sigmoid:
    def __init__(self):
        self.y = None
    def forward(self,x):
        self.y = 1/(1+np.exp(-x))
        return self.y
    def backward(self,dout):
        return dout*(1.0-self.y)*self.y
