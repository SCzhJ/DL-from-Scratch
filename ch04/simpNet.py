import LF
import diff
import gd
import sys,os
sys.path.append(os.pardir)
import numpy as np
from ch03.AF import softmax

class simpNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = LF.cross_entrophy_error_oh(y,t)
        return loss
