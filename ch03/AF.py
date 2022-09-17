import numpy as np
import matplotlib.pyplot as plt

def step_func(x):
    return np.array(x>0,dtype=np.int32)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    exp_x_sum = np.sum(exp_x)
    return exp_x/exp_x_sum

if __name__ == "__main__":
    x = np.arange(-5.0,5.0,0.1)
    y = step_func(x)
    plt.plot(x,y,linestyle="--")
    y2 = sigmoid(x)
    plt.plot(x,y2,linestyle="--")
    y3 = relu(x)
    plt.plot(x,y3)
    plt.ylim(-0.1,1.1)
    plt.show()
