import numpy as np
import matplotlib.pyplot as plt 
from collections import OrderedDict

class SGD:
    def __init__(self,lr=0.05):
        self.lr=lr
    def update(self, params, grad):
        for key in params.keys():
            params[key] -= self.lr * grad[key]


class Momentum:
    def __init__(self,lr=0.05,momentum=0.09):
        self.lr=lr
        self.m=momentum
        self.v=None
    def update(self,params,grad):
        if self.v==None:
            self.v={}
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)
        for key in params.keys():
            self.v[key]=self.m*self.v[key]-grad[key]*self.lr
            params[key]+=self.v[key]

class AdaGrad:
    def __init__(self,lr=0.05):
        self.lr=lr
        self.h=None
    def update(self,params,grad):
        if self.h==None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)
        for key in params.keys():
            self.h[key]+=grad[key]*grad[key]
            params[key]-=self.lr*grad[key]/np.sqrt(self.h[key]+1e-7)
            
class RMSprop:
    def __init__(self,lr=0.05,beta=0.5):
        self.lr=lr
        self.beta=beta
        self.s=None
    def update(self,params,grad):
        if self.s==None:
            self.s={}
            for key,val in params.items():
                self.s[key]=np.zeros_like(val)
        for key in params.keys():
            self.s[key]=self.s[key]*self.beta+(1-self.beta)*grad[key]*grad[key]
            params[key]-=self.lr*grad[key]/np.sqrt(self.s[key]+1e-7)

class Adam:
    def __init__(self,lr=0.4,beta1=0.8,beta2=0.8):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.iter=0
        self.m=None
        self.s=None
    def update(self,params,grad):
        if self.s==None:
            self.s,self.m={},{}
            for key,val in params.items():
                self.s[key]=np.zeros_like(val)
                self.m[key]=np.zeros_like(val)
        self.iter+=1

        lr_t=self.lr
        for key in params.keys():
            self.m[key]=self.m[key]*self.beta1+(1.0-self.beta1)*grad[key]
            self.s[key]=self.s[key]*self.beta2+(1.0-self.beta2)*grad[key]*grad[key]
            params[key]-=lr_t*(self.m[key]/(1.0-self.beta1**self.iter))/np.sqrt((self.s[key]/(1.0-self.beta2**self.iter))+1e-7)

class Adam2:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
def func(x,y):
    return x**2/20+y**2

def dfunc(x,y):
    return x/10,2*y

if __name__=="__main__":
    params=OrderedDict()
    grad=OrderedDict()
    x_his=[-7.0]
    y_his=[3.0]
    params['x'],params['y']=x_his[0],y_his[0]
    grad['x'],grad['y']=None,None
    cycle_num=25
    # optimizer=AdaGrad(2.5)
    # optimizer=RMSprop(0.5,0.3)
    # optimizer=Momentum(0.2,0.9)
    # optimizer=SGD(0.9)
    optimizer=Adam(0.3)
    # optimizer=Adam2(0.3)
    for i in range(cycle_num):
        grad['x'],grad['y']=dfunc(params['x'],params['y'])
        optimizer.update(params,grad)
        x_his.append(params['x'])
        y_his.append(params['y'])

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = func(X, Y)
    
    # for simple contour line  
    # mask = Z > 7
    # Z[mask] = 0
    # plot 
    plt.plot(x_his, y_his, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    # colorbar()
    #spring()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
