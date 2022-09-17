import numpy as np
def num_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)
def eg_func(x):
    return x**2

def func2(x):
    return np.sum(x**2)

def num_grad(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp=x[i]
        x[i]=tmp+h
        fxh1=f(x)
        x[i]=tmp-h
        fxh2=f(x)
        grad[i]=(fxh1-fxh2)/(2*h)
        x[i]=tmp
    return grad

def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)
        x[idx]=tmp_val-h
        fxh2=f(x)
        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
        it.iternext()
    return grad
