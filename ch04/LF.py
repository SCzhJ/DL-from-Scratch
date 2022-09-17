import numpy as np
def main_square_error(y,t):
    return 0.5*np.sum((y-t)**2)
def cross_entrophy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))
def cross_entrophy_error_oh(y,t):
    if y.ndim == 1:
       t=t.reshape(1,t.size)
       y=y.reshape(1,y.size)
    if t.size == y.size:
        t=t.argmax(axis=1)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

