import matplotlib.pyplot as plt
import numpy as np
import diff as df
def gradient_descent(func,init_x,lr=0.02,step_num=100):
    x = init_x
    x_his = []
    for i in range(step_num):
        x_his.append(x.copy())
        gradient = df.num_grad(func,x)
        x -= lr*gradient
    return x,np.array(x_his)

if __name__=="__main__":
    learning_rate=float(input("please input a learning rate:"))
    x0=float(input("please input x0:"))
    x1=float(input("please input x1:"))
    x=np.array([x0,x1])
    print(x)
    new_x,x_his = gradient_descent(df.func2,x,lr=learning_rate)
    print(new_x)
    plt.plot([-5,5],[0,0],'--b')
    plt.plot([0,0],[-5,5],'--b')
    plt.plot(x_his[::5,0],x_his[::5,1],'o')
    plt.xlim(-3.5,3.5)
    plt.ylim(-4.5,4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
