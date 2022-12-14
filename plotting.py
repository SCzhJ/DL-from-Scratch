import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
#生成数据
x = np.arange(0,6,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
#绘制图形
plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle="--",label="cos")
plt.xlabel("x")
plt.xlabel("y")
plt.title("sin&cos")
plt.legend()

plt.show()
img = imread("lena.jpg")
plt.imshow(img)
plt.show()
