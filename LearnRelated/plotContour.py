import numpy as np
import matplotlib.pyplot as plt
xlist=np.linspace(-3.0,3.0,100)
ylist=np.linspace(-3.0,3.0,100)
X,Y = np.meshgrid(xlist,ylist)
Z=np.sqrt(X**2,Y**2)
fig,ax=plt.subplots(1,1)
cp=ax.contourf(X,Y,Z)
fig.colorbar(cp)
ax.set_title('Filled Contours Plot')
#ax.set_x_laber
ax.set_ylabel('y(cm)')
plt.show()

