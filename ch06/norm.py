import numpy as np

m=10
l = np.random.rand(m)
print(l)
avg=np.sum(l)/m
print(avg)
sigsq=np.sum((l-avg)**2)/avg
print("sigsq:"+str(sigsq))
x=(l-avg)/np.sqrt(sigsq)
print(x)
print(np.sum(x))
