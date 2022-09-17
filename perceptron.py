import numpy as np
class perceptron:
	def __init__(self,x1,x2):
		self.x1 = x1
		self.x2 = x2
	def AND(self):
		x = np.array([self.x1,self.x2])
		w = np.array([0.5,0.5])
		b = -0.5
		if np.sum(w*x)+b>0:
			return 1
		else:
			return 0 
	def NAND(self):
		x = np.array([self.x1,self.x2])
		w = np.array([-0.5,-0.5])
		b = 0.7
		if np.sum(w*x)+b>0:
			return 1
		else:
			return 0	
	def OR(self):
		x = np.array([self.x1,self.x2])
		w = np.array([1,1])
		b = -0.7
		if np.sum(w*x)+b>0:
			return 1
		else:
			return 0	
	def XOR(self):
		s1 = self.NAND()
		s2 = self.OR()
		perc2 = perceptron(s1,s2)
		return perc2.AND()

if __name__ == "__main__":
	x1 = int(input("please input x1:"))
	x2 = int(input("please input x2:"))
	perc = perceptron(x1,x2)
	print("AND:"+str(perc.AND()))
	print("NAND:"+str(perc.NAND()))
	print("OR:"+str(perc.OR()))
	print("XOR:"+str(perc.XOR()))
