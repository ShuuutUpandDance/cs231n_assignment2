import numpy as np

x = np.ones((3,1,1,1))
y = x.reshape(x.shape[0],-1)
x = x.transpose(0,1,2,3)
print x.shape
