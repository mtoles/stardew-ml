import numpy as np

x = np.array([1,1,1])
y = np.array([2,3,4])
z = x @ np.transpose(y)
a = np.transpose(x) @ y
print(z)