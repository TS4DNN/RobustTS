import numpy as np

a = np.random.choice(np.arange(10000), int(10000/2), replace=False)
print(len(a))
print(a)
b = np.delete(np.arange(10000), a)
print(len(b))
print(b)

