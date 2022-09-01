import numpy as np

correct_array = np.array([0 for i in range(10)])
index = np.array([1, 2, 3])

correct_array[index] = 1
print(correct_array)
