import numpy as np
def Rastrigin(X):
    sum = 0
    n = len(X)
    for i in range(n):
        sum = sum + X[i].value ** 2 - 3 * np.cos(2 * np.pi * X[i].value)
    return sum + 3 * n