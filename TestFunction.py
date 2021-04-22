import numpy as np
def Rastrigin(X, n=20):
    sum = 0
    for i in range(n):
        sum = sum + X[i].value ** 2 - 3 * np.cos(2 * np.pi * X[i].value)
    return sum + 3 * n